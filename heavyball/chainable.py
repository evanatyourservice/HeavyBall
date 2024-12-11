import functools
import random
from typing import Optional, Union

import torch

from . import utils

balance_probability: float = 0.01


def _key_in_state(state, key):
    if isinstance(key, str):
        return key in state
    for k in key:
        if isinstance(k, (tuple, list)):
            continue
        if k not in state:
            return False
    return True


def _inplace_guard_(state, key, template_fn):
    key_not_in_state = not _key_in_state(state, key)
    if key_not_in_state:
        template_fn()
    return key_not_in_state


def _guard_in_state(state, key, template_fn):
    if not _key_in_state(state, key):
        state[key] = template_fn()
    return state[key]


class FunctionTransform:
    def __init__(self, fn):
        self.fn = fn
        self.fn_name = self.get_fn().__name__

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        raise NotImplementedError

    def get_fn(self):
        if hasattr(self.fn, 'get_fn'):
            return self.fn.get_fn()
        return self.fn

    def val_name(self, name):
        return f"{self.fn_name}_{name}"


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key,
                           lambda: torch.zeros_like(ref, dtype=torch.float32, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get('storage_dtype', "float32")
    return getattr(torch, dtype)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn)
        self.names = names

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        vars = [[_zero_guard(state(p), self.val_name(name), p, _storage_dtype(group)) for p in param]  #
                for name in self.names]
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class CopyGuard(FunctionTransform):
    def __init__(self, fn, index, names):
        super().__init__(fn)
        self.index = index
        self.names = names

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        val = [update, grad, param, *args][self.index]
        vars = [[_guard_in_state(state(p), self.val_name(name), lambda: torch.clone(v)) for p, v in zip(param, val)]  #
                for name in self.names]
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class GeneralGuard(FunctionTransform):  # We can't guard against reuse in the general case
    def __init__(self, fn, names, init_fn):
        super().__init__(fn)
        self.names = names
        self.init_fn = init_fn

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        vars = []
        skip_update = False
        for p, g, u in zip(param, grad, update):
            st = state(p)
            skip_update |= _inplace_guard_(st, self.names, lambda: self.init_fn(st, group, u, g, p, **kwargs))
            vars.append([st[name] if isinstance(name, str) else st.get(name[0], name[1]) for name in self.names])
        if skip_update:
            raise SkipUpdate
        return self.fn(state, group, update, grad, param, *args, *zip(*vars), **kwargs)


class NoState(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        return self.fn(group, update, grad, param, *args, **kwargs)


class NoStateNoForeach(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        for a in zip(update, grad, param, *args):
            return self.fn(group, *a, **kwargs)


def zero_guard(*names):
    return functools.partial(ZeroGuard, names=names)


def copy_guard(index, *names):
    return functools.partial(CopyGuard, index=index, names=names)


def general_guard(*names, init_fn):
    return functools.partial(GeneralGuard, names=names, init_fn=init_fn)


def no_state(fn):
    return NoState(fn)


def no_state_no_foreach(fn):
    return NoStateNoForeach(fn)


class SkipUpdate(ValueError):
    pass


@zero_guard("exp_avg")
@no_state
def exp_avg(group, update, grad, param, exp_avg):
    return utils.scale_by_exp_avg_(exp_avg, update, utils.beta_debias(utils.get_beta1(group), group["step"]))


@zero_guard("exp_avg_sq")
@no_state
def scale_by_exp_avg_sq(group, update, grad, param, exp_avg_sq):
    return utils.scale_by_exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group["step"]),
                                      group['eps'])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adam_(exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group['step'],  #
                       group['eps'])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adam_(param, exp_avg, exp_avg_sq, update, grad, utils.get_beta1(group), utils.get_beta2(group),
                      group['step'], group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.laprop_(exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group['step'])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_laprop_(param, exp_avg, exp_avg_sq, update, grad, utils.get_beta1(group), utils.get_beta2(group),
                        group['step'], group['lr'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@copy_guard(2, "z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    group['weight_sum'] = utils.schedule_free_(group['lr'], group['weight_lr_power'], group.get('weight_sum', 0),
                                               utils.get_beta1(group), param, z, update, group['r'], group['step'],
                                               group['weight_decay'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group['step'] == 1:
        utils.exp_avg_sq_(exp_avg_sq, update, 0, 1)
        raise SkipUpdate

    if group['step'] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group['eps'])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group['step']), 1)
        raise SkipUpdate

    utils.fused_adopt_(param, update, grad, exp_avg_sq, exp_avg, utils.get_beta1(group), utils.get_beta2(group),
                       group['step'] - 2, group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group['step'] == 1:
        utils.exp_avg_sq_(exp_avg_sq, update, 0, 1)
        raise SkipUpdate

    if group['step'] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group['eps'])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group['step']), 1)
        raise SkipUpdate

    return utils.adopt(update, exp_avg_sq, exp_avg, utils.get_beta1(group), utils.get_beta2(group), group['step'] - 2)


def _init_soap(state, group, update, grad, param):
    utils.init_preconditioner(grad, state, utils.get_beta2(group), group['max_precond_dim'], group['precondition_1d'])


def _init_psgd(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q, state["exprs"] = utils.init_Q_exprs(grad, group['precond_init_scale'], group['max_size_triangular'],
                                           group['min_ndim_triangular'], group['memory_save_mode'],
                                           dtype=getattr(torch, group['q_dtype']))
    state["Q"] = utils.triu_to_line(Q) if group['store_triu_as_line'] else Q

    if not cached:
        return

    state['Q_cache'] = [torch.empty_like(q) for q in Q]

    expr = [f'{c.upper()}{c}' if q_.ndim == 2 else c for c, q_ in zip(utils.einsum_base, Q)]
    expr = ','.join(expr)
    grad_expr = ''.join(c for c, _ in zip(utils.einsum_base, grad.shape))
    out_expr = ''.join(c.upper() if c.upper() in expr else c for c in grad_expr)
    expr = f'{expr},{grad_expr}->{out_expr}'

    state['cache_expr'] = expr


def precond_schedule(group, prob: Union[callable, float, None] = None, name: str = 'cumulative_prob'):
    step = group['step']
    if 'precondition_frequency' in group:
        return step > 0 and step % group['precondition_frequency'] == 0
    rng = random.Random(0x172381 ^ step)
    if 'precond_scheduler' in group:
        return utils.precond_schedule(step, group['precond_scheduler'], rng)
    if prob is not None:
        return utils.psgd_should_update(group, prob, rng, name=name)
    raise ValueError("No preconditioner update schedule specified.")


@no_state_no_foreach
def orthogonalize_update(group, update, grad, param, scale_mode: str = "scale"):  # explore scale_mode="graft"
    if update.dim() == 1:
        return update
    original_shape = update.shape
    # doing it this way, as tmp and update are not guaranteed to share memory address or layout
    tmp = update.flatten(1, -1)
    utils.inplace_orthogonal_(tmp, utils.zeroth_power_mode, tmp, scale_mode)
    return tmp.reshape(original_shape)


@zero_guard("momentum")
@no_state
def nesterov_momentum(group, updates, grads, params, momentum):
    return utils.nesterov_momentum(momentum, updates, utils.get_beta1(group))


@zero_guard("momentum")
@no_state
def heavyball_momentum(group, updates, grads, params, momentum):
    return utils.heavyball_momentum(momentum, updates, utils.get_beta1(group))


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
    update = utils.promote(update)

    grad_projected = [utils.project(u, q, False) for u, q in zip(update, Q)]
    precond = utils.adam_(exp_avg, exp_avg_sq, grad_projected, utils.get_beta1(group), utils.get_beta2(group),
                          utils.scalar_guard(group['step'], exp_avg[0]))
    precond = [utils.project(p, q, False) for p, q in zip(precond, Q)]

    for u, q, gg, eas in zip(update, Q, GG, exp_avg_sq):
        utils.update_preconditioner(u, q, gg, eas, group['max_precond_dim'], group['precondition_1d'],
                                    utils.beta_debias(group['shampoo_beta'], group['step']), precond_schedule(group))
    return precond


def _update_psgd_precond(group, param, grad, Q_mat, Q, exprs, prob: Optional[callable] = None):
    if prob is None:
        prob = utils.precond_update_prob_schedule()
    if not precond_schedule(group, prob, name=f"cumulative_prob_{id(Q)}"):
        return

    Q = [utils.promote(q_) for q_ in Q]
    utils.psgd_update_precond(Q_mat, exprs, grad, group['precond_lr'], Q, group['store_triu_as_line'])

    if grad.dim() > 1 and precond_schedule(group, balance_probability, "balance_prob"):
        if group['store_triu_as_line']:
            utils.psgd_balance_Q([q_ for _, q_ in Q])
        else:
            utils.psgd_balance_Q(Q)


def _update_psgd_cache(cached, Q_cache, q):
    if not cached:
        return q

    for c_, q_ in zip(Q_cache, q):
        if q_.ndim == 2:
            torch.matmul(q_.T, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)
    return Q_cache


def _cached_psgd_precond_grad(cached, cache_expr, exprs, update, Q_mat, Q_cache):
    if cached:
        return utils.precond_grad_cached_(cache_expr, update, *Q_cache)
    return utils.psgd_precond_grad(exprs[-1], update, *Q_mat)


def _fused_cached_psgd_precond_grad(group, grad, param, cached, cache_expr, exprs, update, Q_mat, Q_cache):
    if cached:
        utils.fused_precond_grad_cached_(cache_expr, update, param, group['lr'], grad, group['weight_decay'],
                                         group['caution'], *Q_cache)
    else:
        utils.fused_psgd_precond_grad(exprs[-1], update, param, group['lr'], grad, group['weight_decay'],
                                      group['caution'], *Q_mat)


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd)
@no_state_no_foreach
def scale_by_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                  prob: Optional[callable] = None):
    old = update
    update = update.to(memory_format=torch.contiguous_format)
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    _update_psgd_precond(group, param, update, Q_mat, Q, exprs, prob)
    out = _cached_psgd_precond_grad(cached, cache_expr, exprs, update, Q_mat, Q_cache)
    return torch.as_strided(out, old.shape, old.stride())


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd)
@no_state_no_foreach
def scale_by_delayed_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                          prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    precond = _cached_psgd_precond_grad(cached, cache_expr, exprs, update, Q_mat, Q_cache)
    _update_psgd_precond(group, param, update, Q_mat, Q, exprs, prob)
    return precond


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd)
@no_state_no_foreach
def update_by_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                   prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    _update_psgd_precond(group, param, update, Q_mat, Q, exprs, prob)
    _fused_cached_psgd_precond_grad(group, update, param, cached, cache_expr, exprs, update, Q_mat, Q_cache)
    raise SkipUpdate


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd)
@no_state_no_foreach
def update_by_delayed_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                           prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    _fused_cached_psgd_precond_grad(group, update, param, cached, cache_expr, exprs, update, Q_mat, Q_cache)
    _update_psgd_precond(group, param, update, Q_mat, Q, exprs, prob)
    raise SkipUpdate


def palm_beta2(state, group, update, grad, param):
    beta2 = 1 - group['step'] ** -group['beta2_scale']
    group['betas'] = (utils.get_beta1(group), beta2)
    return update


def apply_to_idx(fn, idx):
    def _fn(state, group, update, grad, param):
        args = [state, group, update, grad, param]
        return fn(args[idx])

    return _fn


def chain(state: Union[callable, dict], group, grad, param, *fns):
    update = [torch.clone(g, memory_format=torch.preserve_format) for g in grad]
    skip_update = False
    for fn in fns:
        try:
            update = fn(state, group, update, grad, param)
        except SkipUpdate:
            skip_update = True
            continue
        if update is None:
            break
    if not skip_update and update is not None:
        utils.update_param_(param, update, group['lr'], group['weight_decay'], caution=group['caution'], grad=grad)


class ChainOpt(utils.StatefulOptimizer):
    def __init__(self, params, defaults, foreach: bool, *fns):
        super().__init__(params, defaults, foreach)
        self.fns = tuple(fns)

    def _step(self, group):
        if 'base_lr' not in group:
            group['base_lr'] = group['lr']
        step = group['step'] = group.get('step', 0) + 1
        if group['warmup_steps'] and step < group['warmup_steps']:
            group['lr'] = -group['base_lr'] * step / group['warmup_steps']
        else:
            group['lr'] = -group['base_lr']

        vals = list(self.split_p_and_g_in_group(group, should_promote=False, beta1=utils.get_beta1(group)))
        if not vals:
            return
        p, g = zip(*vals)

        if not group['foreach'] or len(p) == 1:
            for param, grad in zip(p, g):
                chain(self.state_, group, [grad], [param], *self.fns)
            return

        chain(self.state_, group, g, p, *self.fns)


use_default = object()
str_or_fn = Union[Any]


def _get_clip_fn(name: str_or_fn, default_val: str_or_fn):
    name = default(name, default_val)
    if callable(name):
        return name
    elif name not in ('l2_clip_', 'rmsnorm_clip_', 'trust_region_clip_', 'a_law_compress', 'mu_law_compress'):
        raise ValueError(f"Clipping function {name} not found")
    return getattr(utils, name)


def default(a, b):
    return b if a is None or a is use_default else a


# not supported: update_by_schedule_free, scale_by_soap, scale_by_exp_avg_sq
_scale_to_update_map = {scale_by_delayed_psgd: update_by_delayed_psgd,  #
                        scale_by_psgd: update_by_psgd,  #
                        scale_by_adam: update_by_adam,  #
                        scale_by_laprop: update_by_laprop,  #
                        scale_by_adopt: update_by_adopt}


class BaseOpt(ChainOpt):
    gradient_clipping: str_or_fn = None
    update_clipping: str_or_fn = None
    palm: bool = False
    auto_fuse: bool = True
    compile_step: bool = False

    def __init__(self, params, defaults, foreach: bool, gradient_clipping: str_or_fn, update_clipping: str_or_fn,
                 palm: bool = use_default, *fns):
        if default(update_clipping, self.update_clipping) is None:
            if fns and self.auto_fuse:
                args, kwargs = None, None
                fn = fns[-1]
                if isinstance(fn, functools.partial):
                    fn, args, kwargs = fns[-1].func, fns[-1].args, fns[-1].keywords
                if fn in _scale_to_update_map:
                    fn = _scale_to_update_map[fn]
                    if args is not None:
                        fn = functools.partial(fn, *args, **kwargs)
                    fns = tuple(fns)[:-1] + (fn,)
        else:
            if any(fn in (update_by_adopt, update_by_adam, update_by_laprop, update_by_schedule_free) for fn in fns):
                raise ValueError("`update_by` functions do not support update clipping. Use `scale_by`")

        fns = tuple(fns)

        if default(palm, self.palm):
            fns = (palm_beta2,) + fns
        if default(gradient_clipping, self.gradient_clipping) is not None:
            fns = (apply_to_idx(gradient_clipping, 2),) + fns
        if default(update_clipping, self.update_clipping) is not None:
            fns = fns + (apply_to_idx(update_clipping, 2),)

        super().__init__(params, defaults, foreach, *fns)


class ScheduleFree(BaseOpt):
    def eval(self):
        for group in self.param_groups:
            group['train_mode'] = train_mode = not group.get('train_mode')
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        # Set p.data to x
                        z = utils.promote(state['z'])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        utils.copy_stochastic_(p.data, p32)

    def train(self):
        for group in self.param_groups:
            group['train_mode'] = train_mode = not group.get('train_mode')
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        z = utils.promote(state['z'])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        utils.copy_stochastic_(p.data, p32)
