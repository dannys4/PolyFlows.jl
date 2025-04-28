export LinearInterpolant, McCannInterpolant, SqrtInterpolant, SquareInterpolant, TrigInterpolant

abstract type AbstractStochInterpolant end

struct LinearInterpolant{A,B,Adot,Bdot,E,V} <: AbstractStochInterpolant
    alpha::A
    beta::B
    alpha_deriv::Adot
    beta_deriv::Bdot
    eta::E
    velocity::V
    function LinearInterpolant(alpha::_A, beta::_B, alpha_deriv::_Adot, beta_deriv::_Bdot, eta::_E, velocity::_V) where {_A,_B,_Adot,_Bdot,_E,_V}
        new{_A,_B,_Adot,_Bdot,_E,_V}(alpha, beta,alpha_deriv,beta_deriv,eta,velocity)
    end
end

"""
    LinearInterpolant(α, β, α̇, β̇, [tol], [η, α̇/α])
Forms the linear interpolant \$X_t = \\alpha(t) X_1 + \\beta(t) X_0\$.
See [[1]](https://arxiv.org/abs/2409.08861) for definition of \$\\eta\$.
"""
function LinearInterpolant(alpha, beta, alpha_deriv, beta_deriv, tol=0.)
    velocity = t->alpha_deriv(t)/(alpha(t)+tol)
    eta = t->beta(t)*(velocity(t)*beta(t) - beta_deriv(t))
    LinearInterpolant(alpha, beta, alpha_deriv, beta_deriv, eta, velocity)
end

"""See [`LinearInterpolant`](@ref). α = t, β = 1-t"""
McCannInterpolant() = LinearInterpolant(t->t, t->1-t, Returns(1.), Returns(-1.), t->(t+1e-2)/(1-t+1e-3), t->1/(t+1e-2))
"""See [`LinearInterpolant`](@ref). α = √t, β = 1-√t"""
SqrtInterpolant(tol=1e-3) = LinearInterpolant(t->sqrt(t), t->1-sqrt(t), t->1/(2sqrt(t)+tol), t->-1/(2sqrt(t)+tol), tol)
"""See [`LinearInterpolant`](@ref). α = t², β = 1-t²"""
SquareInterpolant(tol=1e-3) = LinearInterpolant(t->t*t, t->1-t*t, t->2t, t->-2t, tol)
"""See [`LinearInterpolant`](@ref). α = sin(πt/2), β = cos(πt/2)"""
function TrigInterpolant(tol=1e-3)
    alpha = t->sin(pi*t/2)
    beta = t->cos(pi*t/2)
    alpha_dot = t->pi*cos(pi*t/2)/2
    beta_dot = t->-pi*sin(pi*t)/2
    LinearInterpolant(alpha, beta, alpha_dot, beta_dot, tol)
end