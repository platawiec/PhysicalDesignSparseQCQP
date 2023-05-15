# This replicates Fig. 4 in the paper "Many Physical Design Problems are Sparse QCQPs"
# In particular, we aim to replicate Fig. 4c (the optimized design) as well as the
# returned optimal point. The implementation follows section 4 of the paper

using PhysicalDesignSparseQCQP
using JuMP, Ipopt
using UnicodePlots
using LinearAlgebra, SparseArrays

## Define the problem

# Material refractive index
n = 2.3 + 0.03im
design_dielectric = n^2
# Size of design domain, in wavelengths
design_domain = 4
# Phase of reflected light
reflected_phase = -0.3π
r_target = cis(reflected_phase)

# PML

# Simulation options
N = design_domain * 100

m = 4
d = 20
R = 9e-1
σmax = -(m+1)*log(R)/(2*d)
pml = PML(d, σmax, m)

model = NormalIncidenceFDFD1D(
    design_domain,
    design_dielectric,
    N,
    pml,
    TotalEH()
)

# quick model test
Lχ1, Lχ2 = build_design_pdes(model)
source = fill(0.0im, 2(N + 2pml.N))
source[pml.N+1] = 1 
source[pml.N+1+(N+2pml.N)] = im 
ψ1 = Lχ1 \ source
ψ2 = Lχ2 \ source

lineplot(real(ψ1[1:end÷2]))
lineplot(real(ψ2[1:end÷2]))

## Define the QCQP
# Eq. 39
# Objective function
# max real(E_scattered[idx_monitor]' * r_target)
# note: don't really need to worry about normalization
# There are two approaches we can take to launching the
# incident field:
# 1. enforce unit amplitude in J current
# 2. Launch eigenmode by setting electric and magnetic currents
# Option (1) allows us the smallest PDE and number of constraints,
# cutting it down by factor of 2 because we will only solve over
# the E-field. However, the current source creates a counter-
# propagating wave we will have to subtract from our objective
# Option (2) allows us to express the reflected field directly
# from the result, but requires a larger PDE, in addition to the
# constraints that are required to be implemented
# max real((E_total[idx_monitor] - E_in[idx_monitor])' * r_target)
# max real(E_total[idx_monitor]' * r_target - E_in[idx_monitor]' * r_target)
# E_in is independent of the design variables, so we see that it
# contributes only a constant factor, and we are left with
# max real(E_total[idx_monitor]' * r_target)
# as the same maximum as the scattered field

# Design constraint
# (𝐋(χ1)ψ - ξ)† 𝐃ᵢ(𝐋(χ2)ψ - ξ) = 0 i ∈ I_design
# (𝐋(χi)ψ - ξ)† 𝐃ᵢ(𝐋(χi)ψ - ξ) = 0 i ∈ I_PML
# (𝐋(χi)ψ - ξ) = 0 i ∈ I_PML

(;D, Lχ1, Lχ2, ξ, Id, Ipml, Im) = build_component_constraints(model)
N_T = size(Lχ1, 1)

m = Model(Ipopt.Optimizer)
@variable(m, ψ[1:N_T] in ComplexPlane())
@constraint(m, c[i=Id], ((Lχ1*ψ - ξ)' * D[i] * (Lχ2*ψ - ξ)) == 0)
@constraint(m, cpml[i=Ipml], (Lχ1*ψ - ξ)' * D[i] * (Lχ1*ψ - ξ) .== 0)
#@constraint(m, (Lχ1*ψ - ξ)[Ipml, :] .== 0)

@objective(m, Max, real(dot(r_target, ψ[only(Im)])))

# Interlude: attempt solve w/ Ipopt
optimize!(m)
solution_summary(m)

ψ_result = value.(ψ)
lineplot(real(ψ_result))

design_result = derive_two_level_design((; Lχ1, Lχ2, D, ξ, Id, Ipml, Im), ψ_result)

lineplot(design_result)
lineplot(real(ψ_result .- ψ1))

abs2.(only(ψ_result[Im]))
abs2.(only(ψ1[Im]))
abs2.(only(ψ1[Im]) - only(ψ_result[Im]))

Lχd = rebuild_design_pde(model, design_result)
isapprox(Lχd \ ξ, ψ_result, rtol=1e-4)

# Define the SDP
# Tested:
using COSMO
m = Model(
    COSMO.Optimizer,
)
set_attribute(m, "merge_strategy", COSMO.CliqueGraphMerge)
set_attribute(m, "decompose", true)
set_attribute(m, "time_limit", 60.0)
# 
# using Hypatia
# m = Model(Hypatia.Optimizer)
#
# Segfault
# using Clarabel
# m = Model(Clarabel.Optimizer)
#
# Error
# using CSDP
# m = Model(CSDP.Optimizer)
#
# really weird StackOverflow
# using ProxSDP
@variable(m, X[1:N_T+1, 1:N_T+1] in HermitianPSDCone())

β = fill(0.0im, N_T)
β[Im] .= 1.0#r_target
A0 = Hermitian([fill(0.0im, N_T, N_T) β/2; β'/2 0.0im])
@objective(m, Max, tr(Hermitian(A0 * X)))


Ai = map(1:N_T) do i
    Bi = Lχ1' * D[i] * Lχ2
    vcol = -Lχ1' * D[i] * ξ
    vrow = -ξ' * D[i] * Lχ2
    return [Bi/2 vcol; vrow 0.0im]
end
Ai_re = map(1:N_T) do i
    # Lχ1 and Lχ2 have same values at Id and Ipml, don't need if/else
    Bi = Lχ1' * D[i] * Lχ2 + (Lχ1' * D[i] * Lχ2)'
    # ((Lχ1*ψ - ξ)' * D[i] * (Lχ2*ψ - ξ))
    # = ψ' Lχ1' D[i] Lχ2 ψ - ψ' Lχ1' D[i] ξ - ξ' D[i] Lχ2 ψ + ξ' D[i] ξ
    # = ψ' Bi ψ + Re(vi' ψ)
    # = (ψ s)' [Bi vi/2; vi'/2 0] (ψ s) = ψ' Bi ψ + ψ' vi/2 s + s' vi'/2 ψ = ψ' Bi ψ + Re(vi' ψ)
    # TODO: check math, add other constraint
    vi = -Lχ2' * D[i] * ξ - Lχ1' * D[i] * ξ
    return [Bi/2   vi/2; vi'/2 0.0im]
end
Ai_im = map(1:N_T) do i
    # Lχ1 and Lχ2 have same values at Id and Ipml, don't need if/else
    Bi = Lχ1' * im * D[i] * Lχ2 + (Lχ1' * im * D[i] * Lχ2)'
    # ((Lχ1*ψ - ξ)' * D[i] * (Lχ2*ψ - ξ))
    # = ψ' Lχ1' D[i] Lχ2 ψ - ψ' Lχ1' D[i] ξ - ξ' D[i] Lχ2 ψ + ξ' D[i] ξ
    # = ψ' Bi ψ + Re(vi' ψ)
    # TODO: check math, add other constraint
    vi = -Lχ2' * im * D[i] * ξ - Lχ1' * im * D[i] * ξ
    return [Bi/2   vi/2; vi'/2 0.0im]
end
Alin = map(Ipml) do i
    u = spzeros(N_T)
    u[i] = 1
    return [spzeros(N_T, N_T) Lχ1' * u / 2; u' * Lχ1 / 2 0.0]
end

γi = map(1:N_T) do i
    return ξ' * D[i] * ξ
end
γi_re = map(1:N_T) do i
    return real(ξ' * D[i] * ξ)
end
γi_im = map(1:N_T) do i
    return real(ξ' * im * D[i] * ξ)
end
δi = map(Ipml) do i
    u = spzeros(N_T)
    u[i] = 1
    return real(u' * ξ)
end

#A_set = [Ai_re; Ai_im; Alin_re; Alin_im]
#a_set = [γi_re; γi_im; δi_re; δi_im]
A_set = [Ai_re; Ai_im; Alin]
a_set = [γi_re; γi_im; δi]

check_constraints = map(eachindex(A_set)) do i
    trial_constraint = abs(tr(Hermitian(A_set[i] * ([ψ_result; 1] * [ψ_result; 1]')))) - a_set[i]
    if trial_constraint > 1e-8
        @show (i, trial_constraint)
    end
    return trial_constraint
end
#A_set = [Ai; Alin]
#a_set = [γi; δi]
@constraint(m, c_set[i=1:length(A_set)], tr(Hermitian(A_set[i] * X)) .== a_set[i])
# slack variable magnitude == 1
@constraint(m, X[end, end] == 1)

set_attribute(m, "time_limit", 5*60.0)
set_attribute(m, "max_iter", 20000)
optimize!(m)
solution_summary(m)

ψ_result = svd(value.(X)).U[:, 1][1:end-1]
lineplot(real(ψ_result))
design_result = derive_two_level_design((; Lχ1, Lχ2, D, ξ, Id, Ipml, Im), ψ_result)

lineplot(design_result)
Lχd = rebuild_design_pde(model, design_result)
ψ_result = Lχd \ ξ
lineplot(real(ψ_result))

abs2.(only(ψ_result[Im]))
abs2.(only(ψ1[Im]))
abs2.(only(ψ1[Im]) - only(ψ_result[Im]))
# Solve to optimal via majorization-minimization algorithm

γ_hp = 1e-6 # from supplemental
@objective(m, Max, tr(LinearAlgebra.Hermitian(A0 * X)) + γ_hp * LinearAlgebra.tr(X))
set_optimizer_attribute(m, "max_iter", 40000)
optimize!(m)
solution_summary(m)

ψ_result = svd(value.(X)).U[:, 1][1:end-1]
lineplot(real(ψ_result))
design_result = derive_two_level_design((; Lχ1, Lχ2, D, ξ, Id, Ipml, Im), ψ_result)

lineplot(design_result)Lχd = rebuild_design_pde(model, design_result)
ψ_result = Lχd \ ξ
lineplot(real(ψ_result))

abs2.(only(ψ_result[Im]))
abs2.(only(ψ1[Im]))
abs2.(only(ψ1[Im]) - only(ψ_result[Im]))