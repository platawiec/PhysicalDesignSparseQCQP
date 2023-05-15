module PhysicalDesignSparseQCQP

using LinearAlgebra, SparseArrays

export NormalIncidenceFDFD1D, PML, build_component_constraints, build_design_pdes, derive_two_level_design, rebuild_design_pde
export TotalE, ScatteredE, TotalEH

struct TotalE end
struct ScatteredE end
struct TotalEH end
"""
    NormalIncidenceFDFD1D

Model for Maxwell's equations for 1D scattering problems with normal incidence
The `design_domain` is taken to be the entire scattering region

# Fields:
- `design_domain`: Size of design domain, in wavelengths
- `design_dielectric``: Choice of dielectric to design under
- `N`: Number of points within design domain
- `pml`: PML settings
- `formulation`: Type of model (`TotalE`, `ScatteredE`, or `TotalEH`)
"""
struct NormalIncidenceFDFD1D
    design_domain
    design_dielectric
    N
    pml
    formulation
end

struct PML
    N
    σ_max
    m
end

function length_model_grid(model)
    return model.N + 2*model.pml.N
end

"""
    function build_component_constraints(model)

Builds and returns the sparse matrices which correspond to the model, as used for optimization
"""
build_component_constraints(model) = _build_component_constraints(model, model.formulation)
function _build_component_constraints(model, ::TotalE)
    Lχ1, Lχ2 = build_design_pdes(model)
    N_T = length_model_grid(model)
    ξ = fill(0.0im, N_T)
    D = map(1:N_T) do i
        Di = spzeros(N_T, N_T)
        Di[i, i] = 1.0
        return Di
    end

    ξ[model.pml.N+1] = 1.0
    Id = collect((model.pml.N+1):(model.pml.N+1+model.N))
    Ipml = vcat(collect(1:model.pml.N), collect((model.pml.N+model.N+2):(model.N + 2model.pml.N)))
    Im = [model.pml.N + 1]
    return (; Lχ1, Lχ2, D, ξ, Id, Ipml, Im)
end

function _build_component_constraints(model, ::TotalEH)
    Lχ1, Lχ2 = build_design_pdes(model)
    N_T = length_model_grid(model)
    ξ = fill(0.0im, 2N_T)
    D = map(1:N_T) do i
        Di = spzeros(2N_T, 2N_T)
        Di[i, i] = 1.0
        Di[i+N_T, i+N_T] = 1.0
        return Di
    end

    ξ[model.pml.N+1] = 1.0
    ξ[model.pml.N+1+N_T] = im
    Id = collect((model.pml.N+1):(model.pml.N+1+model.N))
    Ipml = vcat(collect(1:model.pml.N), collect((model.pml.N+model.N+2):N_T))
    Im = [model.pml.N+1]
    return (; Lχ1, Lχ2, D, ξ, Id, Ipml, Im)
end

function derive_two_level_design(design_params, ψ)
    (; Lχ1, Lχ2, D, ξ, Id, Ipml, Im) = design_params
    design = map(enumerate(D)) do (i, Di)
        r1 = sum(abs, (Lχ1*ψ - ξ)' * Di)
        r2 = sum(abs, (Lχ2*ψ - ξ)' * Di)
        if i in Id
            if r2 < r1
                return 1
            else
                return 0
            end
        end
        return 0
    end
    return design
end

function rebuild_design_pde(model, design_binary)
    mw = mw_op(model)
    Lχd = mw + dielectric_op(model, 1 .+ (design_binary) .* (model.design_dielectric - 1))
    return Lχd
end

function build_design_pdes(model)
    mw = mw_op(model)
    Lχ0 = mw + dielectric_op(model, 1.0)
    Lχd = mw + dielectric_op(model, model.design_dielectric)
    return Lχ0, Lχd
end

mw_op(model) = _mw_op(model, model.formulation)
function _mw_op(model, ::TotalE)
    N_T = length_model_grid(model)
    Δ = model.design_domain / (model.N-1)
    mw = spdiagm(-1 => fill(-1.0/Δ, N_T-1), 0 => fill(2.0/Δ, N_T), 1 => fill(-1.0/Δ, N_T-1))
    return mw
end

function _mw_op(model, ::TotalEH)
    N_T = length_model_grid(model)
    Δ = model.design_domain / (model.N-1)
    curle = spdiagm(0 => fill(-1.0/Δ, N_T), 1 => fill(1.0/Δ, N_T-1))
    curlh = spdiagm(-1 => fill(-1.0/Δ, N_T-1), 0 => fill(1.0/Δ, N_T))
    mw = [spzeros(N_T, N_T) curle; curlh spzeros(N_T, N_T)]
    return mw
end

function dielectric_op(model, dielectric::Number)
    return dielectric_op(model, fill(dielectric, model.N + 2*model.pml.N))
end

dielectric_op(model, dielectric) = _dielectric_op(model, dielectric, model.formulation)
function _dielectric_op(model, dielectric, ::TotalE)
    pml = model.pml
    N_T = length_model_grid(model)
    ω = 1.0
    χ = map(enumerate(dielectric)) do (i, d)
        if i < pml.N
            return 1.0 - im * pml.σ_max * (pml.N - i)^(pml.m)
        elseif i >= N_T - pml.N
            return 1.0 - im * pml.σ_max * (i - N_T + pml.N)^(pml.m)
        else
            return complex(d)
        end
    end
    return spdiagm(-χ * ω^2)
end

function _dielectric_op(model, dielectric, ::TotalEH)
    pml = model.pml
    N_T = length_model_grid(model)
    ω = 2pi
    χ = map(enumerate(dielectric)) do (i, d)
        if i < pml.N
            return 1.0 - im * pml.σ_max * (pml.N - i)^(pml.m)
        elseif i >= N_T - pml.N
            return 1.0 - im * pml.σ_max * (i - N_T + pml.N)^(pml.m)
        else
            return complex(d)
        end
    end
    return spdiagm(im * [χ; fill(1.0, N_T)] * ω)
end

end
