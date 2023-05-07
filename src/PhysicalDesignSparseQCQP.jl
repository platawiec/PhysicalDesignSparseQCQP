module PhysicalDesignSparseQCQP

using LinearAlgebra, SparseArrays

export NormalIncidenceFDFD1D, PML, build_component_constraints, build_design_pdes, derive_two_level_design, rebuild_design_pde

"""
    NormalIncidenceFDFD1D

Model for Maxwell's equations for 1D scattering problems with normal incidence
The `design_domain` is taken to be the entire scattering region

# Fields:
- `design_domain`: Size of design domain, in wavelengths
- `design_dielectric``: Choice of dielectric to design under
- `N`: Number of points within design domain
- `pml`: PML settings
"""
struct NormalIncidenceFDFD1D
    design_domain
    buffer_domain
    design_dielectric
    N
    pml
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
function build_component_constraints(model)
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
    Im = [N_T - model.pml.N - 1]
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
    Lχd = mw - dielectric_op(model, design_binary .* model.dielectric)
    return Lχd
end

function build_design_pdes(model)
    mw = mw_op(model)
    Lχ0 = mw - dielectric_op(model, 1.0)
    Lχd = mw - dielectric_op(model, model.dielectric)
    return Lχ0, Lχd
end

function mw_op(model)
    pml = model.pml
    N_T = length_model_grid(model)
    Δ = model.design_domain / (model.N-1)
    mw = spdiagm(-1 => fill(-1.0/Δ, N_T-1), 0 => fill(2.0/Δ, N_T), 1 => fill(-1.0/Δ, N_T-1))
    return mw
end

function dielectric_op(model, dielectric::Number)
    return dielectric_op(model, fill(dielectric, model.N + 2*model.pml.N))
end

function dielectric_op(model, dielectric)
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
    return spdiagm(χ * ω^2)
end


end
