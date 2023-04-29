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
    design_dielectric
    N
    pml
end

struct PML
    N
    σ_max
    m
end

"""
    function build_component_constraints(model)

Builds and returns the sparse matrices which correspond to the model, as used for optimization
"""
function build_component_constraints(model)
    Lχ1, Lχ2 = build_design_pdes(model)
    N_T = model.N + 2model.pml.N
    ξ = fill(0.0im, N_T)
    D = map(1:N_T) do i
        Di = fill(0.0, N_T, N_T)
        Di[i, i] = 1.0
        return Di
    end

    ξ[model.pml.N] = 1.0
    Id = collect((model.pml.N+1):(model.pml.N+1+model.N))
    Ipml = vcat(collect(1:model.pml.N), collect((model.pml.N+model.N+2):(model.N + 2model.pml.N)))
    Im = [model.pml.N]
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
    pml = model.pml
    N_T = model.N + 2*pml.N
    ω = 1.0
    Δ = model.design_domain / (model.N-1)
    mw = spdiagm(-1 => fill(1.0/Δ, N_T-1), 0 => fill(-2.0/Δ, N_T), 1 => fill(1.0/Δ, N_T-1))
    χd = map(1:N_T) do i
        if i < pml.N
            return 1.0 - im * pml.σ_max * (pml.N - i)^(pml.m)
        elseif i >= N_T - pml.N
            return 1.0 - im * pml.σ_max * (i - N_T - pml.N)^(pml.m)
        else
            if design_binary[i] == 1
                return complex(model.design_dielectric)
            else
                return complex(1.0)
            end
        end
    end
    Lχd = -mw - spdiagm(complex.(χd) * ω^2)
    return Lχd
end

function build_design_pdes(model)
    pml = model.pml
    N_T = model.N + 2*pml.N
    ω = 1.0
    Δ = model.design_domain / (model.N-1)
    mw = spdiagm(-1 => fill(1.0/Δ, N_T-1), 0 => fill(-2.0/Δ, N_T), 1 => fill(1.0/Δ, N_T-1))
    χ1 = map(1:N_T) do i
        if i < pml.N
            return 1.0 - im * pml.σ_max * (pml.N - i)^(pml.m)
        elseif i >= N_T - pml.N
            return 1.0 - im * pml.σ_max * (i - N_T - pml.N)^(pml.m)
        else
            return 1.0 + 0.0im
        end
    end
    χ2 = map(1:N_T) do i
        if i < pml.N
            return 1.0 - im * pml.σ_max * (pml.N - i)^(pml.m)
        elseif i >= N_T - pml.N
            return 1.0 - im * pml.σ_max * (i - N_T + pml.N)^(pml.m)
        else
            return complex(model.design_dielectric)
        end
    end
    Lχ1 = -mw - spdiagm(complex.(χ1) * ω^2)
    Lχ2 = -mw - spdiagm(complex.(χ2) * ω^2)
    return Lχ1, Lχ2
end


end
