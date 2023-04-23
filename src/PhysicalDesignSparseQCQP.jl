module PhysicalDesignSparseQCQP

using SparseArrays

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

"""
    function build_component_constraints(model)

Builds and returns the sparse matrices which correspond to the model, as used for optimization
"""
function build_component_constraints(model)
    Lχ1 = build_design_pde(model)
    Lχ2 = build_bare_pde(model)
    D = build_design_indicator(model)
    ξ = build_source_currents(model)
end

end
