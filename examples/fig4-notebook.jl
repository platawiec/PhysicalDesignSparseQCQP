### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ cbf56fe2-ec6a-11ed-18b2-c9c14774aec3
begin
	using Pkg;
	Pkg.activate(".")
end

# ╔═╡ 44df06fe-a3e6-4a25-8ff3-44be761e0996
using PhysicalDesignSparseQCQP, Plots, LinearAlgebra, SparseArrays, JuMP, Ipopt, COSMO

# ╔═╡ 23b84746-6951-4d4b-b32c-87b9bbdb36f9
md"""
# Introduction

This notebook attempts to replicate Fig. 4 of the paper:
> Gertler, Shai, et al. "Many Physical Design Problems are Sparse QCQPs." arXiv preprint arXiv:2303.17691 (2023).

In particular, we do the following:
1. Define a PDE consistent with the problem statement (1D Maxwell's equation at normal incidence, choosing to work solely in the E-field in a total field formulation)
2. Construct an equivalent formulation as a QCQP, per the paper
3. Define objectives
4. Solve via a local solver (Ipopt) to validate the PDE and the constraints
5. Turn the QCQP into an SDP (§S.III)
6. Verify the constraints
7. Solve to global bound
8. Solve to rank-1 solution (§S.VI)
"""

# ╔═╡ cdbe36ff-fd7d-4f87-a8d3-83a3bcd9d98b
md"""
# Load Packages
"""

# ╔═╡ 08957e59-b5af-497c-80e4-f6e68bc4b5c2
md"""
# Problem Definition

The following parameters are taken from §III.

We make assumptions on the following:
- PML parameters (number of layers, polynomial grading scheme, and target reflectance)
- Resolution

Although we are using the reflected phase target from the text, the problem configuration isn't quite compatible. 
"""

# ╔═╡ 9877c619-efaa-4301-b3f7-44ccaa8abb48
n = 2.3 + 0.03im # Material refractive index

# ╔═╡ 6815721a-3dc6-438e-9a06-5711d5abf601
design_dielectric = n^2

# ╔═╡ e8c27ca8-e4e3-46b1-9eaa-ead1cae1dadf
design_domain = 4 # Size of design domain, in wavelengths

# ╔═╡ 8ba1e987-c1b1-4e94-a577-bf7bcb08c4bc
reflected_phase = -0.3π

# ╔═╡ 8bf1b346-38fb-4d90-b786-58cfe62492d2
r_target = cis(reflected_phase)

# ╔═╡ 720eb905-3eb6-49d7-b825-ad37edecb2d4
begin
	m = 4
	d = 20
	R = 5e-1
	σmax = -(m+1)*log(R)/(2*d)
	pml = PML(d, σmax, m)
end

# ╔═╡ 00a8d627-3fb9-4081-9fbf-e02792fb62cd
N = design_domain * 40

# ╔═╡ c4bac83c-0ac7-4a11-8394-1de7e53eb091
N_T = N + 2pml.N

# ╔═╡ f5b0b141-b214-4b79-b267-307c5cffed90
model = NormalIncidenceFDFD1D(
    design_domain,
    design_dielectric,
    N,
    pml
)

# ╔═╡ 42443748-97e2-49e7-a031-438917261149
md"""
# Solving the PDE
"""

# ╔═╡ 347e5b1d-141f-406e-8cfe-d6a59b1f0751
begin
	Lχ0_trial, Lχd_trial = build_design_pdes(model)
	source = fill(0.0im, N + 2pml.N)
	source[pml.N+1] = 1 
	ψ0 = Lχ0_trial \ source
	ψd = Lχd_trial \ source
end

# ╔═╡ 97ad3034-c16b-47aa-b643-b7b1f48743ce
begin
	ds = range(-pml.N*design_domain/N, design_domain+pml.N*design_domain/N, length=N + 2*pml.N)
	plot(ds, real(ψ0), label="Bare field")
	plot!(ds, real(ψd), label="Field with design dielectric")
end

# ╔═╡ a83353f8-f0c3-4bb5-b662-1e7c446cfa2e
(;D, Lχ1, Lχ2, ξ, Id, Ipml, Im) = build_component_constraints(model)

# ╔═╡ 41349f3e-6261-4c88-990d-df37fbd2e09c
begin
	m_qcqp = Model(Ipopt.Optimizer)
	@variable(m_qcqp, ψ[1:N_T] in ComplexPlane())
	@constraint(m_qcqp, c[i=Id], ((Lχ1*ψ - ξ)' * D[i] * (Lχ2*ψ - ξ)) == 0)
	@constraint(m_qcqp, cpml[i=Ipml], (Lχ1*ψ - ξ)' * D[i] * (Lχ1*ψ - ξ) .== 0)
	#@constraint(m, (Lχ1*ψ - ξ)[Ipml, :] .== 0)
	
	@objective(m_qcqp, Max, real(ψ[only(Im)]))
	md"""
	Here we define the JuMP model for the QCQP which is solved via a local optimizer

	We can then derive the design (we don't obtain it directly) by checking which constraint the field is compatible with at each point in the design domain.
	"""
end

# ╔═╡ 661067e0-5f05-410c-9e2b-911610c44e99
# ╠═╡ show_logs = false
begin
	optimize!(m_qcqp)
	solution_summary(m_qcqp);
end

# ╔═╡ 07eeb04a-9360-4229-8bad-e59b281d06d0
begin
	ψ_qcqp = value.(ψ)
	design_result = derive_two_level_design((; Lχ1, Lχ2, D, ξ, Id, Ipml, Im), ψ_qcqp)
	plot(
		heatmap(reshape(design_result, 1, :), legend=false),
		plot(real(ψ_qcqp), label="Re(E_T)"),
		layout=(2, 1)
	)
end

# ╔═╡ 2d3cedb5-867d-4934-b0eb-af89870e9805
begin
	m_sdp = Model(COSMO.Optimizer)
	set_attribute(m_sdp, "merge_strategy", COSMO.CliqueGraphMerge)
	set_attribute(m_sdp, "decompose", true)

	
	@variable(m_sdp, X[1:N_T+1, 1:N_T+1] in HermitianPSDCone())
		
	β = fill(0.0im, N_T)
	β[Im] .= 1.0#r_target
	Asdp = fill(0.0im, N_T+1, N_T+1)
	Asdp[Im, Im] .= 1.0
	#A0 = Hermitian([fill(0.0im, N_T, N_T) β/2; β'/2 0.0im])
	A0 = Hermitian(Asdp)
	@objective(m_sdp, Max, tr(Hermitian(A0 * X)))

	Ai_re = map(1:N_T) do i
	    # Lχ1 and Lχ2 have same values at Id and Ipml, don't need if/else
	    Bi = Lχ1' * D[i] * Lχ2 + (Lχ1' * D[i] * Lχ2)'
	    # ((Lχ1*ψ - ξ)' * D[i] * (Lχ2*ψ - ξ))
	    # = ψ' Lχ1' D[i] Lχ2 ψ - ψ' Lχ1' D[i] ξ - ξ' D[i] Lχ2 ψ + ξ' D[i] ξ
	    # = ψ' Bi ψ + Re(vi' ψ)
	    # = (ψ s)' [Bi vi/2; vi'/2 0] (ψ s) = ψ' Bi ψ + ψ' vi/2 s + s' vi'/2 ψ = ψ' Bi ψ + Re(vi' ψ)
	    vi = -Lχ2' * D[i] * ξ - Lχ1' * D[i] * ξ
	    return [Bi/2   vi/2; vi'/2 0.0im]
	end
	Ai_im = map(1:N_T) do i
	    Bi = Lχ1' * im * D[i] * Lχ2 + (Lχ1' * im * D[i] * Lχ2)'
	    vi = -Lχ2' * im * D[i] * ξ - Lχ1' * im * D[i] * ξ
	    return [Bi/2   vi/2; vi'/2 0.0im]
	end
	Alin = map(Ipml) do i
	    u = spzeros(N_T)
	    u[i] = 1
	    return [spzeros(N_T, N_T) Lχ1' * u / 2; u' * Lχ1 / 2 0.0]
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
	
	A_set = [Ai_re; Ai_im; Alin]
	a_set = [γi_re; γi_im; δi]
	@constraint(m_sdp, c_set[i=1:length(A_set)], tr(Hermitian(A_set[i] * X)) .== a_set[i])
	# slack variable magnitude == 1
	@constraint(m_sdp, X[end, end] == 1)
	md"""
	Here we define the JuMP model for the SDP
	"""
end

# ╔═╡ 0ae710f2-3848-4e0d-8567-c40556b64a04
# ╠═╡ show_logs = false
begin
	optimize!(m_sdp)
	solution_summary(m_sdp)
end

# ╔═╡ 30908df4-a962-489b-bcef-73855e608420
begin
	ψ_sdp = svd(value.(X)).U[:, 1][1:end-1]
	design_result_sdp = derive_two_level_design((; Lχ1, Lχ2, D, ξ, Id, Ipml, Im), ψ_sdp)
	Lχd = rebuild_design_pde(model, design_result)
	ψ_sdp_rebuilt = Lχd \ ξ
	p_ψ = plot(real(ψ_sdp), label="Re(E_T) (From SDP)")
	plot!(p_ψ, real(ψ_sdp_rebuilt), label="Re(E_T) (From SDP Design)")
	
	plot(
		heatmap(reshape(design_result_sdp, 1, :), legend=false),
		p_ψ,
		layout=(2, 1)
	)
end

# ╔═╡ Cell order:
# ╟─cbf56fe2-ec6a-11ed-18b2-c9c14774aec3
# ╟─23b84746-6951-4d4b-b32c-87b9bbdb36f9
# ╟─cdbe36ff-fd7d-4f87-a8d3-83a3bcd9d98b
# ╠═44df06fe-a3e6-4a25-8ff3-44be761e0996
# ╠═08957e59-b5af-497c-80e4-f6e68bc4b5c2
# ╟─9877c619-efaa-4301-b3f7-44ccaa8abb48
# ╟─6815721a-3dc6-438e-9a06-5711d5abf601
# ╟─e8c27ca8-e4e3-46b1-9eaa-ead1cae1dadf
# ╟─8ba1e987-c1b1-4e94-a577-bf7bcb08c4bc
# ╟─8bf1b346-38fb-4d90-b786-58cfe62492d2
# ╠═720eb905-3eb6-49d7-b825-ad37edecb2d4
# ╠═00a8d627-3fb9-4081-9fbf-e02792fb62cd
# ╠═c4bac83c-0ac7-4a11-8394-1de7e53eb091
# ╟─f5b0b141-b214-4b79-b267-307c5cffed90
# ╟─42443748-97e2-49e7-a031-438917261149
# ╠═347e5b1d-141f-406e-8cfe-d6a59b1f0751
# ╠═97ad3034-c16b-47aa-b643-b7b1f48743ce
# ╠═a83353f8-f0c3-4bb5-b662-1e7c446cfa2e
# ╠═41349f3e-6261-4c88-990d-df37fbd2e09c
# ╠═661067e0-5f05-410c-9e2b-911610c44e99
# ╟─07eeb04a-9360-4229-8bad-e59b281d06d0
# ╠═2d3cedb5-867d-4934-b0eb-af89870e9805
# ╠═0ae710f2-3848-4e0d-8567-c40556b64a04
# ╟─30908df4-a962-489b-bcef-73855e608420
