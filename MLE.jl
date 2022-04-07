using Oceananigans
using Oceananigans.Units

# ---- define size and create a grid

const sponge = 20 #number of points for sponge
const Ny = 46 # number of points in y
const Nz = 48 # number of points in z
const H = 1000 # maximum depth


grid = RectilinearGrid(GPU(),
    size=(Ny+2sponge,Nz),
    halo=(3,3),
    y=(-10*(Ny/2 + sponge)kilometers, 10*(Ny/2 + sponge)kilometers), 
    z=(H * cos.(LinRange(π/2,0,Nz+1)) .- H)meters,
    topology=(Flat, Bounded, Bounded)
)

# ---- 
# ---- rotation

coriolis = FPlane(latitude=60)

# ---- 
# ---- turbulent diffusivity (with sponges)

@inline νh(x,y,z,t) = ifelse((y>-(Ny/2)kilometers)&(y<(Ny/2)kilometers), 1, 100)
horizontal_closure = HorizontalScalarDiffusivity(ν=νh, κ=νh)

@inline νz(x,y,z,t) = ifelse((y>-(Ny/2)kilometers)&(y<(Ny/2)kilometers), 1e-5, 1e-3)
vertical_closure = ScalarDiffusivity(ν=νz, κ=νz)

# ---- 
# ---- instantiate model
ub = Field((Face, Center, Center), grid)
vb = Field((Center, Face, Center), grid)
wb = Field((Center, Center, Face), grid)

model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            closure = (horizontal_closure, vertical_closure),
                            tracers = (:b),
                            background_fields = (u=ub, v=vb, w=wb),
                            buoyancy = BuoyancyTracer())

# ---- 
# ---- initial conditions

const g = 9.82
const ρₒ = 1026

# background density profile based on Argo data
@inline bg(z) = 0.25*tanh(0.0027*(-653.3-z))-6.8*z/1e5+1027.56

# decay function for fronts
@inline decay(z) = (tanh((z+500)/300)+1)/2

# front function
@inline front(x, y, z, cy) = tanh((y-cy)/12kilometers)

@inline function B(x, y, z)
    fronts = front(x, y, z, -120kilometers) + front(x, y, z, 0) + front(x, y, z, 120kilometers)
    return -( g / ρₒ ) * ( bg(z) + 0.8 * decay(z) * (fronts - 3) / 6 )
end

set!(model; b = B)

# ---- 
# ---- geostrophic initial velocity

b = model.tracers.b
f = model.coriolis.f

# shear operations
uz_op = @at((Face, Center, Center), - ∂y(b) / f );

# compute shear
uz = compute!(Field(uz_op))

# include function for cumulative integration
include("src/cumulative_vertical_integration.jl")

# compute geostrophic velocities
U = cumulative_vertical_integration!(uz)

# prescribe geostrophic velocities for initial condition
set!(model; u = U)

# ---- 
# ---- create a simulation with adaptive time stepping based on CFL condition

simulation = Simulation(model, Δt = 1minutes, stop_time = 10day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=5minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ---- 
# ---- compute mixed-layer depth
h = Field{Center, Center, Nothing}(grid)

# buoyancy decrease criterium for determining the mixed-layer depth
const g = 9.82 # gravity
const ρₒ = 1026 # reference density
const Δb=(g/ρₒ) * 0.03

include("src/compute_mixed_layer_depth.jl")
compute_mixed_layer_depth!(simulation) = compute_mixed_layer_depth!(h, simulation.model.tracers.b, Δb)

# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)

# ---- 
# ---- compute gradient of buoyancy
using Oceananigans.BoundaryConditions: fill_halo_regions!

∂b∂x = Field{Face, Center, Center}(grid)
∂b∂y = Field{Center, Face, Center}(grid)
∂b∂z = Field{Center, Center, Face}(grid)

b = model.tracers.b
∂b∂x_op = ∂x(b);
∂b∂y_op = ∂y(b);
∂b∂z_op = ∂z(b);

function compute_∇b!(sim)
    ∂b∂x .= ∂b∂x_op
    ∂b∂y .= ∂b∂y_op
    ∂b∂z .= ∂b∂z_op
    fill_halo_regions!(∂b∂x, sim.model.architecture)
    fill_halo_regions!(∂b∂y, sim.model.architecture)
    fill_halo_regions!(∂b∂z, sim.model.architecture)
    return nothing
end
simulation.callbacks[:compute_∇b] = Callback(compute_∇b!)

# ----
# ---- compute Ψₑ

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ


@kernel function _compute_Ψₑ!(Ψₑ, grid, h, ∂ₕb, N, μ, f, ce, Lfₘ, ΔS, τ)
    i, j = @index(Global, NTuple)

    # average ∇ₕb and N over the mixed layer
    
    ∂ₕb_sum = 0
    N_sum = 0
    Δz_sum = 0

    h_ij = @inbounds h[i, j]
    
    @unroll for k in grid.Nz : -1 : 1 # scroll from surface to bottom       
        z_center = znode(Center(), Face(), Face(), i, j, k, grid)

        if z_center > -h_ij

            Δz_ijk = Δzᶜᶜᶜ(i, j, k, grid)

            ∂ₕb_sum = ∂ₕb_sum + @inbounds ∂ₕb[i, j, k] * Δz_ijk
            N_sum = N_sum + @inbounds N[i, j, k] * Δz_ijk 
            Δz_sum = Δz_sum + Δz_ijk

        end
    end

    ∂ₕbₘₗ = ∂ₕb_sum/Δz_sum
    Nₘₗ = N_sum/Δz_sum
    
    Lf = max(Nₘₗ*h_ij/abs(f), Lfₘ)
    
    # compute eddy stream function
    @unroll for k in grid.Nz : -1 : 1 # scroll to point just above the bottom       
        z_face = znode(Center(), Face(), Face(), i, j, k, grid)

        if z_face > -h_ij
            @inbounds Ψₑ[i, j, k] = ce * (ΔS/Lf) * ((h_ij^2)/sqrt(f^2 + τ^-2)) * μ(z_face,h_ij) * ∂ₕbₘₗ
        else
            @inbounds Ψₑ[i, j, k] = 0.0
        end
    end

end

function compute_Ψₑ!(Ψₑ, h, ∂ₕb, N, μ, f; ce = 0.06, Lfₘ = 500meters, ΔS=10e3, τ=86400)
    grid = h.grid
    arch = architecture(grid)


    event = launch!(arch, grid, :xy,
                    _compute_Ψₑ!, Ψₑ, grid, h, ∂ₕb, N, μ, f, ce, Lfₘ, ΔS, τ,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return nothing
end

# x-component of Ψ
Ψx = Field{Center, Face, Face}(grid)

# structure function
@inline μ(z,h) = (1-(2*z/h + 1)^2)*(1+(5/21)*(2*z/h + 1)^2)

compute_Ψₑ!(simulation) = compute_Ψₑ!(Ψx, h, ∂b∂y, sqrt(∂b∂z), μ, f)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_Ψₑ] = Callback(compute_Ψₑ!)


v_op = @at((Center, Face, Center),   ∂z(Ψx));
w_op = @at((Center, Center, Face), - ∂y(Ψx));

function compute_background_vel!(sim)
    u, v, w = sim.model.background_fields.velocities
    v .= v_op
    w .= w_op
    fill_halo_regions!(v, sim.model.architecture)
    fill_halo_regions!(w, sim.model.architecture)
    return nothing
end
simulation.callbacks[:compute_background_vel] = Callback(compute_background_vel!)

# ----
# ---- setting up the model output

outputs = merge(model.velocities, model.tracers, (; h, ∂b∂y, Ψx)) # make a NamedTuple with all outputs

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filepath = "data/output.nc",
                     schedule=TimeInterval(8hours))

# ----
# ---- setting up the model progress messages

using Printf

function print_progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(1hour))

# ----
# ---- run the simulation

run!(simulation)