using Oceananigans
using Oceananigans.Units

# ---- define size and create a grid

# define the size and max depth of the simulation
const sponge = 20
const Ny = 100
const Nz = 48 # number of points in z
const H = 1000 # maximum depth

# create the grid of the model
grid = RectilinearGrid(GPU(),
    size=(Ny+2sponge, Nz),
    halo=(3,3),
    y=(-(Ny/2 + sponge)kilometers, (Ny/2 + sponge)kilometers), 
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

model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            closure=(horizontal_closure, vertical_closure),
                            tracers = (:b),
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

@inline D(x, y, z) = bg(z) + 0.8*decay(z)*((front(x, y, z, -120kilometers)+front(x, y, z, 0)+front(x, y, z, 120kilometers))-3)/6
@inline B(x, y, z) = -(g/ρₒ)*D(x, y, z)

set!(model; b = B)

# ---- 
# ---- create fields for mixed-layer depth, eddy stream function and gradient of buoyancy

h = Field{Center, Center, Nothing}(grid)
Ψₑ = Field{Center, Center, Center}(grid)
∇ₕb = Field{Center, Center, Center}(grid)


# ---- 
# ---- grostrophic initial velocity

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

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=3hours)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ---- 
# ---- compute mixed-layer depth

# buoyancy decrease criterium for determining the mixed-layer depth
const g = 9.82 # gravity
const ρₒ = 1026 # reference density
const Δb=(g/ρₒ) * 0.03
include("src/compute_mixed_layer_depth.jl")
compute_mixed_layer_depth!(simulation) = compute_mixed_layer_depth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)

# ---- 
# ---- compute horizontal gradient of buoyancy
using Oceananigans.BoundaryConditions: fill_halo_regions!

b = model.tracers.b
∇ₕb_op = ∂x(b) + ∂y(b);

function compute_∇ₕb!(sim)
    ∇ₕb .= ∇ₕb_op
    fill_halo_regions!(∇ₕb, sim.model.architecture)
    return nothing
end
simulation.callbacks[:compute_∇ₕb] = Callback(compute_∇ₕb!)

# ----
# ---- compute Ψₑ

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Operators: Δzᶜᶜᶜ


@kernel function _compute_Ψₑ!(Ψₑ, grid, h, ∇ₕb, μ, f, ce, Lfₘ, dS)
    i, j = @index(Global, NTuple)

    # average ∇ₕb over the mixed layer
    
    ∇ₕb_sum = 0
    dz_sum = 0

    @unroll for k in grid.Nz : -1 : 1 # scroll from surface to bottom       
        z_center = znode(Center(), Center(), Center(), i, j, k, grid)

        h_ijk = @inbounds h[i, j, k]

        if z_center > -h_ijk

            Δz_ijk = Δzᶜᶜᶜ(i, j, k, grid)

            ∇ₕb_sum = ∇ₕb_sum + @inbounds ∇ₕb[i, j, k] * Δz_ijk 
            dz_sum = dz_sum + Δz_ijk

        end
    end

    ∇ₕbₘₗ = ∇ₕb_sum/dz_sum
    
    # compute eddy stream function
    @unroll for k in grid.Nz : -1 : 1 # scroll to point just above the bottom       
        z_center = znode(Center(), Center(), Center(), i, j, k, grid)

        h_ijk = @inbounds h[i, j, k]

        if z_center > -h_ijk
            @inbounds Ψₑ[i, j, k] = ce * (dS/Lfₘ) * ((h_ijk^2)/abs(f)) * μ(z_center,h_ijk) * ∇ₕbₘₗ  
        end
    end

end

function compute_Ψₑ!(Ψₑ, h, ∇ₕb, μ, f; ce = 0.06, Lfₘ = 500meters, dS=10e3)
    grid = h.grid
    arch = architecture(grid)


    event = launch!(arch, grid, :xy,
                    _compute_Ψₑ!, Ψₑ, grid, h, ∇ₕb, μ, f, ce, Lfₘ, dS,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return nothing
end

# structure function
@inline μ(z,h) = (1-(2*z/h + 1)^2)*(1+(5/21)*(2*z/h + 1)^2)

compute_Ψₑ!(simulation) = compute_Ψₑ!(Ψₑ, h, ∇ₕb, μ, f)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_Ψₑ] = Callback(compute_Ψₑ!)

# ----
# ---- setting up the model output

outputs = merge(model.velocities, model.tracers, (; h, ∇ₕb, Ψₑ)) # make a NamedTuple with all outputs

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