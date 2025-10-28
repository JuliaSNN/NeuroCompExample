# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Julia 1.12.1
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # Spiking Neural Network Simulation with Varying Input
#
# This notebook demonstrates how to create and simulate a spiking neural network with varying input rates.
# The network consists of excitatory and inhibitory populations with recurrent connections and external Poisson input.

# %%
# Import necessary packages and set up environment
using DrWatson
findproject(@__DIR__) |> quickactivate

using SpikingNeuralNetworks
using UnPack
using Logging
using Plots

# Set global logger to display messages in console
global_logger(ConsoleLogger())

# Load units for physical quantities
SNN.@load_units

# %% [markdown]
# ## Network Configuration
#
# Define the network parameters including neuron populations, synaptic properties,
# and connection probabilities.

# %%
# Define network configuration parameters
import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population, PostSpike, AdExParameter


Zerlaut2019_network = (
    # Number of neurons in each population
    Npop = (E=4000, I=1000),

    # Parameters for excitatory neurons
    exc = IFParameter(
                τm = 200pF / 10nS,  # Membrane time constant
                El = -70mV,         # Leak reversal potential
                Vt = -50.0mV,       # Spike threshold
                Vr = -70.0f0mV,     # Reset potential
                R  = 1/10nS,        # Membrane resistance
                # a = 4nS,
                # b = 80pA
                ),

    # Parameters for inhibitory neurons
    inh = IFParameter(
                τm = 200pF / 10nS,  # Membrane time constant
                El = -70mV,         # Leak reversal potential
                Vt = -53.0mV,       # Spike threshold
                Vr = -70.0f0mV,     # Reset potential
                R  = 1/10nS,        # Membrane resistance
                ),

    spike_exc = PostSpike(τabs = 2ms),         # Absolute refractory period
    spike_inh = PostSpike(τabs = 1ms),         # Absolute refractory period 

    # Synaptic properties
    synapse_exc = SingleExpSynapse(
                τi=5ms,             # Inhibitory synaptic time constant
                τe=5ms,             # Excitatory synaptic time constant
                E_i = -80mV,        # Inhibitory reversal potential
                E_e = 0mV           # Excitatory reversal potential
            ),

    synapse_inh = SingleExpSynapse(
                τi=5ms,             # Inhibitory synaptic time constant
                τe=5ms,             # Excitatory synaptic time constant
                E_i = -80mV,        # Inhibitory reversal potential
                E_e = 0mV           # Excitatory reversal potential
            ),


    # Connection probabilities and synaptic weights
    connections = (
        E_to_E = (p = 0.05, μ = 2nS,  rule=:Fixed),  # Excitatory to excitatory
        E_to_I = (p = 0.05, μ = 2nS,  rule=:Fixed),  # Excitatory to inhibitory
        I_to_E = (p = 0.05, μ = 10nS, rule=:Fixed), # Inhibitory to excitatory
        I_to_I = (p = 0.05, μ = 10nS, rule=:Fixed), # Inhibitory to inhibitory
        ),

    # Parameters for external Poisson input
    afferents = (
        layer = PoissonLayer(rate=10Hz, N=100), # Poisson input layer
        conn = (p = 0.1f0, μ = 4.0nS), # Connection probability and weight
        ),
)

# %% [markdown]
# ## Network Construction
#
# Define a function to create the network based on the configuration parameters.

# %%
# Function to create the network
function network(config)
    @unpack afferents, connections, Npop, spike_exc, spike_inh, exc, inh = config
    @unpack synapse_exc, synapse_inh = config

    # Create neuron populations
    E = Population(exc; synapse=synapse_exc, spike=spike_exc, N=Npop.E, name="E")  # Excitatory population
    I = Population(inh; synapse=synapse_inh, spike=spike_inh, N=Npop.I, name="I")  # Inhibitory population

    # Create external Poisson input
    @unpack layer = afferents
    afferentE = Stimulus(layer, E, :glu, conn=afferents.conn, name="noiseE")  # Excitatory input
    afferentI = Stimulus(layer, I, :glu, conn=afferents.conn, name="noiseI")  # Inhibitory input

    # Create recurrent connections
    synapses = (
        E_to_E = SpikingSynapse(E, E, :glu, conn = connections.E_to_E, name="E_to_E"),
        E_to_I = SpikingSynapse(E, I, :glu, conn = connections.E_to_I, name="E_to_I"),
        I_to_E = SpikingSynapse(I, E, :gaba, conn = connections.I_to_E, name="I_to_E"),
        I_to_I = SpikingSynapse(I, I, :gaba, conn = connections.I_to_I, name="I_to_I"),
    )

    # Compose the model
    model = compose(; E,I, afferentE, afferentI, synapses..., name="Balanced network")

    # Set up monitoring
    monitor!(model.pop, [:fire])  # Monitor spikes
    monitor!(model.stim, [:fire])  # Monitor input spikes

    return model
end

# %% [markdown]
# ## Network Simulation
#
# Create the network and simulate it for a fixed duration.

# %%
# Create and simulate the network
model = network(Zerlaut2019_network)
SNN.print_model(model)  # Print model summary
SNN.sim!(model, duration=5s)  # Simulate for 5 seconds

# %% [markdown]
# ## Visualization
#
# Visualize the spiking activity of the network.

# %%
# Plot raster plot of network activity
SNN.raster(model.pop, every=1, title="Raster plot of the balanced network")

# %% [markdown]
# ## Afferent Waveform
#
# Define a time-varying afferent input waveform.

# %%
# Define afferent waveform parameters
wave = (
    Faff1= 4.,    # First peak frequency
    Faff2= 20,    # Second peak frequency
    Faff3 =8.,    # Third peak frequency
    DT= 900.,     # Duration between peaks
    rise =50.     # Rise time
)

# Create the waveform using error functions
using SpecialFunctions
waveform = zeros(3000)  # Initialize waveform array
t = 1:3000              # Time points

# Create three peaks with different frequencies
for (tt, fa) in zip(2 .*wave.rise .+(0:2) .*(3 .*wave.rise + wave.DT), [wave.Faff1, wave.Faff2, wave.Faff3])
    waveform .+= fa .* (1 .+erf.((t .-tt) ./wave.rise)) .* (1 .+erf.(-(t.-tt.-wave.DT)./wave.rise))./4
end

# Plot the waveform
plot(waveform, xlabel="Time (ms)", ylabel="Afferent rate (Hz)", title="Afferent waveform", legend=false, lw=4, c=:black)

# %% [markdown]
# ## Simulation with Varying Input
#
# Run the simulation with the time-varying afferent input.
#
# Notice. Sometimes the network enters into runaway activity because of statistical fluctuations, in this case just generate a new model with the `network` function and re-run it.

# %%
# Reset the model and clear previous recordings
model1 = network(Zerlaut2019_network)  # Recreate the network to reset state
SNN.reset_time!(model1)
SNN.clear_records!(model1)

# Monitor membrane potentials
SNN.monitor!(model.pop, [:v])

# Simulate with time-varying input
for t in 1:3000
    # Update input rates based on waveform
    model.stim.afferentE.param.rates .= waveform[t] .*Hz
    model.stim.afferentI.param.rates .= waveform[t] .*Hz

    # Simulate for 1 millisecond
    SNN.sim!(model, duration=1ms)
end

# %% [markdown]
# ## Results Analysis
#
# Analyze the firing rates and membrane potentials of the neurons.

# %%
# Calculate and plot population firing rates
fr, r, labels = SNN.firing_rate(model.pop, interval=0f0:10ms:3s, pop_average=true);
plot(waveform, xlabel="Time (ms)", ylabel="Afferent rate (Hz)", title="Afferent waveform", label="", lw=4, c=:black)
plot!(r, fr, labels=hcat(labels...), xlabel="Time (s)", ylabel="Firing rate (Hz)", title="Population firing rates", lw=2, c=[:darkred :darkblue])

# %%
# Plot membrane potentials for selected neurons
# Get membrane potentials for excitatory neurons

# Create plots for 3 excitatory neurons
# plotsE = map(1:3) do i
SNN.vecplot(model.pop.E, :v, neurons=3, 
            xlabel="Time (s)", 
            ylabel="Potential (mV)", 
            label="Exc _3",  
            lw=2, 
            c=:darkblue)
# end

v, r = SNN.record(model.pop.E, :v, range=true);
# Create plots for 3 inhibitory neurons
plotsE = map(1:3) do i
    plot(r, v(i,r), xlabel="Time (s)", ylabel="Potential (mV)", label="Inh $i",  lw=2, c=:darkred)
end


v, r = SNN.record(model.pop.I, :v, range=true);
# Create plots for 3 inhibitory neurons
plotsI = map(1:3) do i
    plot(r, v(i,r), xlabel="Time (s)", ylabel="Potential (mV)", label="Inh $i",  lw=2, c=:darkblue)
end

# Combine and arrange plots
plots = vcat(plotsE..., plotsI...)[[1,4,2,5,3,6]]
plot(plots..., layout=(3,2), plot_title="Neuron membrane (mV)", size=(900,600))

# %% [markdown]
# ## Connect two modules
#
# Connect two networks with excitatory synapses

# %%
high_noise = SNN.@update Zerlaut2019_network begin
    afferents.layer = PoissonLayer(rate=10Hz, N=100) # Reduced Poisson input layer
end
model1 = network(high_noise);

low_noise = SNN.@update Zerlaut2019_network begin
    afferents.layer = PoissonLayer(rate=5Hz, N=100) # Reduced Poisson input layer
end
model2 = network(low_noise);

# %%
intermodule = SNN.SpikingSynapse(
    model1.pop.E,
    model2.pop.E,
    :ge,
    conn = (p=0.05, μ=0.00f0, rule=:Fixed),
    name="Inter_module_EE"
)

# %%
full_model = SNN.compose(;model1, model2, intermodule)

# %%
SNN.reset_time!(full_model)
SNN.clear_records!(full_model)
sim!(full_model, duration=5s)

# %%
SNN.raster(full_model.pop, title="Raster plot of module 1")

# %%

# %%
