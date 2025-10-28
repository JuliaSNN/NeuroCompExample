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
function network(config, name)
    @unpack afferents, connections, Npop, spike_exc, spike_inh, exc, inh = config
    @unpack synapse_exc, synapse_inh = config

    # Create neuron populations
    E = Population(exc; synapse=synapse_exc, spike=spike_exc, N=Npop.E, name=name*"E")  # Excitatory population
    I = Population(inh; synapse=synapse_inh, spike=spike_inh, N=Npop.I, name=name*"I")  # Inhibitory population

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

high_noise = SNN.@update Zerlaut2019_network begin
    afferents.layer = PoissonLayer(rate=10Hz, N=100) # Reduced Poisson input layer
end

low_noise = SNN.@update Zerlaut2019_network begin
    afferents.layer = PoissonLayer(rate=5Hz, N=100) # Reduced Poisson input layer
end

model1 = network(high_noise, "Model1")
model2 = network(low_noise, "Model2")

inter_EE = SNN.SpikingSynapse(
    model1.pop.E,
    model2.pop.E,
    :ge,
    conn = (p=0.05, μ=0.2f0, rule=:Fixed),
    name="Inter_module_EE"
)

inter_EI = SNN.SpikingSynapse(
    model1.pop.E,
    model2.pop.I,
    :ge,
    conn = (p=0.05, μ=0.2f0, rule=:Fixed),
    name="Inter_module_EI"
)

model = SNN.compose(;model1, model2, inter_EE, inter_EI) 

SNN.print_model(model)  # Print model summary
SNN.sim!(model, duration=5s)  # Simulate for 5 seconds

# %% [markdown]
# ## Visualization
#
# Visualize the spiking activity of the network.

# %%
# Plot raster plot of network activity
SNN.raster(model.pop, every=1, title="Raster plot of the balanced network")
