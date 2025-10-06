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
#     display_name: Julia 1.11.5
#     language: julia
#     name: julia-1.11
# ---

# %%
using DrWatson
findproject(@__DIR__) |> quickactivate

using SpikingNeuralNetworks
using UnPack
using Logging
using Plots

global_logger(ConsoleLogger())
SNN.@load_units

# %%
import SpikingNeuralNetworks: IFSinExpParameter, IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population

Zerlaut2019_network = (
    Npop = (E=4000, I=1000),

    exc = IFParameter(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -50.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                ),

    inh = IFParameter(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -53.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                ),

    synapse = SingleExpSynapse(
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV
            ),

    connections = (
        E_to_E = (p = 0.05, μ = 2nS, rule=:Fixed),
        E_to_I = (p = 0.05, μ = 2nS, rule=:Fixed),
        I_to_E = (p = 0.05, μ = 10nS, rule=:Fixed),
        I_to_I = (p = 0.05, μ = 10nS, rule=:Fixed),
        ),
    
    afferents = (
        N = 100, 
        rate = 20Hz,
        conn = (p = 0.1f0,μ = 4.0),
        ), 
)


# %%

function network(config)
    @unpack afferents, connections, Npop, synapse, exc, inh = config

    E = Population(exc, synapse, N=Npop.E, name="E")
    I = Population(inh, synapse, N=Npop.I, name="I")

    AfferentParam = PoissonLayer(rate=afferents.rate, N=afferents.N)
    afferentE = Stimulus(AfferentParam, E, :ge, conn=afferents.conn, name="noiseE")
    afferentI = Stimulus(AfferentParam, I, :ge, conn=afferents.conn, name="noiseI")

    synapses = (
        E_to_E = SpikingSynapse(E, E, :ge, conn = connections.E_to_E, name="E_to_E"),
        E_to_I = SpikingSynapse(E, I, :ge, conn = connections.E_to_I, name="E_to_I"),
        I_to_E = SpikingSynapse(I, E, :gi, conn = connections.I_to_E, name="I_to_E"),
        I_to_I = SpikingSynapse(I, I, :gi, conn = connections.I_to_I, name="I_to_I"),
    )
    model = compose(;E,I, afferentE, afferentI, synapses..., silent=true, name="Balanced network") 
    monitor!(model.pop, [:fire])
    monitor!(model.stim, [:fire])
    return model
end

# %%
model = network(Zerlaut2019_network) 
SNN.print_model(model)
SNN.sim!(model, duration=5s)



# %%
SNN.raster(model.pop, every=5, title="Raster plot of the balanced network")

# %%
WAVE = (
    Faff1= 4.,
    Faff2= 20,
    Faff3 =8.,
    DT= 900.,
    rise =50.
)
using SpecialFunctions
waveform = zeros(3000)
t = 1:3000
for (tt, fa) in zip(2 .*wave.rise .+(0:2) .*(3 .*wave.rise + wave.DT), [wave.Faff1, wave.Faff2, wave.Faff3])
    waveform .+= fa .* (1 .+erf.((t .-tt) ./wave.rise)) .* (1 .+erf.(-(t.-tt.-wave.DT)./wave.rise))./4
end

plot(waveform, xlabel="Time (ms)", ylabel="Afferent rate (Hz)", title="Afferent waveform", legend=false, lw=4, c=:black)


# %% [markdown]
# Run simulation with varying input

# %%
SNN.reset_time!(model)
SNN.clear_records!(model)
SNN.monitor!(model.pop, [:v])
for t in 1:3000
    model.stim.afferentE.param.rates .= waveform[t] .*Hz
    model.stim.afferentI.param.rates .= waveform[t] .*Hz
    SNN.sim!(model, duration=1ms)
end

# %%
fr, r, labels = SNN.firing_rate(model.pop, interval=0f0:10ms:3s, pop_average=true);
plot(r, fr, labels=hcat(labels...), xlabel="Time (s)", ylabel="Firing rate (Hz)", title="Population firing rates", lw=2)

# %%
v , r = SNN.record(model.pop.E, :v, interval=0f0:10ms:3s, range=true);
plotsE = map(1:3) do i
    plot(r, v[i,:], xlabel="Time (s)", ylabel="Potential (mV)", label="Exc $i",  lw=2, c=:darkblue)
end
v , r = SNN.record(model.pop.I, :v, interval=0f0:10ms:3s, range=true);
plotsI = map(1:3) do i
    plot(r, v[i,:], xlabel="Time (s)", ylabel="Potential (mV)", label="Inh $i",  lw=2, c=:darkred)
end
plots = vcat(plotsE..., plotsI...)
plot(plots..., layout=(3,2), plot_title="Neuron membrane (mV)", size=(900,600))

