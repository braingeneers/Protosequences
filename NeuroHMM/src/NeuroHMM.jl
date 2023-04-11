module NeuroHMM

import Base.length
using Random
using LinearAlgebra
using StatsBase
using Distributions
using HMMBase
using MAT

export load_tj, fit_hmm, MultivariatePoisson

DATA_DIR = "./data/"

EXPERIMENTS = [
               "L1_200123_2953_C.mat",
               "L2_200123_2950_C.mat",
               "L3_200123_2957_C.mat",
               "L5_200520_5116_C.mat",
              ]


function load_tj(source::String, exp::String, bin_size_ms::Real)
    mat = matread(joinpath(DATA_DIR, source, exp))

    # Spike locations are recorded in samples, so convert to ms.
    spike_trains = [unit["spike_train"] ./ mat["fs"] * 1e3
                    for unit in mat["units"]]
    n_neurons = length(spike_trains)

    # Duration should be rounded to the nearest second for consistency.
    max_length = maximum(t[1,end] for t in spike_trains)
    duration_ms = 1000 * Int(ceil(max_length / 1e3))

    # Output matrix is a 2D array in time by neurons format.
    ret = zeros(Int, duration_ms ÷ bin_size_ms, n_neurons)
    for i in 1:length(spike_trains)
        for t in spike_trains[i]
            ret[ceil(Int, t / bin_size_ms), i] += 1
        end
    end
    return ret
end


function fit_hmm(K::Int, raster::AbstractMatrix{Int};
        init=:kmeans, display=:none, tol=1e-3, maxiter=1000)
    # As a prior, assume completely random transitions.
    trans_prior = rand(K,K)
    trans_prior ./= sum(trans_prior; dims=2)

    # Also assume that all states have identical observations, equal to the
    # mean firing rate of each of the neurons.
    poissons = MultivariatePoisson(mean(raster, dims=1)[1,:])
    observation_prior = [MultivariatePoisson(poissons.λ) for _ in 1:K]

    # Create, fit, and return an HMM with those priors. 
    hmm = HMM(trans_prior, observation_prior)
    fit_mle(hmm, raster; init, display, tol, maxiter)
end


struct MultivariatePoisson <: DiscreteMultivariateDistribution
    λ::Vector{Float64}
end

length(d::MultivariatePoisson) = length(d.λ)

function Distributions._logpdf(d::MultivariatePoisson, x::AbstractVector{Int})
    return sum(logpdf.(Poisson.(d.λ), x))
end

function Distributions._rand!(rng::AbstractRNG, d::MultivariatePoisson,
        x::AbstractArray{<:Real})
    x .= rand.(rng, Poisson.(d.λ))
end

function Distributions.mean(d::MultivariatePoisson)
    return d.λ
end

function Distributions.var(d::MultivariatePoisson)
    return d.λ
end

function Distributions.entropy(d::MultivariatePoisson)
    return sum(entropy.(Poisson.(d.λ)))
end

function Distributions.cov(d::MultivariatePoisson)
    return diagm(d.λ)
end

function Distributions.fit_mle(::Type{MultivariatePoisson},
        x::AbstractMatrix{Int}, w::AbstractVector{Float64})
    MultivariatePoisson(mean(x, pweights(w), 2)[:,1])
end

function Distributions.fit_mle(::Type{MultivariatePoisson},
        x::AbstractMatrix{Int})
    MultivariatePoisson(mean(x; dims=2)[:,1])
end



end # module NeuroHMM
