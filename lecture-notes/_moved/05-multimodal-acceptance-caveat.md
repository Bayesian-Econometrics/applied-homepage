> Cut from lecture note 5 in the restructuring pass. Belongs in the **Week 5 exercise
> session** as a discussion prompt. The two-mode thought experiment showing that a high
> acceptance rate certifies nothing about global exploration, with the source connection
> to Lopes on multimodal targets. The point itself is retained in the rewritten chapter,
> where the tiny-step chain accepts 92 percent of proposals and still fails its Rhat
> check; this longer discussion version moves here.

## A chain can accept frequently and still miss the posterior

Imagine a posterior with two separated modes. A very small random-walk proposal
often remains near its current high-density point and is accepted. Nevertheless,
the chain may never cross the low-density gap. A high acceptance rate then tells
us about local step size, not global exploration. Starting several chains in the
same mode can hide the problem even in between-chain diagnostics.

The logistic comparison above illustrates the constructive side of the same point: inspect coefficient
traces, effective sample sizes per second and agreement across dispersed starts, and note that the
tiny-step chain accepted almost every proposal while returning an $\hat R$ far above the threshold. The
posterior under a proper Normal prior need not resemble a deliberately multimodal example; the lesson
is that an acceptance statistic is not a convergence certificate. NUTS adapts
trajectory length and step size but does not remove every difficult geometry.
Explain which posterior feature makes your chosen diagnostic informative.

*Source connection:* Lopes, *Bayesian computation*, PDF pp. 42–56, compares
proposal behaviour in a multimodal target. The scheduled model remains logistic regression.
