data {
  int<lower = 1> trials; // Number of trials
  array[trials] int hand; // Observed hand choice outcomes (0 or 1)
  array[trials] int success; // Success of each trial (1 for success, 0 for failure)
  array[trials] int choice; // Choice made each trial
  real noise; // Overall noise of the model
}

parameters {
  real betaGamble;
  real rate; // Bias of the model
}

transformed parameters {
  array[trials] real gamble;
  for (t in 1:trials) {
    if (t == 1 || t == 2) { // If trial nr is not high enough yet for our model
      gamble[t] = 0; // Random probability of the outcome of 1
    } else if (t>=3) {
        if (choice[t-1] == choice[t-2] && success[t-1] == 1) { // If won twice
    	    gamble[t] = 1;
        } else if (choice[t-1] == choice[t-2] && success[t-1] == 0) { // If lost twice
            gamble[t] = 1;
        } else { // If there was no consecutive pattern
            gamble[t] = 0; // Random probability of the outcome of 1
        }
      }
  }
}

model {
  // Likelihood for choice
  for (t in 1:trials) {
    target += normal_lpdf(betaGamble | 0, 0.3);
    target += normal_lpdf(rate | 0.5, 0.3);
    target += bernoulli_logit_lpmf(choice[t] | rate + betaGamble*gamble[t]);
    //target += bernoulli_logit_lpmf(choice[t] | log(betaGamble[t]*Gamble[t]) - log(1 - betaGamble[t]) + log(rate));
  }
}

generated quantities {
  real betaGamble_prior;
  real rate_prior;
  array[trials] int prior_preds;
  array[trials] int posterior_preds;
  
  betaGamble_prior = normal_rng(0, 1);
  rate_prior = normal_rng(0, 1);
  
  for (t in 1:trials) {
    prior_preds[t] = bernoulli_rng(inv_logit(rate_prior + betaGamble_prior*gamble[t]));
    posterior_preds[t] = bernoulli_rng(inv_logit(rate + betaGamble*gamble[t]));
  }
}
