data {
  int<lower = 1> trials; // Number of trials
  array[trials] int success; // Success of each trial (1 for success, 0 for failure)
  array[trials] int choice; // Choice made each trial
}

parameters {
  real betaGamble; // coefficient/weight for gamble rule
  real rate; // Bias of the model
}

transformed parameters {
  vector[trials] gamble;
  for (t in 1:trials) { // gamble requires at least 3 observations
    if (t == 1 || t == 2) { // therefore in the first 2 trials
      gamble[t] = 0; // gamble is not used; outcome is random probability of 1
    } else if (t>=3) {
        if (choice[t-1] == choice[t-2] && success[t-1] == 1 && success[t-2] == 1) {
    	    gamble[t] = (1-2*choice[t-1]); // given 2 consecutive wins, gamble should be considered
        } else if (choice[t-1] == choice[t-2] && success[t-1] == 0 && success[t-2] == 0) {
            gamble[t] = (-1+choice[t-1]*2); // given 2 consecutive losses, gamble should be considered
        } else { // given no consecutive pattern
            gamble[t] = 0; // gamble is not used; outcome is random probability of 1
        }
      }
  }
}

model {
  target += normal_lpdf(betaGamble | 0,1); //betaGamble prior
  target += normal_lpdf(rate | 0,1); //rate (bias) prior
  target += bernoulli_logit_lpmf(choice | rate + betaGamble*gamble);
} // Bernoulli logit likelihood of choice given linear combination of our parameters 

generated quantities {
  real betaGamble_prior;
  real rate_prior;
  array[trials] int prior_preds; // arrays for trial-wise predictions for 
  array[trials] int posterior_preds; // subsequent plotting and analysis
  
  betaGamble_prior = normal_rng(0, 1); //sampling priors from normal distr.
  rate_prior = normal_rng(0, 1);
  
  for (t in 1:trials) {
    prior_preds[t] = bernoulli_rng(inv_logit(rate_prior + betaGamble_prior*gamble[t]));
    posterior_preds[t] = bernoulli_rng(inv_logit(rate + betaGamble*gamble[t]));
    // populate the prior and post. prediction arrays with the respective preds,
    // using the respective paramaters, at each trial t, with the same specification
    // as our likelihood model.
  }
}
