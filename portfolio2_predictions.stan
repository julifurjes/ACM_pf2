data {
  int<lower = 1> trials; // Number of trials
  array[trials] int success; // Success of each trial (1 for success, 0 for failure)
  array[trials] int choice; // Choice made each trial
}

parameters {
  real betaGamble;
  real rate; // Bias of the model
}

transformed parameters {
  vector[trials] gamble;
  for (t in 1:trials) {
    if (t == 1 || t == 2) { // If trial nr is not high enough yet for our model
      gamble[t] = 0; // Random probability of the outcome of 1
    } else if (t>=3) {
        if (choice[t-1] == choice[t-2] && success[t-1] == 1 && success[t-2] == 1) { // If won twice
    	    gamble[t] = (1-2*choice[t-1]);
        } else if (choice[t-1] == choice[t-2] && success[t-1] == 0 && success[t-2] == 0) { // If lost twice
            gamble[t] = (-1+choice[t-1]*2);
        } else { // If there was no consecutive pattern
            gamble[t] = 0; // Random probability of the outcome of 1
        }
      }
  }
}

model {
  // Likelihood for choice
  target += normal_lpdf(betaGamble | 0, 1);
  target += normal_lpdf(rate | 0, 1);
  target += bernoulli_logit_lpmf(choice | rate + betaGamble*gamble);
  
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

// generated quantities{
//   real<lower=0, upper=1> rate_prior;  // theta prior parameter, on a prob scale (0-1)
//   real<lower=0, upper=1> rate_posterior; // theta posterior parameter, on a prob scale (0-1)
//   real<lower=0, upper=1> betaGamble_prior;  // theta prior parameter, on a prob scale (0-1)
//   real<lower=0, upper=1> betaGamble_posterior; // theta posterior parameter, on a prob scale (0-1)
//   real<lower=-1, upper=1> gamble_mean;
//   int<lower=0, upper=trials> prior_preds;  // distribution of right hand choices according to the prior
//   int<lower=0, upper=trials> posterior_preds; // distribution of right hand choices according to the posterior
//   
//   rate_prior = normal_rng(0,1); // generating the prior on a log odds scale and converting
//   rate_posterior = inv_logit(rate);  // converting the posterior estimate from log odds to prob.
//   betaGamble_prior = normal_rng(0,1); // generating the prior on a log odds scale and converting
//   betaGamble_posterior = inv_logit(betaGamble);  // converting the posterior estimate from log odds to prob.
//   prior_preds = binomial_rng(trials, inv_logit(rate_prior));
//   // gamble_mean = mean(gamble);
//   posterior_preds = binomial_rng(trials, inv_logit(rate));
// }
