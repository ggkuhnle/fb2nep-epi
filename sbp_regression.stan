
data {
  int<lower=1> N;
  vector[N] SBP;
  vector[N] BMI;
  vector[N] age;
  vector[N] sex_M;        // now a vector, not a scalar
}

parameters {
  real intercept;
  real beta_BMI;
  real beta_age;
  real beta_sex;
  real<lower=0> sigma;
}

model {
  // Weakly informative priors
  intercept ~ normal(120, 20);
  beta_BMI  ~ normal(0.5, 0.5);
  beta_age  ~ normal(0.6, 0.3);
  beta_sex  ~ normal(0, 2);
  sigma     ~ normal(15, 5);

  // Linear predictor
  SBP ~ normal(
    intercept
    + beta_BMI * BMI
    + beta_age * age
    + beta_sex * sex_M,
    sigma
  );
}
