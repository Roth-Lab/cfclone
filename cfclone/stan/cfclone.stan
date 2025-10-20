data {
  int<lower=1> num_bins;
  int<lower=1> num_clones;
  matrix[num_bins, num_clones] cn_a;
  matrix[num_bins, num_clones] cn_t;
  array[num_bins] int<lower=0> a;
  array[num_bins] int<lower=0> d;
  array[num_bins] real rdr;
}
transformed data {
  vector[num_clones] ones = rep_vector(1.0, num_clones);
  vector[num_clones] mean_clone_cn;
  for (k in 1 : num_clones) {
    real s = 0;
    for (i in 1 : num_bins) 
      s += cn_t[i, k];
    mean_clone_cn[k] = s / num_bins;
  }
}
parameters {
  // warning: order of these parameters should not be changed
  simplex[num_clones] rho;
  real<lower=0> alpha;
  
  // to avoid numerical issues, the following parameters should not take tiny values
  real<lower=1e-6, upper=1> non_binomiality; // as ->0, we converge to binomial, higher values relaxes the binomial assumption
  real<lower=1e-6> sigma;
}
model {
  rho ~ dirichlet(ones);
  alpha ~ gamma(1, 1);
  non_binomiality ~ beta(1, 100); // prior on relatively low non_binomiality
  sigma ~ gamma(1, 100); // prior on scale of student-t set to low values 
  
  vector[num_bins] baf_a;
  vector[num_bins] baf_b;
  vector[num_bins] mu;
  vector[num_bins] p;
  
  mu = alpha * (cn_t * rho) / dot_product(mean_clone_cn, rho);
  
  p = (cn_a * rho) ./ (cn_t * rho);
  
  baf_a = p / non_binomiality;
  
  baf_b = (1 - p) / non_binomiality;
  
  a ~ beta_binomial(d, baf_a, baf_b);
  
  rdr ~ student_t(25, mu, sigma);
}
