functions {
  real baf_lpmf(int b, int d, real alpha, real beta, real outlier_rate_baf) {
    return log_mix(outlier_rate_baf, beta_binomial_lpmf(b | d, 1., 1.), beta_binomial_lpmf(b | d, alpha, beta));
  }
  
  real rdr_lpdf(real rdr, real mu, real sigma, real sigma_outlier, real outlier_rate_rdr) {
    return log_mix(outlier_rate_rdr, student_t_lpdf(rdr | 25, 0, sigma_outlier), student_t_lpdf(rdr | 25, mu, sigma));
  }
}
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
  real<lower=1e-6> sigma_outlier;
  
  // // these, on the other hand, are numerically stable even close to the boundaries [0, 1]
  real<lower=0, upper=1> outlier_rate_rdr; // probability a genomic bin is are considered outliers in the RDR data.
  real<lower=0, upper=1> outlier_rate_baf; // probability a genomic bin is are considered outliers in the BAF data.
}
model {
  rho ~ dirichlet(ones);
  alpha ~ gamma(1, 1);
  non_binomiality ~ beta(1, 100); // prior on relatively low non_binomiality
  sigma ~ gamma(1, 100); // prior on scale of student-t set to low values 
  sigma_outlier ~ gamma(1, 1); // prior on scale of student-t set to high values
  outlier_rate_rdr ~ beta(1, 100); // prior on outlier rate for RDR data is set to low values (expect few genomic bins to be outliers)
  outlier_rate_baf ~ beta(1, 100); // prior on outlier rate for BAF data is set to low values (expect few genomic bins to be outliers)
  
  vector[num_bins] baf_a;
  vector[num_bins] baf_b;
  vector[num_bins] mu;
  vector[num_bins] p;
  
  mu = alpha * (cn_t * rho) / dot_product(mean_clone_cn, rho);
  
  p = (cn_a * rho) ./ (cn_t * rho);
  
  baf_a = p / non_binomiality;
  
  baf_b = (1 - p) / non_binomiality;
  
  for (i in 1 : num_bins) {
    a[i] ~ baf(d[i], baf_a[i], baf_b[i], outlier_rate_baf);
    
    rdr[i] ~ rdr(mu[i], sigma, sigma_outlier, outlier_rate_rdr);
  }
}
