functions {
  real baf_lpmf(int a, int d, real alpha, real beta, real outlier_lpmf, real outlier_rate) {
    return log_mix(outlier_rate, outlier_lpmf, beta_binomial_lpmf(a | d, alpha, beta));
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
  vector[num_clones] pi;
}
transformed data {
  vector[num_clones] mean_clone_cn;
  vector[num_bins] baf_outlier_dist;
  for (k in 1 : num_clones) {
    real s = 0;
    for (i in 1 : num_bins) 
      s += cn_t[i, k];
    mean_clone_cn[k] = s / num_bins;
  }
  for (i in 1 : num_bins){
    baf_outlier_dist[i] = beta_binomial_lpmf(a[i] | d[i], 1., 1.);
  }
}
parameters {
  // warning: order of these parameters should not be changed
  simplex[num_clones] rho;
  
  // to avoid numerical issues, the following parameters should not take tiny values
  real<lower=1e-6, upper=1> non_binomiality; // as ->0, we converge to binomial, higher values relaxes the binomial assumption
  
  // // these, on the other hand, are numerically stable even close to the boundaries [0, 1]
  real<lower=1e-6, upper=1> outlier_rate_baf; // probability a genomic bin is are considered outliers in the BAF data.  
}
model {
  rho ~ dirichlet(pi);
  non_binomiality ~ beta(1, 100); // prior on relatively low non_binomiality
  outlier_rate_baf ~ beta(1, 100); // prior on outlier rate for BAF data is set to low values (expect few genomic bins to be outliers)

  vector[num_bins] baf_a;
  vector[num_bins] baf_b;
  vector[num_bins] p;
  
  
  p = (cn_a * rho) ./ (cn_t * rho);
  
  baf_a = p / non_binomiality;
  
  baf_b = (1 - p) / non_binomiality;

  for (i in 1 : num_bins) {
    a[i] ~ baf(d[i], baf_a[i], baf_b[i], baf_outlier_dist[i], outlier_rate_baf);
    
  }
}
