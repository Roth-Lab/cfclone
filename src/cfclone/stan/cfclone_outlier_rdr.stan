functions {
  real rdr_lpdf(real rdr, real mu, real sigma, real oultier_lpdf, real outlier_rate) {
    return log_mix(outlier_rate, oultier_lpdf, student_t_lpdf(rdr | 25, mu, sigma));
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
  vector[num_bins] rdr_outlier_dist;
  for (k in 1 : num_clones) {
    real s = 0;
    for (i in 1 : num_bins) 
      s += cn_t[i, k];
    mean_clone_cn[k] = s / num_bins;
  }
  for (i in 1 : num_bins){
    rdr_outlier_dist[i] = student_t_lpdf(rdr[i] | 4, 1, 1);
  }
}
parameters {
  // warning: order of these parameters should not be changed
  simplex[num_clones] rho;
  real<lower=1e-6> alpha;
  
  // to avoid numerical issues, the following parameters should not take tiny values
  real<lower=1e-6> sigma;
  
  // // these, on the other hand, are numerically stable even close to the boundaries [0, 1]
  real<lower=1e-6, upper=1> outlier_rate_rdr; // probability a genomic bin is are considered outliers in the RDR data.
}
model {
  rho ~ dirichlet(pi);
  alpha ~ gamma(1, 1);
  sigma ~ gamma(1, 100); // prior on scale of student-t set to low values 
  outlier_rate_rdr ~ beta(1, 100); // prior on outlier rate for RDR data is set to low values (expect few genomic bins to be outliers)

  vector[num_bins] mu;
  
  mu = alpha * (cn_t * rho) / dot_product(mean_clone_cn, rho);
  
  for (i in 1 : num_bins) {
    rdr[i] ~ rdr(mu[i], sigma, rdr_outlier_dist[i], outlier_rate_rdr);
  }
}
