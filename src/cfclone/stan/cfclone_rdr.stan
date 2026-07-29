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
  real<lower=1e-6> alpha;
  real<lower=1e-6> sigma;
}
model {
  rho ~ dirichlet(pi);
  alpha ~ gamma(1, 1);
  sigma ~ gamma(1, 100); // prior on scale of student-t set to low values 
  
  vector[num_bins] baf_a;
  vector[num_bins] baf_b;
  vector[num_bins] mu;
  vector[num_bins] p;
  
  mu = alpha * (cn_t * rho) / dot_product(mean_clone_cn, rho);
  rdr ~ student_t(25, mu, sigma);
}
