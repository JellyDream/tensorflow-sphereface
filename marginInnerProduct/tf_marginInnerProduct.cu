#include "caffe/math_functions.hpp"
namespace caffe {
template <typename Dtype>
__global__ void Weight_norm_gpu(int nthreads, int K_,
          Dtype* weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	Dtype sum_sqaure = 0.;
  	for (int i = 0; i < K_; i++) {
  	  sum_sqaure += weight[index * K_ + i] * weight[index * K_ + i];
  	}
  	sum_sqaure = sqrt(sum_sqaure);
    for (int i = 0; i < K_; i++) {
  	  weight[index * K_ + i] = weight[index * K_ + i] / sum_sqaure;
  	}
  }
}
template<typename Dtype>
__global__ void Compute_bottom_norm_gpu(int nthreads, int K_,
          Dtype* bottom, Dtype* x_norm) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype sum_sqaure = 0.;
    for (int i = 0; i < K_; i++) {
      sum_sqaure += bottom[index * K_ + i] * bottom[index * K_ + i];
    }
    x_norm[index] = sqrt(sum_sqaure);
  }
}

template <typename Dtype>
__global__ void Compute_cos_theta_gpu(int nthreads, int N_,
          Dtype* x_norm, Dtype* cos_theta) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index / N_;
    cos_theta[index] = cos_theta[index] / x_norm[i];
  }
}

template <typename Dtype>
__global__ void Compute_sign_1_gpu(int nthreads, Dtype* cos_theta, Dtype* sign_1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_1[index] = abs(cos_theta[index]) - (Dtype)0.5;
  }
}

template <typename Dtype>
__global__ void Compute_sign_2_gpu(int nthreads, Dtype* sign_0, 
	      Dtype* sign_1, Dtype* sign_2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_2[index] = sign_0[index] * ((Dtype)1. + sign_1[index]) - (Dtype)2.;
  }
}

template <typename Dtype>
__global__ void Compute_sign_3_gpu(int nthreads, Dtype* sign_0, 
	      Dtype* cos_theta_quadratic, Dtype* sign_3) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_3[index] = sign_0[index] * ((Dtype)2. * cos_theta_quadratic[index] - (Dtype)1.);
  }
}

template <typename Dtype>
__global__ void Compute_sign_4_gpu(int nthreads, Dtype* sign_0, 
	      Dtype* sign_3, Dtype* sign_4) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    sign_4[index] = (Dtype)2. * sign_0[index] + sign_3[index] - (Dtype)3.;
  }
}

template <typename Dtype>
__global__ void Margin_double_forward_gpu(int nthreads, int N_, Dtype lambda,
            const Dtype* label, Dtype* x_norm, Dtype* sign_0, 
            Dtype* cos_theta_quadratic, Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    int i = index / N_;
    int j = index % N_;
    int label_value = static_cast<int>(label[i]);
    if (label_value == j) {
      top[index] *= lambda;
      top[index] += x_norm[i] * ((Dtype)2. * sign_0[index] * cos_theta_quadratic[index] - 
      	                         (Dtype)1.);
      top[index] /= ((Dtype)1. + lambda);
    }
  }
}

template <typename Dtype>
__global__ void Margin_triple_forward_gpu(int nthreads, int N_, Dtype lambda,
            const Dtype* label, Dtype* x_norm, Dtype* sign_1, Dtype* sign_2,
            Dtype* cos_theta, Dtype* cos_theta_cubic,
            Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    int i = index / N_;
    int j = index % N_;
    int label_value = static_cast<int>(label[i]);
    if (label_value == j) {
      top[index] *= lambda;
      top[index] += x_norm[i] * (sign_1[index] * ((Dtype)4. * cos_theta_cubic[index] - 
      	                        (Dtype)3. * cos_theta[index]) + sign_2[index]);
      top[index] /= ((Dtype)1. + lambda);
    }
  }
}


template <typename Dtype>
__global__ void Margin_quadruple_forward_gpu(int nthreads, int N_, Dtype lambda,
            const Dtype* label, Dtype* x_norm, Dtype* sign_3, Dtype* sign_4,
            Dtype* cos_theta_quadratic, Dtype* cos_theta_quartic,
            Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the label[i]_th top_data
    int i = index / N_;
    int j = index % N_;
    int label_value = static_cast<int>(label[i]);
    if (label_value == j) {
      top[index] *= lambda;
      top[index] += x_norm[i] * (sign_3[index] * ((Dtype)8. * cos_theta_quartic[index] - 
      	            (Dtype)8. * cos_theta_quadratic[index] + (Dtype)1.) + sign_4[index]);
      top[index] /= ((Dtype)1. + lambda);
    }
  }
}

template <typename Dtype>
__global__ void Margin_bottom_double_backward_gpu(int nthreads, int N_, int K_, Dtype lambda,
            Dtype* bottom, Dtype* weight, const Dtype* top_diff, const Dtype* label,
            Dtype* x_norm, Dtype* sign_0, Dtype* cos_theta,
            Dtype* cos_theta_quadratic, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index / K_;
    int j = index % K_;
    bottom_diff[index] = (Dtype)0.;
    int label_value = static_cast<int>(label[i]);
    for (int n = 0; n < N_; n++) {
      if (label_value != n) {
        bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
      } else {
        Dtype coeff_w = (Dtype)4. * sign_0[i * N_ + n] * cos_theta[i * N_ + n];
        Dtype coeff_x = - (Dtype)1./ x_norm[i] * ((Dtype)2. * sign_0[i * N_ + n] *  
                     cos_theta_quadratic[i * N_ + n] + (Dtype)1.);
        Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
        coeff_w = coeff_w / coeff_norm;
        coeff_x = coeff_x / coeff_norm;
        bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
                              (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
        bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
      }
    }
  }
}


template <typename Dtype>
__global__ void Margin_bottom_triple_backward_gpu(int nthreads, int N_, int K_, Dtype lambda,
            Dtype* bottom, Dtype* weight, const Dtype* top_diff, const Dtype* label,
            Dtype* x_norm, Dtype* sign_1, Dtype* sign_2, Dtype* cos_theta_quadratic,
            Dtype* cos_theta_cubic, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index / K_;
    int j = index % K_;
    bottom_diff[index] = (Dtype)0.;
    int label_value = static_cast<int>(label[i]);
    for (int n = 0; n < N_; n++) {
      if (label_value != n) {
        bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
      } else {
        Dtype coeff_w = sign_1[i * N_ + n] * ((Dtype)12. * cos_theta_quadratic[i * N_ + n] - (Dtype)3.);
        Dtype coeff_x = - (Dtype)1./ x_norm[i] * ((Dtype)8. * sign_1[i * N_ + n] * cos_theta_cubic[i * N_ + n] - 
                    sign_2[i * N_ + n]);
        Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
        coeff_w = coeff_w / coeff_norm;
        coeff_x = coeff_x / coeff_norm;
        bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
                              (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
        bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
      }
    }
  }
}

template <typename Dtype>
__global__ void Margin_bottom_quadruple_backward_gpu(int nthreads, int N_, int K_, Dtype lambda,
            Dtype* bottom, Dtype* weight, const Dtype* top_diff, const Dtype* label,
            Dtype* x_norm, Dtype* sign_3, Dtype* sign_4,
            Dtype* cos_theta, Dtype* cos_theta_quadratic, 
            Dtype* cos_theta_cubic, Dtype* cos_theta_quartic, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index / K_;
    int j = index % K_;
    bottom_diff[index] = (Dtype)0.;
    int label_value = static_cast<int>(label[i]);
    for (int n = 0; n < N_; n++) {
      if (label_value != n) {
        bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j];
      } else {
        Dtype coeff_w = sign_3[i * N_ + n] * ((Dtype)32. * cos_theta_cubic[i * N_ + n] - (Dtype)16. * cos_theta[i * N_ + n]);
        Dtype coeff_x = - (Dtype)1./ x_norm[i] * (sign_3[i * N_ + n] * ((Dtype)24. * cos_theta_quartic[i * N_ + n] - 
                    (Dtype)8. * cos_theta_quadratic[i * N_ + n] - 1) - sign_4[i * N_ + n]);
        Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
        coeff_w = coeff_w / coeff_norm;
        coeff_x = coeff_x / coeff_norm;
        bottom_diff[index] += (Dtype)1./ ((Dtype)1. + lambda) * top_diff[i * N_ + n] * 
                              (coeff_w * weight[n * K_ + j] + coeff_x * bottom[index]);
        bottom_diff[index] += lambda / ((Dtype)1. + lambda) * top_diff[i * N_ + n] * weight[n * K_ + j];
      }
    }
  }
}

template <typename Dtype>
void MarginInnerProductKernelLauncher(int M_, int N_, int K_, Dtype* bottom_data, Dtype* weight, int m_value, Dtype lambda_, const Dtype* label, Dtype* x_norm_data, Dtype* sign_0_data, Dtype* sign_1_data, Dtype* sign_2_data, Dtype* sign_3_data, Dtype* sign_4_data, Dtype* cos_theta_data, Dtype* cos_theta_quadratic_data, Dtype* cos_theta_cubic_data, Dtype* cos_theta_quartic_data, Dtype* top_data){



  /************************* common variables *************************/
  // x_norm_ = |x|
  int nthreads = M_;
  Compute_bottom_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom_data,
                                x_norm_data);

  nthreads = M_ * N_;
  // cos_theta = x'w / |x|

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., cos_theta_data);
  Compute_cos_theta_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, x_norm_data, cos_theta_data);
  // sign_0
  caffe_gpu_sign(M_ * N_, cos_theta_data, sign_0_data);
  
  /************************* optional variables *************************/
  switch (m_value) {
  case 1:
    break;
  case 2:
    // cos_theta_quadratic
    caffe_gpu_powx(M_ * N_, cos_theta_data, (Dtype)2., cos_theta_quadratic_data);
    break;
  case 3:
    // cos_theta_quadratic && cos_theta_cubic
    caffe_gpu_powx(M_ * N_, cos_theta_data, (Dtype)2., cos_theta_quadratic_data);
    caffe_gpu_powx(M_ * N_, cos_theta_data, (Dtype)3., cos_theta_cubic_data);
    // sign_1 = sign(abs(cos_theta) - 0.5)
    Compute_sign_1_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, cos_theta_data, sign_1_data);
    caffe_gpu_sign(M_ * N_, sign_1_data, sign_1_data);
    // sign_2 = sign_0 * (1 + sign_1) - 2
    Compute_sign_2_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, sign_0_data,
                                sign_1_data, sign_2_data);
    break;
  case 4:
    // cos_theta_quadratic && cos_theta_cubic && cos_theta_quartic
    caffe_gpu_powx(M_ * N_, cos_theta_data, (Dtype)2., cos_theta_quadratic_data);
    caffe_gpu_powx(M_ * N_, cos_theta_data, (Dtype)3., cos_theta_cubic_data);
    caffe_gpu_powx(M_ * N_, cos_theta_data, (Dtype)4., cos_theta_quartic_data);
    // sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
    Compute_sign_3_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, sign_0_data, cos_theta_quadratic_data,
                                sign_3_data);
    caffe_gpu_sign(M_ * N_, sign_3_data, sign_3_data);
    // sign_4 = 2 * sign_0 + sign_3 - 3
    Compute_sign_4_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, sign_0_data,
                                sign_3_data, sign_4_data);

    break;
  }

  /************************* Forward *************************/
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  switch (m_value) {
  case 1:
    break;
  case 2:
    // caffe_gpu_memcpy(M_ * N_, cos_theta_data, top_data);
    Margin_double_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, lambda_, label, x_norm_data, 
      	                        sign_0_data, cos_theta_quadratic_data, top_data);
    break;
  case 3:
    Margin_triple_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, lambda_, label, x_norm_data, sign_1_data, 
                                sign_2_data, cos_theta_data, 
                                cos_theta_cubic_data, top_data);
    break;
  case 4:
    Margin_quadruple_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, lambda_, label, x_norm_data, sign_3_data, 
                                sign_4_data, cos_theta_quadratic_data, 
                                cos_theta_quartic_data, top_data);
    
    break;
  }
}

template <typename Dtype>
void MarginInnerProductGradKernelLauncher(int M_, int N_, int K_, Dtype* bottom_data, Dtype* weight, int m_value, Dtype lambda_, const Dtype* label, Dtype* x_norm_data, Dtype* sign_0_data, Dtype* sign_1_data, Dtype* sign_2_data, Dtype* sign_3_data, Dtype* sign_4_data, Dtype* cos_theta_data, Dtype* cos_theta_quadratic_data, Dtype* cos_theta_cubic_data, Dtype* cos_theta_quartic_data, const Dtype* top_diff, Dtype* bottom_diff, Dtype* weight_diff){



    // Gradient with respect to bottom data
    int nthreads = M_ * K_;
    switch (m_value) {
    case 1:
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weight, (Dtype)0.,
        bottom_diff);
      break;
    case 2:
      Margin_bottom_double_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, K_, lambda_, bottom_data, weight, top_diff, label,
                                  x_norm_data, sign_0_data, 
                                  cos_theta_data, cos_theta_quadratic_data,                                  
                                  bottom_diff);
      break;
    case 3:
      Margin_bottom_triple_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, K_, lambda_, bottom_data, weight, top_diff, label,
                                  x_norm_data, sign_1_data, sign_2_data,
                                  cos_theta_quadratic_data, cos_theta_cubic_data,
                                  bottom_diff);
      break;
    case 4:
      Margin_bottom_quadruple_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, K_, lambda_, bottom_data, weight, top_diff, label,
                                  x_norm_data, sign_3_data, sign_4_data,
                                  cos_theta_data, cos_theta_quadratic_data,
                                  cos_theta_cubic_data, cos_theta_quartic_data,
                                  bottom_diff);
      break;
  }
}

template void MarginInnerProductKernelLauncher<float>(int M_, int N_, int K_, float* bottom_data, float* weight, int m_value, float lambda_,const float* label, float* x_norm_data, float* sign_0_data, float* sign_1_data, float* sign_2_data, float* sign_3_data, float* sign_4_data, float* cos_theta_data, float* cos_theta_quadratic_data, float* cos_theta_cubic_data, float* cos_theta_quartic_data, float* top_data);

template void MarginInnerProductGradKernelLauncher<float>(int M_, int N_, int K_, float* bottom_data, float* weight, int m_value, float lambda_,const float* label, float* x_norm_data, float* sign_0_data, float* sign_1_data, float* sign_2_data, float* sign_3_data, float* sign_4_data, float* cos_theta_data, float* cos_theta_quadratic_data, float* cos_theta_cubic_data, float* cos_theta_quartic_data, const float* top_diff, float* bottom_diff, float* weight_diff);
}  // namespace caffe




