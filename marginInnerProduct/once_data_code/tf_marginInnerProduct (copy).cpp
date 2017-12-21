#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "caffe/math_functions.hpp"
REGISTER_OP("MarginInnerProduct")
    .Input("input:float32")
    .Input("weight:float32")
    .Input("m_value:int32")
    .Input("lambda:float32")
    .Input("label:int32")
    .Output("output:float32");
REGISTER_OP("MarginInnerProductGrad")
    .Input("input:float32")
    .Input("weight:float32")
    .Input("m_value:int32")
    .Input("lambda:float32")
    .Input("label:int32")
    .Input("grad_output:float32")
    .Output("grad_input:float32")
    .Output("grad_weight:float32");
using namespace tensorflow;
using namespace caffe;
using Dtype = float;
class MarginInnerProductOp : public OpKernel{
    public:
        
        int M_ = 0 , N_ = 0, K_ = 0;
        /*
        std::shared_ptr<Dtype> x_norm_ptr;
        std::shared_ptr<Dtype> cos_theta_ptr;
        std::shared_ptr<Dtype> sign_0_ptr;
        std::shared_ptr<Dtype> cos_theta_quadratic_ptr;
        std::shared_ptr<Dtype> sign_1_ptr;
        std::shared_ptr<Dtype> sign_2_ptr;
        std::shared_ptr<Dtype> cos_theta_cubic_ptr;
        std::shared_ptr<Dtype> sign_3_ptr;
        std::shared_ptr<Dtype> sign_4_ptr;
        std::shared_ptr<Dtype> cos_theta_quartic_ptr;
        */
    public:
        explicit MarginInnerProductOp(OpKernelConstruction* context) : OpKernel(context){}
        void Compute(OpKernelContext* context) override{
            const Tensor& input_tensor = context->input(0);
            const Tensor& weight_tensor = context->input(1);
            const Tensor& m_value_tensor = context->input(2);
            const Tensor& lambda_tensor = context->input(3);
            const Tensor& label_tensor = context->input(4);
            OP_REQUIRES(context, input_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires input be of shape (batch, features-dim)"));
            OP_REQUIRES(context, weight_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires weight be of shape (num_output, features)"));
            OP_REQUIRES(context, weight_tensor.shape().dim_size(1) == input_tensor.shape().dim_size(1), errors::InvalidArgument("marginInnerProduct requires the shape of input and weight must be consistent(input:batch*features-dim, weight:num_output*features-dim)"));
            OP_REQUIRES(context, m_value_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, m_value_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            auto input_flat = input_tensor.flat<Dtype>();
            const Dtype* temp_input = &input_flat(0);
            auto weight_flat = weight_tensor.flat<Dtype>();
            const Dtype* temp_weight = &weight_flat(0);
            //Dtype* input = input_tensor.flat<Dtype>();
            //Dtype* weight = weight_tensor.flat<Dtype>();
            int m_value = m_value_tensor.flat<int32>()(0);
            Dtype lambda_ = lambda_tensor.flat<Dtype>()(0);
            const int* label = &label_tensor.flat<int32>()(0);
            //common constant variable
            M_ = input_tensor.shape().dim_size(0);
            N_ = weight_tensor.shape().dim_size(0);
            K_ = input_tensor.shape().dim_size(1);

            Tensor bottom_data_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_, K_}, &bottom_data_tensor));
            Dtype* bottom_data = &bottom_data_tensor.flat<Dtype>()(0);

            Tensor weight_data_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{N_, K_}, &weight_data_tensor));
            Dtype* weight = &weight_data_tensor.flat<Dtype>()(0);

            caffe_copy(M_ * K_, temp_input, bottom_data);
            caffe_copy(N_ * K_, temp_weight, weight);


            //output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{M_ , N_},&output_tensor));
            auto top_data = &output_tensor->flat<Dtype>()(0);


            
            /************allocate memory for variable*******/
            int top_size = M_ * N_;

            Tensor x_norm_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_}, &x_norm_tensor));
            Dtype* x_norm_data = &x_norm_tensor.flat<Dtype>()(0);

            Tensor sign_0_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_0_tensor));
            Dtype* sign_0_data = &sign_0_tensor.flat<Dtype>()(0);

            Tensor sign_1_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_1_tensor));
            Dtype* sign_1_data = &sign_1_tensor.flat<Dtype>()(0);

            Tensor sign_2_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_2_tensor));
            Dtype* sign_2_data = &sign_2_tensor.flat<Dtype>()(0);

            Tensor sign_3_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_3_tensor));
            Dtype* sign_3_data = &sign_3_tensor.flat<Dtype>()(0);

            Tensor sign_4_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_4_tensor));
            Dtype* sign_4_data = &sign_4_tensor.flat<Dtype>()(0);

            Tensor cos_theta_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_tensor));
            Dtype* cos_theta_data = &cos_theta_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quadratic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quadratic_tensor));
            Dtype* cos_theta_quadratic_data = &cos_theta_quadratic_tensor.flat<Dtype>()(0);
            Tensor cos_theta_cubic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_cubic_tensor));
            Dtype* cos_theta_cubic_data = &cos_theta_cubic_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quartic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quartic_tensor));
            Dtype* cos_theta_quartic_data = &cos_theta_quartic_tensor.flat<Dtype>()(0);


  /************************* normalize weight *************************/
  Dtype temp_norm = (Dtype)0.;
  for (int i = 0; i < N_; i++) {
  	temp_norm = caffe_cpu_dot(K_, weight + i * K_, weight + i * K_);
  	temp_norm = (Dtype)1./sqrt(temp_norm);
  	caffe_scal(K_, temp_norm, weight + i * K_);
  }

  /************************* common variables *************************/
  // x_norm_ = |x|
  for (int i = 0; i < M_; i++) {
    x_norm_data[i] = sqrt(caffe_cpu_dot(K_, bottom_data + i * K_, bottom_data + i * K_));
  }
  // cos_theta = x'w/|x|
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., cos_theta_data);
  for (int i = 0; i < M_; i++) {
    caffe_scal(N_, (Dtype)1./x_norm_data[i], cos_theta_data + i * N_);
  }
  // sign_0 = sign(cos_theta)
  caffe_cpu_sign(M_ * N_, cos_theta_data, sign_0_data);

            /************************* optional variables *************************/
            switch (m_value) {
            case 1:
              break;
            case 2:
              // cos_theta_quadratic
              caffe_powx(M_ * N_, cos_theta_data, (Dtype)2., cos_theta_quadratic_data);
              break;
            case 3:
              // cos_theta_quadratic && cos_theta_cubic
              caffe_powx(M_ * N_, cos_theta_data, (Dtype)2., cos_theta_quadratic_data);
              caffe_powx(M_ * N_, cos_theta_data, (Dtype)3., cos_theta_cubic_data);
              // sign_1 = sign(abs(cos_theta) - 0.5)
              caffe_abs(M_ * N_, cos_theta_data, sign_1_data);
              caffe_add_scalar(M_ * N_, -(Dtype)0.5, sign_1_data);
              caffe_cpu_sign(M_ * N_, sign_1_data, sign_1_data);
              // sign_2 = sign_0 * (1 + sign_1) - 2
              caffe_copy(M_ * N_, sign_1_data, sign_2_data);
              caffe_add_scalar(M_ * N_, (Dtype)1., sign_2_data);
              caffe_mul(M_ * N_, sign_0_data, sign_2_data, sign_2_data);
              caffe_add_scalar(M_ * N_, - (Dtype)2., sign_2_data);
              break;
            case 4:
              // cos_theta_quadratic && cos_theta_cubic && cos_theta_quartic
              caffe_powx(M_ * N_, cos_theta_data, (Dtype)2., cos_theta_quadratic_data);
              caffe_powx(M_ * N_, cos_theta_data, (Dtype)3., cos_theta_cubic_data);
              caffe_powx(M_ * N_, cos_theta_data, (Dtype)4., cos_theta_quartic_data);
              // sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
              caffe_copy(M_ * N_, cos_theta_quadratic_data, sign_3_data);
              caffe_scal(M_ * N_, (Dtype)2., sign_3_data);
              caffe_add_scalar(M_ * N_, (Dtype)-1., sign_3_data);
              caffe_cpu_sign(M_ * N_, sign_3_data, sign_3_data);
              caffe_mul(M_ * N_, sign_0_data, sign_3_data, sign_3_data);
              // sign_4 = 2 * sign_0 + sign_3 - 3
              caffe_copy(M_ * N_, sign_0_data, sign_4_data);
              caffe_scal(M_ * N_, (Dtype)2., sign_4_data);
              caffe_add(M_ * N_, sign_4_data, sign_3_data, sign_4_data);
              caffe_add_scalar(M_ * N_, - (Dtype)3., sign_4_data);
              break;
            default:
              std::cout  << "Unknown margin type." << std::endl;
            }
          
            /************************* Forward *************************/
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                bottom_data, weight, (Dtype)0., top_data);
              switch (m_value) {
            case 1: {
              break;
            }
            case 2: {
            	//const Dtype* sign_0_data = sign_0_data;
            	//const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_data;
              // the label[i]_th top_data
              for (int i = 0; i < M_; i++) {
                const int label_value = static_cast<int>(label[i]);
                // |x| * (2 * sign_0 * cos_theta_quadratic - 1)
                top_data[i * N_ + label_value] = x_norm_data[i] * ((Dtype)2. * sign_0_data[i * N_ + label_value] * 
                                                 cos_theta_quadratic_data[i * N_ + label_value] - (Dtype)1.);
              }
              // + lambda * x'w
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, lambda_,
                bottom_data, weight, (Dtype)1., top_data);
              // * 1 / (1 + lambda)
              caffe_scal(M_ * N_, (Dtype)1./((Dtype)1. + lambda_), top_data);
              break;
            }
            case 3: {
            	//const Dtype* sign_1_data = sign_1_data;
              //const Dtype* sign_2_data = sign_2_data;
              //const Dtype* cos_theta_data = cos_theta_data;
              //const Dtype* cos_theta_cubic_data = cos_theta_cubic_data;
              // the label[i]_th output
              for (int i = 0; i < M_; i++) {
                const int label_value = static_cast<int>(label[i]);
                // |x| * (sign_1 * (4 * cos_theta_cubic - 3 * cos_theta) + sign_2)
                top_data[i * N_ + label_value] = x_norm_data[i] * (sign_1_data[i * N_ + label_value] * 
                                                ((Dtype)4. * cos_theta_cubic_data[i * N_ + label_value] - 
                                                 (Dtype)3. * cos_theta_data[i * N_ + label_value]) + 
                                                 sign_2_data[i * N_ + label_value]);
              }
              // + lambda * x'w
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, lambda_,
                bottom_data, weight, (Dtype)1., top_data);
              // / (1 + lambda)
              caffe_scal(M_ * N_, (Dtype)1./((Dtype)1. + lambda_), top_data);
              break;
            }
            case 4: {
            	//const Dtype* sign_3_data = sign_3_data;
              //const Dtype* sign_4_data = sign_4_data;
              //const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_data;
              //const Dtype* cos_theta_quartic_data = cos_theta_quartic_data;
              // the label[i]_th output
              for (int i = 0; i < M_; i++) {
                const int label_value = static_cast<int>(label[i]);
                // // |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
                top_data[i * N_ + label_value] = x_norm_data[i] * (sign_3_data[i * N_ + label_value] * 
                                                 ((Dtype)8. * cos_theta_quartic_data[i * N_ + label_value] - 
                                                  (Dtype)8. * cos_theta_quadratic_data[i * N_ + label_value] + 
                                                  (Dtype)1.) + sign_4_data[i * N_ + label_value]);
              }
              // + lambda * x'w
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, lambda_,
                bottom_data, weight, (Dtype)1., top_data);
              // / (1 + lambda)
              caffe_scal(M_ * N_, (Dtype)1./((Dtype)1. + lambda_), top_data);
              break;
            }
            default: {
                std::cout << "Unknown margin type.";
            }
        }
        //caffe_set(M_ * N_, Dtype(0), top_data);
    }
};
REGISTER_KERNEL_BUILDER(Name("MarginInnerProduct").Device(DEVICE_CPU), MarginInnerProductOp);

class MarginInnerProductGradOp : public OpKernel{
    public:
        int M_ = 0 , N_ = 0, K_ = 0;
        /*
        std::shared_ptr<Dtype> x_norm_ptr;
        std::shared_ptr<Dtype> cos_theta_ptr;
        std::shared_ptr<Dtype> sign_0_ptr;
        std::shared_ptr<Dtype> cos_theta_quadratic_ptr;
        std::shared_ptr<Dtype> sign_1_ptr;
        std::shared_ptr<Dtype> sign_2_ptr;
        std::shared_ptr<Dtype> cos_theta_cubic_ptr;
        std::shared_ptr<Dtype> sign_3_ptr;
        std::shared_ptr<Dtype> sign_4_ptr;
        std::shared_ptr<Dtype> cos_theta_quartic_ptr;
        */
    public:
        explicit MarginInnerProductGradOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext* context) override{
            const Tensor& input_tensor = context->input(0);
            const Tensor& weight_tensor = context->input(1);
            const Tensor& m_value_tensor = context->input(2);
            const Tensor& lambda_tensor = context->input(3);
            const Tensor& label_tensor = context->input(4);
            OP_REQUIRES(context, input_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires input be of shape (batch, features-dim)"));
            OP_REQUIRES(context, weight_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires weight be of shape (num_output, features)"));
            OP_REQUIRES(context, weight_tensor.shape().dim_size(1) == input_tensor.shape().dim_size(1), errors::InvalidArgument("marginInnerProduct requires the shape of input and weight must be consistent(input:batch*features-dim, weight:num_output*features-dim)"));
            OP_REQUIRES(context, m_value_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, m_value_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            auto input_flat = input_tensor.flat<Dtype>();
            const Dtype* temp_input = &input_flat(0);
            auto weight_flat = weight_tensor.flat<Dtype>();
            const Dtype* temp_weight = &weight_flat(0);
            int m_value = m_value_tensor.flat<int32>()(0);
            Dtype lambda_ = lambda_tensor.flat<Dtype>()(0);
            const int* label = &label_tensor.flat<int32>()(0);
            //common constant variable
            M_ = input_tensor.shape().dim_size(0);
            N_ = weight_tensor.shape().dim_size(0);
            K_ = input_tensor.shape().dim_size(1);


            Tensor bottom_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_, K_}, &bottom_tensor));
            Dtype* bottom_data = &bottom_tensor.flat<Dtype>()(0);

            Tensor weight_data_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{N_, K_}, &weight_data_tensor));
            Dtype* weight = &weight_data_tensor.flat<Dtype>()(0);

            caffe_copy(M_ * K_, temp_input, bottom_data);
            caffe_copy(N_ * K_, temp_weight, weight);


            /************allocate memory for variable*******/
            int top_size = M_ * N_;

            Tensor x_norm_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_}, &x_norm_tensor));
            Dtype* x_norm_data = &x_norm_tensor.flat<Dtype>()(0);

            Tensor sign_0_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_0_tensor));
            Dtype* sign_0_data = &sign_0_tensor.flat<Dtype>()(0);

            Tensor sign_1_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_1_tensor));
            Dtype* sign_1_data = &sign_1_tensor.flat<Dtype>()(0);

            Tensor sign_2_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_2_tensor));
            Dtype* sign_2_data = &sign_2_tensor.flat<Dtype>()(0);

            Tensor sign_3_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_3_tensor));
            Dtype* sign_3_data = &sign_3_tensor.flat<Dtype>()(0);

            Tensor sign_4_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_4_tensor));
            Dtype* sign_4_data = &sign_4_tensor.flat<Dtype>()(0);

            Tensor cos_theta_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_tensor));
            Dtype* cos_theta_data = &cos_theta_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quadratic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quadratic_tensor));
            Dtype* cos_theta_quadratic_data = &cos_theta_quadratic_tensor.flat<Dtype>()(0);
            Tensor cos_theta_cubic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_cubic_tensor));
            Dtype* cos_theta_cubic_data = &cos_theta_cubic_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quartic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quartic_tensor));
            Dtype* cos_theta_quartic_data = &cos_theta_quartic_tensor.flat<Dtype>()(0);




  /************************* normalize weight *************************/
  Dtype temp_norm = (Dtype)0.;
  std::vector<Dtype> temp_norm_vector;
  for (int i = 0; i < N_; i++) {
  	temp_norm = caffe_cpu_dot(K_, weight + i * K_, weight + i * K_);
  	temp_norm = (Dtype)1./sqrt(temp_norm);
  	caffe_scal(K_, temp_norm, weight + i * K_);
        temp_norm_vector.push_back(temp_norm);
  }



            //grad of output
            const Tensor& grad_output_tensor = context->input(5);
            const Dtype* top_diff = &grad_output_tensor.flat<Dtype>()(0);
            //grad for input
            Tensor * grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{M_, K_}, &grad_input_tensor));
            Tensor * grad_weight_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{N_, K_}, &grad_weight_tensor));
            Dtype* bottom_diff = &grad_input_tensor->flat<Dtype>()(0);
            Dtype* weight_diff = &grad_weight_tensor->flat<Dtype>()(0);
            //Gradient with respect to weight
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1., top_diff, bottom_data, (Dtype)0., weight_diff);
            //adjust the gradient for tensorflow
  for (int i = 0; i < N_; i++) {
  	temp_norm = (Dtype)1./temp_norm_vector[i];
  	caffe_scal(K_, temp_norm, weight_diff + i * K_);
  }
        
            //gradient with respect to bottom data
            //const Dtype* x_norm_data = x_norm_data;
            caffe_set(M_ * K_, Dtype(0), bottom_diff);
            switch (m_value) {
                case 1: {
                  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                    top_diff, weight, (Dtype)0.,
                    bottom_diff);
                  break;
                }
                case 2: {
                  const Dtype* sign_0_data = sign_0_data;
                  const Dtype* cos_theta_data = cos_theta_data;
                  const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_data;
                  for (int i = 0; i < M_; i++) {
                    const int label_value = static_cast<int>(label[i]);
                    for (int j = 0; j < N_; j++) {
                      if (label_value != j) {
                        // 1 / (1 + lambda) * w
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j], 
                                        weight + j * K_, (Dtype)1., bottom_diff + i * K_);
                      } else {
                        // 4 * sign_0 * cos_theta * w
                        Dtype coeff_w = (Dtype)4. * sign_0_data[i * N_ + j] * cos_theta_data[i * N_ + j];
                        // 1 / (-|x|) * (2 * sign_0 * cos_theta_quadratic + 1) * x
                        Dtype coeff_x = (Dtype)1. / (-x_norm_data[i]) * ((Dtype)2. * 
                                        sign_0_data[i * N_ + j] * cos_theta_quadratic_data[i * N_ + j] + (Dtype)1.);
                        Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
                        coeff_w = coeff_w / coeff_norm;
                        coeff_x = coeff_x / coeff_norm;
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * coeff_w, 
                                        weight + j * K_, (Dtype)1., bottom_diff + i * K_);
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * coeff_x, 
                                        bottom_data + i * K_, (Dtype)1., bottom_diff + i * K_);
                      }
                    }
                  }
                  // + lambda/(1 + lambda) * w
                  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, lambda_/((Dtype)1. + lambda_),
                    top_diff, weight, (Dtype)1.,
                    bottom_diff);
                  break;
                }
                case 3: {
                  //const Dtype* sign_1_data = sign_1_data;
                  //const Dtype* sign_2_data = sign_2_data;
                  //const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_data;
                  //const Dtype* cos_theta_cubic_data = cos_theta_cubic_data;
                  for (int i = 0; i < M_; i++) {
                    const int label_value = static_cast<int>(label[i]);
                    for (int j = 0; j < N_; j++) {
                      if (label_value != j) {
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j], 
                                        weight + j * K_, (Dtype)1., bottom_diff + i * K_);
                      } else {
                        // sign_1 * (12 * cos_theta_quadratic - 3) * w
                        Dtype coeff_w = sign_1_data[i * N_ + j] * ((Dtype)12. * 
                                        cos_theta_quadratic_data[i * N_ + j] - (Dtype)3.);
                        // 1 / (-|x|) * (8 * sign_1 * cos_theta_cubic - sign_2) * x
                        Dtype coeff_x = (Dtype)1. / (-x_norm_data[i]) * ((Dtype)8. * sign_1_data[i * N_ + j] * 
                                          cos_theta_cubic_data[i * N_ + j] - sign_2_data[i * N_ +j]);
                        Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
                        coeff_w = coeff_w / coeff_norm;
                        coeff_x = coeff_x / coeff_norm;
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * coeff_w, 
                                        weight + j * K_, (Dtype)1., bottom_diff + i * K_);
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * coeff_x, 
                                        bottom_data + i * K_, (Dtype)1., bottom_diff + i * K_);
                      }
                    }
                  }
                  // + lambda/(1 + lambda) * w
                  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, lambda_/((Dtype)1. + lambda_),
                    top_diff, weight, (Dtype)1.,
                    bottom_diff);
                  break;
                }
                case 4: {
                  /*
                  const Dtype* sign_3_data = sign_3_data;
                  const Dtype* sign_4_data = sign_4_data;
                  const Dtype* cos_theta_data = cos_theta_data;
                  const Dtype* cos_theta_quadratic_data = cos_theta_quadratic_data;
                  const Dtype* cos_theta_cubic_data = cos_theta_cubic_data;
                  const Dtype* cos_theta_quartic_data = cos_theta_quartic_data;
                  */
                  for (int i = 0; i < M_; i++) {
                    const int label_value = static_cast<int>(label[i]);
                    for (int j = 0; j < N_; j++) {
                      if (label_value != j) {
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j], 
                                        weight + j * K_, (Dtype)1., bottom_diff + i * K_);
                      } else {
                        // 1 / (1 + lambda) * sign_3 * (32 * cos_theta_cubic - 16 * cos_theta) * w
                        Dtype coeff_w = sign_3_data[i * N_ + j] * ((Dtype)32. * cos_theta_cubic_data[i * N_ + j] -
                                            (Dtype)16. * cos_theta_data[i * N_ + j]);
                        // 1 / (-|x|) * (sign_3 * (24 * cos_theta_quartic - 8 * cos_theta_quadratic - 1) + 
                        //                        sign_4) * x
                        Dtype coeff_x = (Dtype)1. / (-x_norm_data[i]) * (sign_3_data[i * N_ + j] * 
                                        ((Dtype)24. * cos_theta_quartic_data[i * N_ + j] - 
                                        (Dtype)8. * cos_theta_quadratic_data[i * N_ + j] - (Dtype)1.) - 
                                         sign_4_data[i * N_ + j]);
                        Dtype coeff_norm = sqrt(coeff_w * coeff_w + coeff_x * coeff_x);
                        coeff_w = coeff_w / coeff_norm;
                        coeff_x = coeff_x / coeff_norm;
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * coeff_w, 
                                        weight + j * K_, (Dtype)1., bottom_diff + i * K_);
                        caffe_cpu_axpby(K_, (Dtype)1. / ((Dtype)1. + lambda_) * top_diff[i * N_ + j] * coeff_x, 
                                        bottom_data + i * K_, (Dtype)1., bottom_diff + i * K_);
                      }
                    }
                  }
                  // + lambda/(1 + lambda) * w
                  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, lambda_/((Dtype)1. + lambda_),
                    top_diff, weight, (Dtype)1.,
                    bottom_diff);
                  break;
                }
                default: {
                    std::cout << "Unknown margin type.";
                }
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("MarginInnerProductGrad").Device(DEVICE_CPU), MarginInnerProductGradOp);

/*
void MarginInnerProductKernelLauncher(Dtype* bottom_data, Dtype* weight, int m_value, Dtype lambda_, const int* label, Dtype* top_data, Dtype* x_norm_data, Dtype* sign_0_data, Dtype* sign_1_data, Dtype* sign_2_data, Dtype* sign_3_data, Dtype* sign_4_data, Dtype* cos_theta_data, Dtype* cos_theta_quadratic_data, Dtype* cos_theta_cubic_data, Dtype* cos_theta_quartic_data);

void MarginInnerProductGradKernelLauncher(Dtype* bottom_data, Dtype* weight, int m_value, Dtype lambda_, const int* label, Dtype* top_data, Dtype* x_norm_data, Dtype* sign_0_data, Dtype* sign_1_data, Dtype* sign_2_data, Dtype* sign_3_data, Dtype* sign_4_data, Dtype* cos_theta_data, Dtype* cos_theta_quadratic_data, Dtype* cos_theta_cubic_data, Dtype* cos_theta_quartic_data);
*/

class MarginInnerProductGpuOp : public OpKernel{
    public:
        
        int M_ = 0 , N_ = 0, K_ = 0;
    public:
        explicit MarginInnerProductGpuOp(OpKernelConstruction* context) : OpKernel(context){}
        void Compute(OpKernelContext* context) override{
            const Tensor& input_tensor = context->input(0);
            const Tensor& weight_tensor = context->input(1);
            const Tensor& m_value_tensor = context->input(2);
            const Tensor& lambda_tensor = context->input(3);
            const Tensor& label_tensor = context->input(4);
            OP_REQUIRES(context, input_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires input be of shape (batch, features-dim)"));
            OP_REQUIRES(context, weight_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires weight be of shape (num_output, features)"));
            OP_REQUIRES(context, weight_tensor.shape().dim_size(1) == input_tensor.shape().dim_size(1), errors::InvalidArgument("marginInnerProduct requires the shape of input and weight must be consistent(input:batch*features-dim, weight:num_output*features-dim)"));
            OP_REQUIRES(context, m_value_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, m_value_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            auto input_flat = input_tensor.flat<Dtype>();
            const Dtype* temp_input = &input_flat(0);
            auto weight_flat = weight_tensor.flat<Dtype>();
            const Dtype* temp_weight = &weight_flat(0);
            //Dtype* input = input_tensor.flat<Dtype>();
            //Dtype* weight = weight_tensor.flat<Dtype>();
            int m_value = m_value_tensor.flat<int32>()(0);
            Dtype lambda_ = lambda_tensor.flat<Dtype>()(0);
            const int* label = &label_tensor.flat<int32>()(0);
            //common constant variable
            M_ = input_tensor.shape().dim_size(0);
            N_ = weight_tensor.shape().dim_size(0);
            K_ = input_tensor.shape().dim_size(1);

            Tensor bottom_data_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_, K_}, &bottom_data_tensor));
            Dtype* bottom_data = &bottom_data_tensor.flat<Dtype>()(0);

            Tensor weight_data_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{N_, K_}, &weight_data_tensor));
            Dtype* weight = &weight_data_tensor.flat<Dtype>()(0);

            caffe_gpu_copy(M_ * K_, temp_input, bottom_data);
            caffe_gpu_copy(N_ * K_, temp_weight, weight);


            //output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{M_ , N_},&output_tensor));
            auto top_data = &output_tensor->flat<Dtype>()(0);


            
            /************allocate memory for variable*******/
            int top_size = M_ * N_;

            Tensor x_norm_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_}, &x_norm_tensor));
            Dtype* x_norm_data = &x_norm_tensor.flat<Dtype>()(0);

            Tensor sign_0_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_0_tensor));
            Dtype* sign_0_data = &sign_0_tensor.flat<Dtype>()(0);

            Tensor sign_1_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_1_tensor));
            Dtype* sign_1_data = &sign_1_tensor.flat<Dtype>()(0);

            Tensor sign_2_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_2_tensor));
            Dtype* sign_2_data = &sign_2_tensor.flat<Dtype>()(0);

            Tensor sign_3_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_3_tensor));
            Dtype* sign_3_data = &sign_3_tensor.flat<Dtype>()(0);

            Tensor sign_4_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_4_tensor));
            Dtype* sign_4_data = &sign_4_tensor.flat<Dtype>()(0);

            Tensor cos_theta_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_tensor));
            Dtype* cos_theta_data = &cos_theta_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quadratic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quadratic_tensor));
            Dtype* cos_theta_quadratic_data = &cos_theta_quadratic_tensor.flat<Dtype>()(0);
            Tensor cos_theta_cubic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_cubic_tensor));
            Dtype* cos_theta_cubic_data = &cos_theta_cubic_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quartic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quartic_tensor));
            Dtype* cos_theta_quartic_data = &cos_theta_quartic_tensor.flat<Dtype>()(0);


    }
};
REGISTER_KERNEL_BUILDER(Name("MarginInnerProduct").Device(DEVICE_GPU), MarginInnerProductOp);

class MarginInnerProductGradGpuOp : public OpKernel{
    public:
        int M_ = 0 , N_ = 0, K_ = 0;
    public:
        explicit MarginInnerProductGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext* context) override{
            const Tensor& input_tensor = context->input(0);
            const Tensor& weight_tensor = context->input(1);
            const Tensor& m_value_tensor = context->input(2);
            const Tensor& lambda_tensor = context->input(3);
            const Tensor& label_tensor = context->input(4);
            OP_REQUIRES(context, input_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires input be of shape (batch, features-dim)"));
            OP_REQUIRES(context, weight_tensor.dims() == 2, errors::InvalidArgument("marginInnerProduct requires weight be of shape (num_output, features)"));
            OP_REQUIRES(context, weight_tensor.shape().dim_size(1) == input_tensor.shape().dim_size(1), errors::InvalidArgument("marginInnerProduct requires the shape of input and weight must be consistent(input:batch*features-dim, weight:num_output*features-dim)"));
            OP_REQUIRES(context, m_value_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, m_value_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires m_value be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.dims() == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            OP_REQUIRES(context, lambda_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("marginInnerProduct requires lambda be of shape (1)"));
            auto input_flat = input_tensor.flat<Dtype>();
            const Dtype* temp_input = &input_flat(0);
            auto weight_flat = weight_tensor.flat<Dtype>();
            const Dtype* temp_weight = &weight_flat(0);
            int m_value = m_value_tensor.flat<int32>()(0);
            Dtype lambda_ = lambda_tensor.flat<Dtype>()(0);
            const int* label = &label_tensor.flat<int32>()(0);
            //common constant variable
            M_ = input_tensor.shape().dim_size(0);
            N_ = weight_tensor.shape().dim_size(0);
            K_ = input_tensor.shape().dim_size(1);


            Tensor bottom_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_, K_}, &bottom_tensor));
            Dtype* bottom_data = &bottom_tensor.flat<Dtype>()(0);

            Tensor weight_data_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{N_, K_}, &weight_data_tensor));
            Dtype* weight = &weight_data_tensor.flat<Dtype>()(0);

            caffe_gpu_copy(M_ * K_, temp_input, bottom_data);
            caffe_gpu_copy(N_ * K_, temp_weight, weight);


            /************allocate memory for variable*******/
            int top_size = M_ * N_;

            Tensor x_norm_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_}, &x_norm_tensor));
            Dtype* x_norm_data = &x_norm_tensor.flat<Dtype>()(0);

            Tensor sign_0_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_0_tensor));
            Dtype* sign_0_data = &sign_0_tensor.flat<Dtype>()(0);

            Tensor sign_1_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_1_tensor));
            Dtype* sign_1_data = &sign_1_tensor.flat<Dtype>()(0);

            Tensor sign_2_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_2_tensor));
            Dtype* sign_2_data = &sign_2_tensor.flat<Dtype>()(0);

            Tensor sign_3_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_3_tensor));
            Dtype* sign_3_data = &sign_3_tensor.flat<Dtype>()(0);

            Tensor sign_4_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &sign_4_tensor));
            Dtype* sign_4_data = &sign_4_tensor.flat<Dtype>()(0);

            Tensor cos_theta_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_tensor));
            Dtype* cos_theta_data = &cos_theta_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quadratic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quadratic_tensor));
            Dtype* cos_theta_quadratic_data = &cos_theta_quadratic_tensor.flat<Dtype>()(0);
            Tensor cos_theta_cubic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_cubic_tensor));
            Dtype* cos_theta_cubic_data = &cos_theta_cubic_tensor.flat<Dtype>()(0);

            Tensor cos_theta_quartic_tensor;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value, TensorShape{M_ , N_}, &cos_theta_quartic_tensor));
            Dtype* cos_theta_quartic_data = &cos_theta_quartic_tensor.flat<Dtype>()(0);
        }
};
REGISTER_KERNEL_BUILDER(Name("MarginInnerProductGrad").Device(DEVICE_GPU), MarginInnerProductGradGpuOp);
