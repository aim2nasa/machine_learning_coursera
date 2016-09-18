function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%








X=[ones(m,1) X];

delta_cap2 = 0;
delta_cap1 = 0;

for i=1:m
  %forward propagation
  a1 = X(i,:)';
  
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  
  a2 = [1; a2]; %add 1
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  h = a3;
 
  %cost를 구하기 위한 작업들
  yy=zeros(num_labels,1);
  if y(i)==10
      yy(10)=1;
  else
      yy(y(i))=1;
  end 
  
  %backpropagation
  delta_3 = h-yy;
  delta_2 = Theta2(:,2:end)'*delta_3.*sigmoidGradient(z2);
  
  delta_cap2 = delta_cap2 + delta_3 * a2';
  delta_cap1 = delta_cap1 + delta_2 * a1';

% cost계산
%   for k=1:K
%       J = J + (-yy(k)*log(h(k))-(1-yy(k))*log(1-h(k)));
%   end
% 위의 for loop식을 아래의 벡터곱으로 변경
  J = J + sum(-yy.*log(h)-(1-yy).*log(1-h));
end
J = J/m;

%Regularizing
t1sq=Theta1.^2;
t1sqr = sum(t1sq);   %theta1제곱하고 나온 25x401행렬을 칼럼별로 합한 결과 (1x401)

t2sq=Theta2.^2;
t2sqr = sum(t2sq);   %theta2제곱하고 나온 10x26행렬을 칼럼별로 합한 결과 (1x26)

%bias를 빼기 위해서 sum은 2부터 시작함
reg=(lambda/(2*m))*(sum(t1sqr(2:end))+sum(t2sqr(2:end)));
J = J + reg;

%regularizing term추가 (일단은 모든 ij에 적용하고 다음단계에서 뺀다)
Theta1_grad = (1/m)*delta_cap1 + (lambda/m)*Theta1;
Theta2_grad = (1/m)*delta_cap2 + (lambda/m)*Theta2;

%regularization term이 bias칼럼에도 적용되었으므로 이를 뺀다
Theta1_grad(:,1) = Theta1_grad(:,1) - (lambda/m)*Theta1(:,1);
Theta2_grad(:,1) = Theta2_grad(:,1) - (lambda/m)*Theta2(:,1);










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
