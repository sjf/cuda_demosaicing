function [ mse, psnr ] = mse( i1, i2 )
h = size(i1,1);
w = size(i1,2);
N = w*h;
L = 255;
diff = double(rgb2gray(i1)) - double(rgb2gray(i2));
sqs = diff.^2;
mse = sum(sum(sqs)) / N;
psnr = 10 * log10(L^2/mse) ;

end

