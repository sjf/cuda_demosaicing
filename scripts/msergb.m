function [ rm, gm, bm ] = msergb( i1, i2 )
h = size(i1,1);
w = size(i1,2);
N = w*h;

rd  = double(i1(:,:,1)) - double(i2(:,:,1));
rm = (sum(sum(rd.^2))) / N;

gd  = double(i1(:,:,2)) - double(i2(:,:,2));
gm = (sum(sum(gd.^2))) / N;

bd  = double(i1(:,:,3)) - double(i2(:,:,3));
bm = (sum(sum(bd.^2))) / N;
end

