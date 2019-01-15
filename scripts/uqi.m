%F='lh2';

function  uqi(base_name)
fprintf(' ++ Starting QI... %s\n',base_name);
fprintf('Date, Image, MSE, PSNR, SSIM, ART\n');


out = fopen('/tmp/uqi.m.out','a');

ref  = imread([ base_name, '.orig.ppm']);
%bil  = imread([ base_name, '.bilin.ppm']);
%ahd  = imread([ base_name, '.ahd.ppm']);
%mask = imread([ base_name, '.ahdmask.ppm']);
mask2= imread([ base_name, '.ahdmask2.ppm']);

d = datestr(now);

% [qi qi_map] = ssim_index(rgb2gray(ref), rgb2gray(bil));
% [ms, psnr] = mse(ref,bil);
% a = artefacts(ref,bil);

% fprintf( '%s, %s, bil, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);
% fprintf(out, '%s, %s, bil, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);

%imshow(max(0, qi_map).^4);

% [qi qi_map] = ssim_index(rgb2gray(ref), rgb2gray(ahd));[
% ms, psnr] = mse(ref,ahd);
% a = artefacts(ref,ahd);

% [qi qi_map] = ssim_index(rgb2gray(ref), rgb2gray(mask));
% [ms, psnr] = mse(ref,mask);
% a = artefacts(ref,mask);

% fprintf('%s, %s, Mask, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);
% fprintf(out,'%s, %s, Mask, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);


% fprintf('%s, %s, AHD, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);
% fprintf(out,'%s, %s, AHD, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);

[qi qi_map] = ssim_index(rgb2gray(ref), rgb2gray(mask2));
%[ms, psnr] = mse(ref,mask2);
ms = 0;
psnr = 0;
%a = artefacts(ref,mask2);
a = 0;

fprintf('%s, %s, Mask2, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);
fprintf(out,'%s, %s, Mask2, %f, %f, %f, %f\n',d,base_name,ms,psnr,a,qi);


fprintf(' ++ end\n');
