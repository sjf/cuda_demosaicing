%F='lh2';

function  uqi(base_name)
fprintf(' ++ Starting QI... %s\n',base_name);
fprintf('Date, Image, MSE R, MSE G, MSE B, \n');


out = fopen('/tmp/uqi.m.out','a');

ref  = imread([ base_name, '.orig.ppm']);
bil  = imread([ base_name, '.bilin.ppm']);
ahd  = imread([ base_name, '.ahd.ppm']);
mask = imread([ base_name, '.ahdmask.ppm']);
mask2 = imread([ base_name, '.ahdmask2.ppm']);

d = datestr(now);

[r,g,b ] = msergb(ref,bil);

fprintf( '%s, %s, bil, %f, %f, %f\n',d,base_name,r,g,b);
fprintf(out, '%s, %s, bil, %f, %f, %f\n',d,base_name,r,g,b);

[r,g,b] = msergb(ref,ahd);

fprintf('%s, %s, AHD, %f, %f, %f\n',d,base_name,r,g,b);
fprintf(out,'%s, %s, AHD, %f, %f, %f\n',d,base_name,r,g,b);

[r,g,b] = msergb(ref,mask);

fprintf('%s, %s, Mask, %f, %f, %f\n',d,base_name,r,g,b);
fprintf(out,'%s, %s, Mask, %f, %f, %f\n',d,base_name,r,g,b);

[r,g,b] = msergb(ref,mask2);

fprintf('%s, %s, Mask2, %f, %f, %f\n',d,base_name,r,g,b);
fprintf(out,'%s, %s, Mask2, %f, %f, %f\n',d,base_name,r,g,b);

fprintf(' ++ end\n');
