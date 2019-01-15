function [ res ] = artefacts( i1, i2 )
h = size(i1,1);
w = size(i1,2);

ci1 = RGB2Lab(i1);
ci2 = RGB2Lab(i2);

for j = 2:h-1
    for i = 2:w-1
        a = ci1(j,i,1);
        b = ci1(j,i,2);
        m = 99999;
        l = [0,0,0,0,0,0,0,0,0];
        for dy = -1:1
            for dx = -1:1
                if not (dx == 0 && dy == 0)
                  a1 = ci2(j+dy,i+dx,1); 
                  b1 = ci1(j+dy,i+dx,2);
                  m = min(m,dist2(a,b,a1,b1));
                end
            end
        end
    end
end

% diff = imsubtract(i1,i2);
% redchannel = diff(:,:,1);
% bluechannel = diff(:,:,3);
% 
% a = imsubtract(redchannel,bluechannel);
% 
% res = sum(sum(a)) / (h*w*255) * 100; 

end

