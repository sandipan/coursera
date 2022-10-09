function [s1, s2, sums] = sines(pts,amp,f1,f2)
    if ~exist('pts','var')
      pts = 1000;
    end
    if ~exist('f1','var')
      f1 = 100;
    end
    if ~exist('amp','var')
      amp = 1;
    end
    if ~exist('f2','var')
      f2 = f1 * 1.05;
    end
    s1 = zeros(1, pts);
    s2 = zeros(1, pts);
    x1 = zeros(1, pts);
    x2 = zeros(1, pts);
    for i = 1:(pts-1)
        s1(i) = amp * sin(x1(i));
        x1(i + 1) = x1(i) + 2 * pi * f1 / (pts - 1);
        s2(i) = amp * sin(x2(i));
        x2(i + 1) = x2(i) + 2 * pi * f2 / (pts - 1);
    end
    sums = s1 + s2;
    %plot(x1, sums);
end