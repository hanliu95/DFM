function s = ScaleScore(y, scale)
maxy = max(y);
miny = min(y);
s = (y-miny)/(maxy-miny);
s = 2*scale*s-scale;
end

