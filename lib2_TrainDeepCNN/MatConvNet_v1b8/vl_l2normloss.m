function Y = vl_l2normloss(X,C,dzdy)

    assert(numel(X) == numel(C));
    n = size(X,1) * size(X,2);
    if nargin <= 2
      Y = sum((X(:) - C(:)).^2) ./ (2*n);
    else
      assert(numel(dzdy) == 1);
      Y = reshape((dzdy / n) * (X(:) - C(:)), size(X));
    end

end