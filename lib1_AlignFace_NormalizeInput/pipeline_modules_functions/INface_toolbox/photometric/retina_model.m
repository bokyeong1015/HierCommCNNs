% The function applies the retina model to an image
% 
% PROTOTYPE
% Y = retina_model(X,sigma1, sigma2, Dogsigma1, Dogsigma2, treshold, normalize);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = retina_model(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = retina_model(X,1);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = retina_model(X,1,3);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y = retina_model(X,1,3,0.5);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 5:
%       X=imread('sample_image.bmp');
%       Y = retina_model(X,1,3,0.5,4);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 6:
%       X=imread('sample_image.bmp');
%       Y = retina_model(X,1,3,0.5,4,5,1);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using a 
% photometric normalization technique that is based on retina modeling. 
% 
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. The default parameters are set as described in the
% corresponding reference.
%
%
% 
% REFERENCES
% This function is an implementation of the retina modeling based 
% photometric normalization technique presented in:
%
% N.S Vu, A. Caplier: Illumination-robust face recognition using retina
% modeling. Proceedings of ICIP 2009, pp. 3289-3292.
% 
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% sigma1                - a scalar value defining the standard deviation of
%                         the first gaussian filter (see the reference for 
%                         a detailed description), default = 1;
% sigma2                - a scalar value defining the standard deviation of
%                         the second gaussian filter (see the reference for 
%                         a detailed description), default = 3;
% Dogsigma1             - a scalar value defining the standard deviation of
%                         the highpass part of the DoG filter (see the 
%                         reference for a detailed description), 
%                         default = 0.5;
% Dogsigma2             - a scalar value defining the standard deviation of
%                         the lowpass part of the DoG filter (see the 
%                         reference for a detailed description);
%                         default = 4;
% treshold              - a scalar value defining the threshold needed by
%                         the technique (see the reference for 
%                         a detailed description), default = 5;
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (normalization 
%                                to the 8-bit interval) - default
% 
% For a more detailed description of the parameters please type 
% "help retina_model" into Matlabs command prompt. 
% 
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the retina 
%                         modeling-based normalization technique 
%                         
%
% NOTES / COMMENTS
% This function applies a retina modeling based normalization technique to 
% the grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate          - a function provided by Peter Kovesi
% normalize8            - auxilary function
% perform_nl_means_adap - a function based on the code of Gabriel Peyre
% 
% 
% ABOUT
% Created:        26.8.2009
% Last Update:    23.1.2012
% Revision:       2.1
% 
%
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
% OR ANY PART OF IT, PLEASE MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
%
% 1. Štruc V., Pavešic, N.:Photometric normalization techniques for illumination 
% invariance, in: Y.J. Zhang (Ed), Advances in Face Image Analysis: Techniques 
% and Technologies, IGI Global, pp. 279-300, 2011.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/IGI.bib)
% 
% 2. Štruc, V., Pavešic, N.: Gabor-based kernel-partial-least-squares 
% discrimination features for face recognition, Informatica (Vilnius), 
% vol. 20, no. 1, pp. 115-138, 2009.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/InforVI.bib)
% 
%
% Official website:
% If you have down-loaded the toolbox from any other location than the
% official website, plese check the following link to make sure that you
% have the most recent version:
% 
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/INFace/index.html
% 
% 
% Copyright (c) 2011 Vitomir Štruc
% Faculty of Electrical Engineering,
% University of Ljubljana, Slovenia
% http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files, to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
% 
% January 2012
function Y = retina_model(X,sigma1, sigma2, Dogsigma1, Dogsigma2, treshold, normalize);

%% Default return
Y=[];

%% Parameter checking

if nargin == 1
    sigma1 = 1;
    sigma2 = 3;
    Dogsigma1 = 0.5;
    Dogsigma2 = 4;
    treshold = 5;
    normalize = 1;
elseif nargin == 2
    if isempty(sigma1)
        sigma1 = 1; 
    end
    sigma2 = 3;
    Dogsigma1 = 0.5;
    Dogsigma2 = 4;
    treshold = 5;
    normalize = 1;
elseif nargin == 3
     if isempty(sigma1)
        sigma1 = 1; 
     end
    if isempty(sigma2)
        sigma2 =3; 
    end
    Dogsigma1 = 0.5;
    Dogsigma2 = 4;
    treshold = 5;
    normalize = 1;
elseif nargin == 4
     if isempty(sigma1)
        sigma1 = 1; 
     end
    if isempty(sigma2)
        sigma2 =3; 
    end
    if isempty(Dogsigma1)
        Dogsigma1 =0.5; 
    end
    Dogsigma2 = 4;
    treshold = 5;
    normalize = 1;    
elseif nargin == 5
     if isempty(sigma1)
        sigma1 = 1; 
     end
    if isempty(sigma2)
        sigma2 =3; 
    end
    if isempty(Dogsigma1)
        Dogsigma1 =0.5; 
    end
    if isempty(Dogsigma2)
        Dogsigma2 =4; 
    end
    treshold = 5;
    normalize = 1;        
elseif nargin == 6
     if isempty(sigma1)
        sigma1 = 1; 
     end
    if isempty(sigma2)
        sigma2 =3; 
    end
    if isempty(Dogsigma1)
        Dogsigma1 =0.5; 
    end
    if isempty(Dogsigma2)
        Dogsigma2 =4; 
    end
    if isempty(treshold)
        treshold =5; 
    end
    normalize = 1;         
elseif nargin == 7
    if isempty(sigma1)
        sigma1 = 1; 
     end
    if isempty(sigma2)
        sigma2 =3; 
    end
    if isempty(Dogsigma1)
        Dogsigma1 =0.5; 
    end
    if isempty(Dogsigma2)
        Dogsigma2 =4; 
    end
    if isempty(treshold)
        treshold =5; 
    end
    
    if ~(normalize==1 || normalize==0)
         disp('Error: The seventh parameter can only be 0 or 1.');
         return;
     end
elseif nargin >7
    disp('Error! Wrong number of input parameters.')
    return;
end


%% Init. operations
[a,b]=size(X);
X=normalize8(X);


%% First non-linearity
Filter = fspecial('gaussian',2*ceil(3*sigma1)+1,sigma1);
F1 = double(imfilter(uint8(X),Filter,'replicate','same')+mean(X(:))/2);

Ila1 = (max(max(X))+F1).*(X./(X+F1));
Ila1 = normalize8(Ila1);

%% Second non-linearity
Filter = fspecial('gaussian',2*ceil(3*sigma2)+1,sigma2);
F2 = double(imfilter(uint8(Ila1),Filter,'replicate','same')+mean(Ila1(:))/2);

Ila2 = (max(max(Ila1))+F2).*(Ila1./(Ila1+F2));

%% Dog filtering
Ibip = dog(Ila2,Dogsigma1, Dogsigma2, 0);

%% Rescaling 
% Inor = (Ibip - mean(Ibip(:)))/std(Ibip(:));
Inor = (Ibip)/sqrt(mean(Ibip(:).^2));

%% Truncation
thresh_img = treshold*ones(a,b);
Ipp = reshape((min([thresh_img(:),Inor(:)]')').*(Inor(:)>=0)+(-min(abs([thresh_img(:),Inor(:)]'))').*(Inor(:)<0),[a,b]);
Y=Ipp;

%% Normalization to 8bits
if normalize ~= 0
    Y=normalize8(Y);  
end





