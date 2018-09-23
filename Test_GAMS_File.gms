Integer variables x1, x2, x3, x4, x5, x6;
free variable z total profit;

Equations        objfun  defines the objective function
                 maxengavl enforces the maximum availability of engineers
                 maxlaidoff enforces the maximum number of laid off engineers
;

objfun..     z =e= 0.5*power(x1,2)+0.1*power(x2,2)+0.5*power(x3,2)+0.1*power(x4,2)+0.5*power(x5,2)+0.1*power(x6,2);
maxengavl..  48 -x1 +0.2*x2 -x3 +0.2*x4 -x5 +0.2*x6 =l= 0;
maxlaidoff.. 250 -5*x1 +x2 -5*x3 +x4 -5*x5 +x6 =l= 0 ;


Model ProjectSelection /all/;
options MINLP = BARON;
options optcr = 0.0;
solve ProjectSelection using MINLP minimizing z;
display x1.l,x2.l,x3.l,x4.l,x5.l,x6.l,z.l;