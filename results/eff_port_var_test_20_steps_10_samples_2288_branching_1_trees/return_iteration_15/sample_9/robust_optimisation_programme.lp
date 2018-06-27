\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.14006 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.9696 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.27474 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.96199 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.888278 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.24174 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.21326 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.80561 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.30105 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.03393 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.985948 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.09449 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.27095 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.972545 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.21279 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.05004 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.31577 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.11932 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.53064 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.22878 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.858357 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.04548 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.821131 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.837057 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.883043 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.27324 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.1898 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.81621 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.28972 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.04858 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.852551 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.23873 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.933173 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.927664 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.08191 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.03745 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.709148 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.800501 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.33807 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.950206 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.35381 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.986908 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.808706 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.27573 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.02355 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.835169 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.74256 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.803191 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.00909 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.02819 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.45477 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.691496 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.00907 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.1586 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.36981 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.798252 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.72669 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.33831 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.626733 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.908707 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.588985 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.836069 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.974844 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.50461 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.10547 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.03933 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.4116 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.10156 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.00569 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.52669 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.06375 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.913972 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.17499 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.45788 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.773614 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.735636 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38858 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.920464 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.30636 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.00215 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.855129 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.37257 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.20477 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.76634 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 2.07516 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.999321 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.37295 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.48943 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.13591 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.649204 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.18162 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.11523 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.30489 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.815718 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.95075 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.75835 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.558577 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.916134 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.562571 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.712001 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.86666 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.77218 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.20729 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.04805 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.3078 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.00192 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.974735 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.76153 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.062 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.85051 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.18444 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.36143 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.718515 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.658018 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24414 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.953777 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.2487 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.93348 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.874256 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.2279 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.26189 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.852949 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.51876 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.06745 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.02976 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.19797 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.14305 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.859495 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.2994 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.13753 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.13123 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.00081 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.51946 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.40095 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.740954 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.896267 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.715222 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.741052 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.937272 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.42095 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.15656 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.889866 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.24035 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.14604 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.955406 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.48076 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.928701 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.988565 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.16806 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11537 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.748299 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.757315 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17236 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.973614 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.902678 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.0801 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.960462 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.195 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.0209 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.19462 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.92969 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.844169 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.83325 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.36432 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.50016 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.574265 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.683456 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.05193 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.83079 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.696822 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.70659 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.26782 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.649592 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.19286 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.506286 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.883159 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.781528 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.5695 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.80656 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.58485 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.04627 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.580338 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.09053 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.13021 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.48526 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.65024 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.862984 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.92691 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.09307 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.610232 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1729 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.979473 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.22446 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.967955 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.851038 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.14692 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.961691 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.885927 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.32608 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.01259 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.16928 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.1109 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.993102 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.759177 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.17252 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.08192 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.1362 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.890096 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.22537 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.4848 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.776422 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.884113 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.792776 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.887309 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.949542 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.34594 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.15058 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.04237 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.16201 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.996436 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.12549 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.36904 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.02531 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.926276 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.02209 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.24428 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.792777 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.782037 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.23317 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.917203 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.00863 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.06287 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.904794 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.24647 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.20921 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.936422 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.56076 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.02508 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.57748 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.6405 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.26927 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.729431 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.931835 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.980136 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.29951 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.805784 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.64426 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.32372 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.716102 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.00316 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.619881 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.858914 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.830459 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.62557 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.05231 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.33054 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.28232 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.827927 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.08475 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.32442 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.203 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.765275 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.03169 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.41907 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.81827 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.706049 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2862 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.01305 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.933312 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.04959 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.869283 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.22077 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.01323 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.01268 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.9394 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.88347 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.51902 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.34296 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.85391 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.438978 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.687224 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.961291 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.66105 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.736093 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.35886 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.15343 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.54983 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.07138 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.357123 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.781609 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.860085 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.63562 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.736333 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.68164 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.15344 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.754033 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.13987 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.27006 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.1175 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.571749 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.99563 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 2.04094 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.971169 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.594481 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22387 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.943506 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.18627 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.982175 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.907334 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.34162 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.07319 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.863931 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.52937 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.979605 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.09178 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.21881 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.95656 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.825747 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.12733 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.02257 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.961147 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.885194 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.37366 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.36837 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.730238 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.857063 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.754941 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.845134 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.950632 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.40767 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.11497 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.969583 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.12012 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.02561 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.918822 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.2563 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.884743 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.86086 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.92963 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.03797 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.706956 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.691316 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16869 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.977102 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.11256 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.04265 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.902766 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.1072 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.07492 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.947501 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.21195 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.981392 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05593 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.15098 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05481 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.822486 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.994676 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.923187 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.10094 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.863497 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.5222 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.07141 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.742816 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.887683 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.756126 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.887517 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.925327 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.34217 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.08739 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.08037 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.15933 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.981197 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.03847 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.2097 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.04889 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.87241 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.03195 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15224 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.905691 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.791081 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.066000023782053 z_k1_s0000
                                 - 0.0869997068492234 z_k1_s0001
                                 - 0.0529999758310631 z_k1_s0002
                                 - 0.0499999747500607 z_k1_s0003
                                 - 0.076999919579113 z_k1_s0004
                                 - 0.16300016690108 z_k1_s0005
                                 - 0.0980002516459764 z_k1_s0006
                                 - 0.0549999981850544 z_k1_s0007
                                 - 0.144000156223157 z_k1_s0010
                                 - 0.131000031619502 z_k1_s0011
                                 - 0.0940001665791077 z_k1_s0012
                                 - 0.316000580823416 z_k1_s0013
                                 - 0.104000140027879 z_k1_s0014
                                 - 0.104000140027879 z_k1_s0015
                                 - 0.0890001186546251 z_k1_s0016
                                 - 0.241999871541075 z_k1_s0017
                                 - 0.0290000350353075 z_k1_s0020
                                 - 0.0460001319733693 z_k1_s0021
                                 - 0.104000049243739 z_k1_s0022
                                 - 0.0420002035410359 z_k1_s0023
                                 - 0.0210000253703951 z_k1_s0024
                                 - 0.0210000253703951 z_k1_s0025
                                 - 0.0690000069597467 z_k1_s0026
                                 - 0.0500000604057026 z_k1_s0027
                                 - 0.0899999592537942 z_k1_s0030
                                 - 0.0560000124623395 z_k1_s0031
                                 - 0.148999766821375 z_k1_s0032
                                 - 0.100999841939321 z_k1_s0033
                                 - 0.172999729262402 z_k1_s0034
                                 - 0.266999582156424 z_k1_s0035
                                 - 0.110999826289749 z_k1_s0036
                                 - 0.0540000155922539 z_k1_s0037
                                 - 0.0160000115407858 z_k1_s0040
                                 - 0.0660001200558533 z_k1_s0041
                                 - 0.0180000210333964 z_k1_s0042
                                 - 0.0690001986948689 z_k1_s0043
                                 - 0.0440000800372356 z_k1_s0044
                                 - 0.0300000457890107 z_k1_s0045
                                 - 0.0300000457890107 z_k1_s0046
                                 - 0.0490001037687622 z_k1_s0047
                                 - 0.0420002042209978 z_k1_s0050
                                 - 0.0290000105905702 z_k1_s0051
                                 - 0.056000160961229 z_k1_s0052
                                 - 0.0530001702311795 z_k1_s0053
                                 - 0.0260000198605207 z_k1_s0054
                                 - 0.0670001269714108 z_k1_s0055
                                 - 0.0320000347206502 z_k1_s0056
                                 - 0.0290000105905702 z_k1_s0057
                                 - 0.0350000520608874 z_k1_s0060
                                 - 0.0310000225681917 z_k1_s0061
                                 - 0.0449999197924264 z_k1_s0062
                                 - 0.0590001054169417 z_k1_s0063
                                 - 0.0260000269023621 z_k1_s0064
                                 - 0.054000068551072 z_k1_s0065
                                 - 0.0499999566582961 z_k1_s0066
                                 - 0.11200024899492 z_k1_s0067
                                 - 0.0550000557526236 z_k1_s0070
                                 - 0.184000303256239 z_k1_s0071
                                 - 0.195000580087194 z_k1_s0072
                                 - 0.0650001564623383 z_k1_s0073
                                 - 0.134999975828906 z_k1_s0074
                                 - 0.154000554627992 z_k1_s0075
                                 - 0.195000580087194 z_k1_s0076
                                 - 0.123999698997951 z_k1_s0077
                                 - 0.350000585680985 z_k1_s0100
                                 - 0.126000052445179 z_k1_s0101
                                 - 0.15999981516852 z_k1_s0102
                                 - 0.354000185060259 z_k1_s0103
                                 - 0.153999940899682 z_k1_s0104
                                 - 0.0779999494946476 z_k1_s0105
                                 - 0.162000406858035 z_k1_s0106
                                 - 0.200000560960528 z_k1_s0107
                                 - 0.409999401864222 z_k1_s0110
                                 - 0.291000191615997 z_k1_s0111
                                 - 0.252000110784972 z_k1_s0112
                                 - 0.533001600024148 z_k1_s0113
                                 - 0.267000096150738 z_k1_s0114
                                 - 0.266000017872998 z_k1_s0115
                                 - 0.215999967587083 z_k1_s0116
                                 - 0.736999736678107 z_k1_s0117
                                 - 0.264000492691971 z_k1_s0120
                                 - 0.0890002622971903 z_k1_s0121
                                 - 0.329001095000085 z_k1_s0122
                                 - 0.410001727168319 z_k1_s0123
                                 - 0.256000477761912 z_k1_s0124
                                 - 0.376000701712807 z_k1_s0125
                                 - 0.0940001754282019 z_k1_s0126
                                 - 0.106000390223843 z_k1_s0127
                                 - 0.836000509890152 z_k1_s0130
                                 - 0.473000088384849 z_k1_s0131
                                 - 0.341000488129394 z_k1_s0132
                                 - 0.655002164063122 z_k1_s0133
                                 - 0.457001242877137 z_k1_s0134
                                 - 0.253000374425057 z_k1_s0135
                                 - 0.258000393846919 z_k1_s0136
                                 - 0.528999849668853 z_k1_s0137
                                 - 0.185999451259809 z_k1_s0140
                                 - 0.0759998826737781 z_k1_s0141
                                 - 0.0699999327363279 z_k1_s0142
                                 - 0.146000089009793 z_k1_s0143
                                 - 0.0439999368742871 z_k1_s0144
                                 - 0.207000051239998 z_k1_s0145
                                 - 0.118999447892258 z_k1_s0146
                                 - 0.063999891598982 z_k1_s0147
                                 - 0.0939999383457374 z_k1_s0150
                                 - 0.105999887649481 z_k1_s0151
                                 - 0.220000057234552 z_k1_s0152
                                 - 0.168999295894382 z_k1_s0153
                                 - 0.0979999609136221 z_k1_s0154
                                 - 0.106999893291452 z_k1_s0155
                                 - 0.275999662785475 z_k1_s0156
                                 - 0.11399993278525 z_k1_s0157
                                 - 0.244999865548713 z_k1_s0160
                                 - 0.288999769364075 z_k1_s0161
                                 - 0.218999894721486 z_k1_s0162
                                 - 0.508998984141231 z_k1_s0163
                                 - 0.396000351536861 z_k1_s0164
                                 - 0.407000783940178 z_k1_s0165
                                 - 0.792998805930786 z_k1_s0166
                                 - 0.184999886055165 z_k1_s0167
                                 - 0.494001088198334 z_k1_s0170
                                 - 0.208999850007323 z_k1_s0171
                                 - 0.166999918115879 z_k1_s0172
                                 - 0.332000217331957 z_k1_s0173
                                 - 0.55399917718711 z_k1_s0174
                                 - 0.307999712137145 z_k1_s0175
                                 - 0.377001164572229 z_k1_s0176
                                 - 0.732999982700046 z_k1_s0177
                                 - 0.2850005139605 z_k1_s1000
                                 - 0.166000007629686 z_k1_s1001
                                 - 0.259999828027258 z_k1_s1002
                                 - 0.315000137640644 z_k1_s1003
                                 - 0.276999705645987 z_k1_s1004
                                 - 0.644001008331829 z_k1_s1005
                                 - 0.222999837821844 z_k1_s1006
                                 - 0.556000294870458 z_k1_s1007
                                 - 0.505000436509568 z_k1_s1010
                                 - 0.799004980745689 z_k1_s1011
                                 - 1.7680064639626 z_k1_s1012
                                 - 0.636000634570984 z_k1_s1013
                                 - 0.689001084102472 z_k1_s1014
                                 - 0.849998378379038 z_k1_s1015
                                 - 0.831998375387428 z_k1_s1016
                                 - 1.85400030813583 z_k1_s1017
                                 - 1.02399658824289 z_k1_s1020
                                 - 2.6959922327306 z_k1_s1021
                                 - 0.840997835966068 z_k1_s1022
                                 - 0.776998049200887 z_k1_s1023
                                 - 1.50500121379266 z_k1_s1024
                                 - 1.22199440948848 z_k1_s1025
                                 - 1.05999707592199 z_k1_s1026
                                 - 0.59699852739908 z_k1_s1027
                                 - 0.447999116623757 z_k1_s1030
                                 - 0.359999828054589 z_k1_s1031
                                 - 0.436000126454867 z_k1_s1032
                                 - 0.312999934177984 z_k1_s1033
                                 - 0.838998846196969 z_k1_s1034
                                 - 0.37299957082055 z_k1_s1035
                                 - 0.308999936088488 z_k1_s1036
                                 - 0.268999955193534 z_k1_s1037
                                 - 0.908005458272995 z_k1_s1040
                                 - 0.502001463183628 z_k1_s1041
                                 - 0.28900100286218 z_k1_s1042
                                 - 0.675999999926519 z_k1_s1043
                                 - 0.545001660469497 z_k1_s1044
                                 - 1.28700512941423 z_k1_s1045
                                 - 0.987004011467091 z_k1_s1046
                                 - 0.363001019295015 z_k1_s1047
                                 - 1.64400382328239 z_k1_s1050
                                 - 0.508001248621194 z_k1_s1051
                                 - 0.655000806278319 z_k1_s1052
                                 - 0.593000919758863 z_k1_s1053
                                 - 1.4329991594866 z_k1_s1054
                                 - 0.54900119833249 z_k1_s1055
                                 - 1.12900080512209 z_k1_s1056
                                 - 1.16300435710344 z_k1_s1057
                                 - 2.27500090943452 z_k1_s1060
                                 - 1.35799699820103 z_k1_s1061
                                 - 0.685000409122156 z_k1_s1062
                                 - 1.45800244987301 z_k1_s1063
                                 - 0.751000313742572 z_k1_s1064
                                 - 0.917003911233496 z_k1_s1065
                                 - 0.735000496755939 z_k1_s1066
                                 - 0.615000110554714 z_k1_s1067
                                 - 0.229999993101508 z_k1_s1070
                                 - 0.432999827134572 z_k1_s1071
                                 - 0.19399988667692 z_k1_s1072
                                 - 0.773001360589037 z_k1_s1073
                                 - 0.756999199955803 z_k1_s1074
                                 - 0.261000058328236 z_k1_s1075
                                 - 0.264000146446955 z_k1_s1076
                                 - 0.257999970209517 z_k1_s1077
                                 - 0.422001156560259 z_k1_s1100
                                 - 1.16000421583618 z_k1_s1101
                                 - 0.480001590552563 z_k1_s1102
                                 - 0.667003428508029 z_k1_s1103
                                 - 0.327000776663254 z_k1_s1104
                                 - 0.767001713868617 z_k1_s1105
                                 - 0.241000733489788 z_k1_s1106
                                 - 0.400000730259151 z_k1_s1107
                                 - 0.105000048899259 z_k1_s1110
                                 - 0.156999939650335 z_k1_s1111
                                 - 0.147999970276109 z_k1_s1112
                                 - 0.225000249855541 z_k1_s1113
                                 - 0.448999216814078 z_k1_s1114
                                 - 0.175999942695916 z_k1_s1115
                                 - 0.347000376306102 z_k1_s1116
                                 - 0.423999640385638 z_k1_s1117
                                 - 0.498998597632427 z_k1_s1120
                                 - 0.208999868173802 z_k1_s1121
                                 - 0.222999928277295 z_k1_s1122
                                 - 0.620000374869482 z_k1_s1123
                                 - 0.429000965114372 z_k1_s1124
                                 - 0.228000025971382 z_k1_s1125
                                 - 0.267000040945432 z_k1_s1126
                                 - 0.19299987571265 z_k1_s1127
                                 - 0.991999618965245 z_k1_s1130
                                 - 0.561000176109264 z_k1_s1131
                                 - 0.304999856740935 z_k1_s1132
                                 - 0.851999150615463 z_k1_s1133
                                 - 0.295000121644435 z_k1_s1134
                                 - 0.2589998221974 z_k1_s1135
                                 - 0.575001476043878 z_k1_s1136
                                 - 0.337999817959141 z_k1_s1137
                                 - 1.22800191998931 z_k1_s1140
                                 - 0.846002339723857 z_k1_s1141
                                 - 0.28000020920083 z_k1_s1142
                                 - 0.459999833944244 z_k1_s1143
                                 - 1.163001213007 z_k1_s1144
                                 - 0.638999173334097 z_k1_s1145
                                 - 0.765997352438134 z_k1_s1146
                                 - 0.565000061069549 z_k1_s1147
                                 - 0.394999040220207 z_k1_s1150
                                 - 0.142999903813752 z_k1_s1151
                                 - 0.581000257019132 z_k1_s1152
                                 - 0.212999833593224 z_k1_s1153
                                 - 0.382999372200683 z_k1_s1154
                                 - 0.134000025549182 z_k1_s1155
                                 - 0.407998892657902 z_k1_s1156
                                 - 0.28799966746414 z_k1_s1157
                                 - 0.27800013811483 z_k1_s1160
                                 - 0.113999825077418 z_k1_s1161
                                 - 0.471000743777035 z_k1_s1162
                                 - 0.391999002392718 z_k1_s1163
                                 - 0.31999908758594 z_k1_s1164
                                 - 0.173999834149693 z_k1_s1165
                                 - 0.515000558270131 z_k1_s1166
                                 - 0.137999876746304 z_k1_s1167
                                 - 0.47100122746512 z_k1_s1170
                                 - 0.305000471886052 z_k1_s1171
                                 - 0.267000981113707 z_k1_s1172
                                 - 0.39200100364182 z_k1_s1173
                                 - 0.195000745030335 z_k1_s1174
                                 - 0.29099993300423 z_k1_s1175
                                 - 0.297000163953582 z_k1_s1176
                                 - 0.599003337374794 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.841934 w_k1_ADM_s000
                                 + 1.27137 w_k1_AXP_s000 + 1.1925 w_k1_DIS_s000
                                 + 1.04937 w_k1_FDX_s000
                                 + 1.38893 w_k1_IBM_s000
                                 + 1.40572 w_k1_INTC_s000
                                 + 1.25343 w_k1_KO_s000 + 1.81194 w_k1_MAT_s000
                                 + 1.11338 w_k1_MSFT_s000
                                 + 1.19579 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.959627 w_k1_ADM_s000
                                 + 0.961253 w_k1_AXP_s000
                                 + 0.729279 w_k1_DIS_s000
                                 + 0.885861 w_k1_FDX_s000
                                 + 0.603224 w_k1_IBM_s000
                                 + 1.08905 w_k1_INTC_s000
                                 + 0.693884 w_k1_KO_s000
                                 + 0.544075 w_k1_MAT_s000
                                 + 0.945892 w_k1_MSFT_s000
                                 + 0.762346 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.914561 w_k1_ADM_s000
                                 + 0.976252 w_k1_AXP_s000
                                 + 0.913493 w_k1_DIS_s000
                                 + 0.886074 w_k1_FDX_s000
                                 + 0.911161 w_k1_IBM_s000
                                 + 1.04847 w_k1_INTC_s000
                                 + 0.79739 w_k1_KO_s000
                                 + 0.903951 w_k1_MAT_s000
                                 + 0.970962 w_k1_MSFT_s000
                                 + 0.888013 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.939974 w_k1_ADM_s000
                                 + 0.982738 w_k1_AXP_s000
                                 + 0.945234 w_k1_DIS_s000
                                 + 0.80442 w_k1_FDX_s000
                                 + 1.28237 w_k1_IBM_s000
                                 + 1.29041 w_k1_INTC_s000
                                 + 0.904879 w_k1_KO_s000
                                 + 1.10504 w_k1_MAT_s000
                                 + 0.996531 w_k1_MSFT_s000
                                 + 0.976499 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.866767 w_k1_ADM_s000
                                 + 0.85737 w_k1_AXP_s000
                                 + 1.09895 w_k1_DIS_s000
                                 + 1.04426 w_k1_FDX_s000
                                 + 0.926794 w_k1_IBM_s000
                                 + 1.09024 w_k1_INTC_s000
                                 + 1.04807 w_k1_KO_s000 + 1.16593 w_k1_MAT_s000
                                 + 1.19474 w_k1_MSFT_s000
                                 + 0.981371 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.34886 w_k1_ADM_s000
                                 + 1.26037 w_k1_AXP_s000
                                 + 1.38841 w_k1_DIS_s000
                                 + 1.22269 w_k1_FDX_s000
                                 + 1.02644 w_k1_IBM_s000
                                 + 1.18731 w_k1_INTC_s000
                                 + 1.07507 w_k1_KO_s000 + 1.0322 w_k1_MAT_s000
                                 + 1.22501 w_k1_MSFT_s000
                                 + 1.15398 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.91194 w_k1_ADM_s000
                                 + 0.880642 w_k1_AXP_s000
                                 + 0.851809 w_k1_DIS_s000
                                 + 0.783847 w_k1_FDX_s000
                                 + 1.03957 w_k1_IBM_s000
                                 + 1.12429 w_k1_INTC_s000
                                 + 0.937691 w_k1_KO_s000
                                 + 1.32932 w_k1_MAT_s000
                                 + 1.07709 w_k1_MSFT_s000
                                 + 0.948911 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.01935 w_k1_ADM_s000
                                 + 1.43342 w_k1_AXP_s000
                                 + 0.848331 w_k1_DIS_s000
                                 + 1.04637 w_k1_FDX_s000
                                 + 0.751654 w_k1_IBM_s000
                                 + 0.983603 w_k1_INTC_s000
                                 + 0.70458 w_k1_KO_s000 + 1.11848 w_k1_MAT_s000
                                 + 0.947638 w_k1_MSFT_s000
                                 + 0.982653 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.31617 w_k1_ADM_s001
                                 + 0.973362 w_k1_AXP_s001
                                 + 1.22551 w_k1_DIS_s001
                                 + 1.13596 w_k1_FDX_s001
                                 + 1.12103 w_k1_IBM_s001
                                 + 0.943794 w_k1_INTC_s001
                                 + 1.24241 w_k1_KO_s001 + 1.27382 w_k1_MAT_s001
                                 + 0.956803 w_k1_MSFT_s001
                                 + 1.09412 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.82027 w_k1_ADM_s001
                                 + 1.60648 w_k1_AXP_s001
                                 + 1.44366 w_k1_DIS_s001
                                 + 1.34798 w_k1_FDX_s001
                                 + 1.11943 w_k1_IBM_s001
                                 + 1.05161 w_k1_INTC_s001
                                 + 1.21887 w_k1_KO_s001 + 1.51946 w_k1_MAT_s001
                                 + 1.23222 w_k1_MSFT_s001
                                 + 1.19168 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.32978 w_k1_ADM_s001
                                 + 1.34304 w_k1_AXP_s001
                                 + 1.67266 w_k1_DIS_s001
                                 + 1.29435 w_k1_FDX_s001 + 1.1397 w_k1_IBM_s001
                                 + 1.35818 w_k1_INTC_s001
                                 + 1.37841 w_k1_KO_s001 + 1.04691 w_k1_MAT_s001
                                 + 1.21764 w_k1_MSFT_s001
                                 + 1.07989 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.908144 w_k1_ADM_s001
                                 + 0.971193 w_k1_AXP_s001
                                 + 0.981253 w_k1_DIS_s001
                                 + 1.01519 w_k1_FDX_s001
                                 + 0.711079 w_k1_IBM_s001
                                 + 0.832752 w_k1_INTC_s001
                                 + 0.848131 w_k1_KO_s001
                                 + 0.881561 w_k1_MAT_s001
                                 + 1.00263 w_k1_MSFT_s001
                                 + 0.799399 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.918872 w_k1_ADM_s001
                                 + 1.37462 w_k1_AXP_s001
                                 + 0.994958 w_k1_DIS_s001
                                 + 1.01337 w_k1_FDX_s001 + 1.0161 w_k1_IBM_s001
                                 + 1.03376 w_k1_INTC_s001
                                 + 0.920024 w_k1_KO_s001
                                 + 1.42085 w_k1_MAT_s001
                                 + 0.924602 w_k1_MSFT_s001
                                 + 0.997497 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.05351 w_k1_ADM_s001
                                 + 0.868287 w_k1_AXP_s001
                                 + 0.934605 w_k1_DIS_s001
                                 + 0.885365 w_k1_FDX_s001
                                 + 1.02882 w_k1_IBM_s001
                                 + 1.03259 w_k1_INTC_s001
                                 + 1.05634 w_k1_KO_s001
                                 + 0.958955 w_k1_MAT_s001
                                 + 0.991907 w_k1_MSFT_s001
                                 + 0.832814 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.0158 w_k1_ADM_s001
                                 + 1.20863 w_k1_AXP_s001
                                 + 1.06089 w_k1_DIS_s001 + 1.0105 w_k1_FDX_s001
                                 + 1.51068 w_k1_IBM_s001
                                 + 0.951545 w_k1_INTC_s001
                                 + 1.2825 w_k1_KO_s001 + 1.78226 w_k1_MAT_s001
                                 + 0.991424 w_k1_MSFT_s001
                                 + 1.1467 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.858619 w_k1_ADM_s001
                                 + 0.632681 w_k1_AXP_s001
                                 + 0.613683 w_k1_DIS_s001
                                 + 0.720943 w_k1_FDX_s001
                                 + 0.717302 w_k1_IBM_s001
                                 + 0.592531 w_k1_INTC_s001
                                 + 0.761193 w_k1_KO_s001
                                 + 0.69767 w_k1_MAT_s001
                                 + 0.76801 w_k1_MSFT_s001
                                 + 0.740361 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.30451 w_k1_ADM_s002
                                 + 1.16108 w_k1_AXP_s002
                                 + 1.50448 w_k1_DIS_s002
                                 + 1.19537 w_k1_FDX_s002
                                 + 1.76141 w_k1_IBM_s002
                                 + 0.96293 w_k1_INTC_s002
                                 + 1.73508 w_k1_KO_s002 + 2.00616 w_k1_MAT_s002
                                 + 1.00121 w_k1_MSFT_s002
                                 + 1.16077 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.09068 w_k1_ADM_s002
                                 + 1.05444 w_k1_AXP_s002
                                 + 1.17991 w_k1_DIS_s002
                                 + 1.05355 w_k1_FDX_s002
                                 + 1.23647 w_k1_IBM_s002
                                 + 1.03977 w_k1_INTC_s002 + 1.3663 w_k1_KO_s002
                                 + 1.10986 w_k1_MAT_s002
                                 + 0.852095 w_k1_MSFT_s002
                                 + 1.17409 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.33349 w_k1_ADM_s002
                                 + 1.09209 w_k1_AXP_s002
                                 + 1.34437 w_k1_DIS_s002
                                 + 1.26874 w_k1_FDX_s002
                                 + 0.75719 w_k1_IBM_s002
                                 + 1.02497 w_k1_INTC_s002
                                 + 1.08872 w_k1_KO_s002 + 0.67381 w_k1_MAT_s002
                                 + 1.0168 w_k1_MSFT_s002 + 1.17534 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.897407 w_k1_ADM_s002
                                 + 0.590863 w_k1_AXP_s002
                                 + 0.650583 w_k1_DIS_s002
                                 + 0.764131 w_k1_FDX_s002
                                 + 0.659128 w_k1_IBM_s002
                                 + 0.899813 w_k1_INTC_s002
                                 + 0.825272 w_k1_KO_s002
                                 + 0.655307 w_k1_MAT_s002
                                 + 0.737944 w_k1_MSFT_s002
                                 + 0.948385 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.805779 w_k1_ADM_s002
                                 + 0.597734 w_k1_AXP_s002
                                 + 0.764994 w_k1_DIS_s002
                                 + 0.965696 w_k1_FDX_s002
                                 + 0.489414 w_k1_IBM_s002
                                 + 1.20656 w_k1_INTC_s002
                                 + 0.811171 w_k1_KO_s002
                                 + 0.541156 w_k1_MAT_s002
                                 + 1.01831 w_k1_MSFT_s002
                                 + 0.975301 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.13719 w_k1_ADM_s002
                                 + 0.773605 w_k1_AXP_s002
                                 + 1.00453 w_k1_DIS_s002
                                 + 0.970149 w_k1_FDX_s002
                                 + 0.842426 w_k1_IBM_s002
                                 + 0.809064 w_k1_INTC_s002
                                 + 1.12086 w_k1_KO_s002 + 1.00636 w_k1_MAT_s002
                                 + 1.00936 w_k1_MSFT_s002
                                 + 1.18289 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.12935 w_k1_ADM_s002
                                 + 1.01095 w_k1_AXP_s002
                                 + 0.820536 w_k1_DIS_s002
                                 + 1.00785 w_k1_FDX_s002
                                 + 0.748852 w_k1_IBM_s002
                                 + 0.889768 w_k1_INTC_s002
                                 + 0.98431 w_k1_KO_s002
                                 + 0.938108 w_k1_MAT_s002
                                 + 0.844185 w_k1_MSFT_s002
                                 + 0.985018 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.959303 w_k1_ADM_s002
                                 + 0.773302 w_k1_AXP_s002
                                 + 0.850703 w_k1_DIS_s002
                                 + 0.848897 w_k1_FDX_s002
                                 + 1.1314 w_k1_IBM_s002
                                 + 1.02257 w_k1_INTC_s002
                                 + 1.31249 w_k1_KO_s002 + 1.39675 w_k1_MAT_s002
                                 + 0.830993 w_k1_MSFT_s002
                                 + 1.12467 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.177 w_k1_ADM_s003
                                 + 1.29372 w_k1_AXP_s003
                                 + 1.32527 w_k1_DIS_s003
                                 + 1.21265 w_k1_FDX_s003 + 0.9625 w_k1_IBM_s003
                                 + 0.99704 w_k1_INTC_s003
                                 + 1.16521 w_k1_KO_s003
                                 + 0.776758 w_k1_MAT_s003
                                 + 1.20423 w_k1_MSFT_s003
                                 + 1.15424 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.926242 w_k1_ADM_s003
                                 + 0.925996 w_k1_AXP_s003
                                 + 1.22377 w_k1_DIS_s003
                                 + 1.03309 w_k1_FDX_s003
                                 + 0.861664 w_k1_IBM_s003
                                 + 1.08563 w_k1_INTC_s003
                                 + 1.13911 w_k1_KO_s003
                                 + 0.571788 w_k1_MAT_s003
                                 + 1.17848 w_k1_MSFT_s003
                                 + 1.17257 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.752362 w_k1_ADM_s003
                                 + 0.703198 w_k1_AXP_s003
                                 + 0.752482 w_k1_DIS_s003
                                 + 0.761045 w_k1_FDX_s003
                                 + 0.807059 w_k1_IBM_s003
                                 + 0.838663 w_k1_INTC_s003 + 0.774 w_k1_KO_s003
                                 + 0.445141 w_k1_MAT_s003
                                 + 0.911739 w_k1_MSFT_s003
                                 + 0.776727 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.857864 w_k1_ADM_s003
                                 + 1.22343 w_k1_AXP_s003
                                 + 1.50781 w_k1_DIS_s003
                                 + 1.02004 w_k1_FDX_s003
                                 + 1.31778 w_k1_IBM_s003
                                 + 1.17091 w_k1_INTC_s003
                                 + 1.26372 w_k1_KO_s003 + 1.42186 w_k1_MAT_s003
                                 + 1.3176 w_k1_MSFT_s003 + 1.05699 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.05676 w_k1_ADM_s003
                                 + 1.40739 w_k1_AXP_s003
                                 + 1.01186 w_k1_DIS_s003
                                 + 1.03427 w_k1_FDX_s003 + 1.1213 w_k1_IBM_s003
                                 + 0.804076 w_k1_INTC_s003
                                 + 1.06589 w_k1_KO_s003 + 1.21696 w_k1_MAT_s003
                                 + 0.958531 w_k1_MSFT_s003
                                 + 0.93329 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.86105 w_k1_ADM_s003
                                 + 0.988719 w_k1_AXP_s003
                                 + 0.898352 w_k1_DIS_s003
                                 + 0.810571 w_k1_FDX_s003
                                 + 1.06646 w_k1_IBM_s003
                                 + 0.908073 w_k1_INTC_s003
                                 + 0.935667 w_k1_KO_s003
                                 + 0.907162 w_k1_MAT_s003
                                 + 0.891565 w_k1_MSFT_s003
                                 + 0.94503 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.920959 w_k1_ADM_s003
                                 + 0.845446 w_k1_AXP_s003
                                 + 0.902753 w_k1_DIS_s003
                                 + 0.961088 w_k1_FDX_s003
                                 + 0.671666 w_k1_IBM_s003
                                 + 0.892807 w_k1_INTC_s003
                                 + 0.793971 w_k1_KO_s003
                                 + 0.395623 w_k1_MAT_s003
                                 + 0.968409 w_k1_MSFT_s003
                                 + 0.834114 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.46922 w_k1_ADM_s003
                                 + 1.85487 w_k1_AXP_s003
                                 + 1.66685 w_k1_DIS_s003
                                 + 1.47018 w_k1_FDX_s003
                                 + 0.87222 w_k1_IBM_s003
                                 + 1.23664 w_k1_INTC_s003 + 1.1583 w_k1_KO_s003
                                 + 1.31973 w_k1_MAT_s003
                                 + 1.50681 w_k1_MSFT_s003
                                 + 1.10411 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.932516 w_k1_ADM_s004
                                 + 1.06629 w_k1_AXP_s004
                                 + 0.858348 w_k1_DIS_s004
                                 + 1.05184 w_k1_FDX_s004
                                 + 0.981918 w_k1_IBM_s004
                                 + 0.985526 w_k1_INTC_s004
                                 + 0.918879 w_k1_KO_s004
                                 + 1.13546 w_k1_MAT_s004
                                 + 0.959359 w_k1_MSFT_s004
                                 + 1.02936 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.855683 w_k1_ADM_s004
                                 + 0.80056 w_k1_AXP_s004
                                 + 0.67712 w_k1_DIS_s004
                                 + 0.88067 w_k1_FDX_s004
                                 + 0.804098 w_k1_IBM_s004
                                 + 0.592533 w_k1_INTC_s004
                                 + 0.750657 w_k1_KO_s004
                                 + 1.14519 w_k1_MAT_s004
                                 + 0.832824 w_k1_MSFT_s004
                                 + 0.82582 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.06727 w_k1_ADM_s004
                                 + 1.22859 w_k1_AXP_s004
                                 + 0.739468 w_k1_DIS_s004
                                 + 0.934763 w_k1_FDX_s004
                                 + 1.30418 w_k1_IBM_s004
                                 + 0.758062 w_k1_INTC_s004
                                 + 0.891698 w_k1_KO_s004
                                 + 1.95289 w_k1_MAT_s004
                                 + 0.810335 w_k1_MSFT_s004
                                 + 0.975189 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.901031 w_k1_ADM_s004
                                 + 1.17727 w_k1_AXP_s004 + 1.0773 w_k1_DIS_s004
                                 + 0.987614 w_k1_FDX_s004
                                 + 1.25915 w_k1_IBM_s004
                                 + 0.960724 w_k1_INTC_s004
                                 + 1.0695 w_k1_KO_s004 + 1.29528 w_k1_MAT_s004
                                 + 0.965716 w_k1_MSFT_s004
                                 + 1.01392 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.996641 w_k1_ADM_s004
                                 + 0.841884 w_k1_AXP_s004
                                 + 0.803083 w_k1_DIS_s004
                                 + 0.853633 w_k1_FDX_s004
                                 + 0.896213 w_k1_IBM_s004
                                 + 1.06911 w_k1_INTC_s004
                                 + 0.833389 w_k1_KO_s004
                                 + 0.867899 w_k1_MAT_s004
                                 + 0.980815 w_k1_MSFT_s004
                                 + 1.03908 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.08784 w_k1_ADM_s004
                                 + 1.2068 w_k1_AXP_s004 + 1.12021 w_k1_DIS_s004
                                 + 1.22898 w_k1_FDX_s004
                                 + 1.05186 w_k1_IBM_s004
                                 + 0.981399 w_k1_INTC_s004
                                 + 1.16515 w_k1_KO_s004 + 1.38407 w_k1_MAT_s004
                                 + 1.12104 w_k1_MSFT_s004
                                 + 1.22689 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.863624 w_k1_ADM_s004
                                 + 0.924444 w_k1_AXP_s004
                                 + 1.05514 w_k1_DIS_s004
                                 + 1.09439 w_k1_FDX_s004
                                 + 0.621796 w_k1_IBM_s004
                                 + 0.924787 w_k1_INTC_s004
                                 + 0.835541 w_k1_KO_s004
                                 + 0.989866 w_k1_MAT_s004
                                 + 1.18687 w_k1_MSFT_s004
                                 + 0.994755 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.06218 w_k1_ADM_s004
                                 + 1.51082 w_k1_AXP_s004
                                 + 1.58386 w_k1_DIS_s004
                                 + 1.31949 w_k1_FDX_s004
                                 + 1.18707 w_k1_IBM_s004
                                 + 1.02522 w_k1_INTC_s004
                                 + 1.37095 w_k1_KO_s004 + 2.1805 w_k1_MAT_s004
                                 + 1.23949 w_k1_MSFT_s004
                                 + 1.27823 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.39996 w_k1_ADM_s005
                                 + 1.36122 w_k1_AXP_s005
                                 + 1.35508 w_k1_DIS_s005
                                 + 1.30515 w_k1_FDX_s005
                                 + 0.882587 w_k1_IBM_s005
                                 + 1.30845 w_k1_INTC_s005
                                 + 0.892239 w_k1_KO_s005
                                 + 1.08525 w_k1_MAT_s005
                                 + 1.19882 w_k1_MSFT_s005
                                 + 0.980975 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.842029 w_k1_ADM_s005
                                 + 1.04051 w_k1_AXP_s005
                                 + 0.854119 w_k1_DIS_s005
                                 + 0.915985 w_k1_FDX_s005
                                 + 1.16779 w_k1_IBM_s005
                                 + 0.909425 w_k1_INTC_s005
                                 + 0.78661 w_k1_KO_s005 + 1.33738 w_k1_MAT_s005
                                 + 0.890064 w_k1_MSFT_s005
                                 + 0.792283 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.904193 w_k1_ADM_s005
                                 + 0.8703 w_k1_AXP_s005
                                 + 0.813065 w_k1_DIS_s005
                                 + 0.894816 w_k1_FDX_s005
                                 + 0.770411 w_k1_IBM_s005
                                 + 0.809609 w_k1_INTC_s005
                                 + 0.675132 w_k1_KO_s005
                                 + 0.688907 w_k1_MAT_s005
                                 + 0.936876 w_k1_MSFT_s005
                                 + 0.869281 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.12103 w_k1_ADM_s005
                                 + 1.20327 w_k1_AXP_s005
                                 + 1.25147 w_k1_DIS_s005
                                 + 1.08437 w_k1_FDX_s005
                                 + 1.34397 w_k1_IBM_s005
                                 + 1.08451 w_k1_INTC_s005
                                 + 1.04055 w_k1_KO_s005 + 1.16169 w_k1_MAT_s005
                                 + 1.04817 w_k1_MSFT_s005
                                 + 0.961597 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.28086 w_k1_ADM_s005
                                 + 1.09568 w_k1_AXP_s005
                                 + 1.09529 w_k1_DIS_s005
                                 + 1.16685 w_k1_FDX_s005
                                 + 0.824985 w_k1_IBM_s005
                                 + 1.26808 w_k1_INTC_s005
                                 + 0.879355 w_k1_KO_s005
                                 + 0.734555 w_k1_MAT_s005
                                 + 1.25952 w_k1_MSFT_s005
                                 + 1.14199 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.881201 w_k1_ADM_s005
                                 + 0.72472 w_k1_AXP_s005
                                 + 0.684912 w_k1_DIS_s005
                                 + 0.797418 w_k1_FDX_s005
                                 + 0.788853 w_k1_IBM_s005
                                 + 0.941108 w_k1_INTC_s005
                                 + 0.660803 w_k1_KO_s005
                                 + 0.666622 w_k1_MAT_s005
                                 + 0.785917 w_k1_MSFT_s005
                                 + 0.83804 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.785869 w_k1_ADM_s005
                                 + 0.877761 w_k1_AXP_s005
                                 + 1.10146 w_k1_DIS_s005
                                 + 0.954154 w_k1_FDX_s005
                                 + 0.84991 w_k1_IBM_s005
                                 + 0.971722 w_k1_INTC_s005
                                 + 0.766556 w_k1_KO_s005
                                 + 0.838166 w_k1_MAT_s005
                                 + 0.974783 w_k1_MSFT_s005
                                 + 0.957349 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.02495 w_k1_ADM_s005
                                 + 0.917917 w_k1_AXP_s005
                                 + 1.10537 w_k1_DIS_s005
                                 + 1.04734 w_k1_FDX_s005
                                 + 1.12597 w_k1_IBM_s005
                                 + 1.05609 w_k1_INTC_s005
                                 + 0.98856 w_k1_KO_s005 + 1.2588 w_k1_MAT_s005
                                 + 1.05351 w_k1_MSFT_s005
                                 + 1.12732 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.853562 w_k1_ADM_s006
                                 + 1.2239 w_k1_AXP_s006 + 1.15173 w_k1_DIS_s006
                                 + 1.16264 w_k1_FDX_s006
                                 + 1.72491 w_k1_IBM_s006
                                 + 1.14349 w_k1_INTC_s006
                                 + 1.19914 w_k1_KO_s006 + 1.75561 w_k1_MAT_s006
                                 + 1.05245 w_k1_MSFT_s006
                                 + 1.18484 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.87117 w_k1_ADM_s006
                                 + 0.696079 w_k1_AXP_s006
                                 + 0.745766 w_k1_DIS_s006
                                 + 0.822755 w_k1_FDX_s006
                                 + 0.81424 w_k1_IBM_s006
                                 + 1.04856 w_k1_INTC_s006
                                 + 0.926864 w_k1_KO_s006
                                 + 0.835047 w_k1_MAT_s006
                                 + 1.00768 w_k1_MSFT_s006
                                 + 0.976556 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.16655 w_k1_ADM_s006
                                 + 0.840971 w_k1_AXP_s006
                                 + 1.28259 w_k1_DIS_s006
                                 + 1.16154 w_k1_FDX_s006
                                 + 0.796708 w_k1_IBM_s006
                                 + 1.09653 w_k1_INTC_s006
                                 + 1.06949 w_k1_KO_s006
                                 + 0.728412 w_k1_MAT_s006
                                 + 1.17542 w_k1_MSFT_s006
                                 + 1.12237 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.644 w_k1_ADM_s006
                                 + 0.87608 w_k1_AXP_s006
                                 + 0.955974 w_k1_DIS_s006
                                 + 0.862518 w_k1_FDX_s006
                                 + 1.11182 w_k1_IBM_s006
                                 + 1.38362 w_k1_INTC_s006
                                 + 1.20973 w_k1_KO_s006 + 1.09898 w_k1_MAT_s006
                                 + 1.15594 w_k1_MSFT_s006
                                 + 0.944706 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.32153 w_k1_ADM_s006
                                 + 1.01176 w_k1_AXP_s006
                                 + 1.34224 w_k1_DIS_s006 + 1.1039 w_k1_FDX_s006
                                 + 1.30341 w_k1_IBM_s006
                                 + 1.14692 w_k1_INTC_s006
                                 + 1.36647 w_k1_KO_s006 + 1.20516 w_k1_MAT_s006
                                 + 1.32103 w_k1_MSFT_s006
                                 + 1.08376 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.776899 w_k1_ADM_s006
                                 + 0.850158 w_k1_AXP_s006
                                 + 0.637406 w_k1_DIS_s006
                                 + 0.741313 w_k1_FDX_s006
                                 + 0.696073 w_k1_IBM_s006
                                 + 0.903139 w_k1_INTC_s006
                                 + 0.777525 w_k1_KO_s006
                                 + 0.953494 w_k1_MAT_s006
                                 + 0.936117 w_k1_MSFT_s006
                                 + 0.809157 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.09855 w_k1_ADM_s006
                                 + 1.3507 w_k1_AXP_s006 + 1.54987 w_k1_DIS_s006
                                 + 1.37305 w_k1_FDX_s006
                                 + 1.30466 w_k1_IBM_s006
                                 + 1.47605 w_k1_INTC_s006
                                 + 1.28369 w_k1_KO_s006 + 1.59369 w_k1_MAT_s006
                                 + 1.36966 w_k1_MSFT_s006
                                 + 1.11693 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.989141 w_k1_ADM_s006
                                 + 1.15428 w_k1_AXP_s006
                                 + 1.11393 w_k1_DIS_s006
                                 + 1.24536 w_k1_FDX_s006
                                 + 0.866281 w_k1_IBM_s006
                                 + 1.35394 w_k1_INTC_s006
                                 + 0.875713 w_k1_KO_s006
                                 + 0.767012 w_k1_MAT_s006
                                 + 1.29992 w_k1_MSFT_s006
                                 + 1.19069 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.17992 w_k1_ADM_s007
                                 + 1.36382 w_k1_AXP_s007
                                 + 1.69697 w_k1_DIS_s007
                                 + 1.21722 w_k1_FDX_s007
                                 + 1.65016 w_k1_IBM_s007
                                 + 1.2743 w_k1_INTC_s007 + 1.80165 w_k1_KO_s007
                                 + 2.98961 w_k1_MAT_s007
                                 + 1.28863 w_k1_MSFT_s007
                                 + 1.19775 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.96924 w_k1_ADM_s007
                                 + 0.850842 w_k1_AXP_s007
                                 + 1.16841 w_k1_DIS_s007
                                 + 1.03958 w_k1_FDX_s007
                                 + 1.17527 w_k1_IBM_s007
                                 + 1.11522 w_k1_INTC_s007
                                 + 1.11437 w_k1_KO_s007 + 1.18378 w_k1_MAT_s007
                                 + 1.1639 w_k1_MSFT_s007 + 1.07049 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.656217 w_k1_ADM_s007
                                 + 0.849781 w_k1_AXP_s007
                                 + 0.649637 w_k1_DIS_s007
                                 + 0.684347 w_k1_FDX_s007
                                 + 1.1099 w_k1_IBM_s007
                                 + 0.79074 w_k1_INTC_s007
                                 + 0.819619 w_k1_KO_s007
                                 + 1.79585 w_k1_MAT_s007
                                 + 0.770453 w_k1_MSFT_s007
                                 + 0.877143 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.22563 w_k1_ADM_s007
                                 + 1.28487 w_k1_AXP_s007
                                 + 1.70822 w_k1_DIS_s007
                                 + 1.61998 w_k1_FDX_s007
                                 + 0.975955 w_k1_IBM_s007
                                 + 1.1982 w_k1_INTC_s007 + 1.23199 w_k1_KO_s007
                                 + 1.48636 w_k1_MAT_s007
                                 + 1.37562 w_k1_MSFT_s007
                                 + 1.28484 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.25426 w_k1_ADM_s007
                                 + 1.33114 w_k1_AXP_s007
                                 + 1.27779 w_k1_DIS_s007 + 1.2934 w_k1_FDX_s007
                                 + 1.12241 w_k1_IBM_s007
                                 + 1.27784 w_k1_INTC_s007
                                 + 1.04078 w_k1_KO_s007 + 1.41142 w_k1_MAT_s007
                                 + 1.15429 w_k1_MSFT_s007
                                 + 1.02728 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.765882 w_k1_ADM_s007
                                 + 0.838461 w_k1_AXP_s007
                                 + 0.940649 w_k1_DIS_s007
                                 + 0.983162 w_k1_FDX_s007
                                 + 0.823188 w_k1_IBM_s007
                                 + 1.06359 w_k1_INTC_s007 + 0.7461 w_k1_KO_s007
                                 + 0.663662 w_k1_MAT_s007
                                 + 0.926712 w_k1_MSFT_s007
                                 + 0.869809 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.75832 w_k1_ADM_s007
                                 + 0.733476 w_k1_AXP_s007
                                 + 0.877779 w_k1_DIS_s007
                                 + 0.834132 w_k1_FDX_s007
                                 + 0.827863 w_k1_IBM_s007
                                 + 0.970146 w_k1_INTC_s007
                                 + 0.832511 w_k1_KO_s007
                                 + 0.941448 w_k1_MAT_s007
                                 + 0.773069 w_k1_MSFT_s007
                                 + 0.854378 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.996671 w_k1_ADM_s007
                                 + 1.42173 w_k1_AXP_s007
                                 + 1.09441 w_k1_DIS_s007
                                 + 1.05499 w_k1_FDX_s007 + 1.4759 w_k1_IBM_s007
                                 + 1.31523 w_k1_INTC_s007
                                 + 1.27547 w_k1_KO_s007 + 1.88983 w_k1_MAT_s007
                                 + 1.14407 w_k1_MSFT_s007
                                 + 1.13269 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.08466 w_k1_ADM_s010
                                 + 1.20121 w_k1_AXP_s010
                                 + 1.22413 w_k1_DIS_s010
                                 + 1.12278 w_k1_FDX_s010 + 1.2035 w_k1_IBM_s010
                                 + 1.03717 w_k1_INTC_s010
                                 + 1.05216 w_k1_KO_s010 + 1.88194 w_k1_MAT_s010
                                 + 1.21131 w_k1_MSFT_s010
                                 + 1.04066 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.995677 w_k1_ADM_s010
                                 + 0.810077 w_k1_AXP_s010
                                 + 0.778334 w_k1_DIS_s010
                                 + 0.74292 w_k1_FDX_s010
                                 + 1.04003 w_k1_IBM_s010
                                 + 0.812921 w_k1_INTC_s010
                                 + 0.813859 w_k1_KO_s010
                                 + 1.06695 w_k1_MAT_s010
                                 + 0.906557 w_k1_MSFT_s010
                                 + 0.857807 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.881576 w_k1_ADM_s010
                                 + 0.652711 w_k1_AXP_s010
                                 + 0.687573 w_k1_DIS_s010
                                 + 0.735122 w_k1_FDX_s010
                                 + 0.742863 w_k1_IBM_s010
                                 + 0.839229 w_k1_INTC_s010
                                 + 0.850948 w_k1_KO_s010
                                 + 0.530681 w_k1_MAT_s010
                                 + 0.880224 w_k1_MSFT_s010
                                 + 0.874014 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.05472 w_k1_ADM_s010
                                 + 1.17623 w_k1_AXP_s010
                                 + 1.59608 w_k1_DIS_s010
                                 + 1.24764 w_k1_FDX_s010
                                 + 1.11153 w_k1_IBM_s010
                                 + 1.13421 w_k1_INTC_s010
                                 + 1.17716 w_k1_KO_s010 + 1.07886 w_k1_MAT_s010
                                 + 1.31223 w_k1_MSFT_s010
                                 + 1.16801 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.01785 w_k1_ADM_s010
                                 + 0.736907 w_k1_AXP_s010
                                 + 0.813286 w_k1_DIS_s010
                                 + 0.90779 w_k1_FDX_s010
                                 + 0.669793 w_k1_IBM_s010
                                 + 0.717955 w_k1_INTC_s010
                                 + 0.754509 w_k1_KO_s010
                                 + 0.689237 w_k1_MAT_s010
                                 + 0.925015 w_k1_MSFT_s010
                                 + 0.839429 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.16602 w_k1_ADM_s010
                                 + 0.81057 w_k1_AXP_s010
                                 + 1.17184 w_k1_DIS_s010
                                 + 0.880964 w_k1_FDX_s010
                                 + 1.22294 w_k1_IBM_s010
                                 + 0.849958 w_k1_INTC_s010
                                 + 1.06189 w_k1_KO_s010 + 1.06407 w_k1_MAT_s010
                                 + 1.06884 w_k1_MSFT_s010
                                 + 1.02452 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.02347 w_k1_ADM_s010
                                 + 0.867892 w_k1_AXP_s010
                                 + 1.06696 w_k1_DIS_s010
                                 + 0.924486 w_k1_FDX_s010
                                 + 1.54139 w_k1_IBM_s010
                                 + 1.01184 w_k1_INTC_s010
                                 + 0.94382 w_k1_KO_s010 + 1.13304 w_k1_MAT_s010
                                 + 0.962971 w_k1_MSFT_s010
                                 + 0.946495 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04026 w_k1_ADM_s010
                                 + 0.895628 w_k1_AXP_s010
                                 + 1.00545 w_k1_DIS_s010
                                 + 0.983916 w_k1_FDX_s010
                                 + 0.78445 w_k1_IBM_s010
                                 + 0.823472 w_k1_INTC_s010
                                 + 0.851657 w_k1_KO_s010
                                 + 0.886195 w_k1_MAT_s010
                                 + 0.992887 w_k1_MSFT_s010
                                 + 0.971402 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.16451 w_k1_ADM_s011
                                 + 0.880544 w_k1_AXP_s011
                                 + 0.684599 w_k1_DIS_s011
                                 + 0.789288 w_k1_FDX_s011
                                 + 0.86507 w_k1_IBM_s011
                                 + 0.869988 w_k1_INTC_s011
                                 + 0.784596 w_k1_KO_s011
                                 + 1.08677 w_k1_MAT_s011
                                 + 0.836707 w_k1_MSFT_s011
                                 + 0.948664 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.57266 w_k1_ADM_s011
                                 + 1.43372 w_k1_AXP_s011
                                 + 1.24449 w_k1_DIS_s011
                                 + 1.21655 w_k1_FDX_s011
                                 + 1.05559 w_k1_IBM_s011
                                 + 1.42254 w_k1_INTC_s011
                                 + 1.11112 w_k1_KO_s011
                                 + 0.989357 w_k1_MAT_s011
                                 + 1.21231 w_k1_MSFT_s011
                                 + 1.27743 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.9378 w_k1_ADM_s011
                                 + 0.622732 w_k1_AXP_s011
                                 + 0.697161 w_k1_DIS_s011
                                 + 0.822264 w_k1_FDX_s011
                                 + 0.624799 w_k1_IBM_s011
                                 + 0.772589 w_k1_INTC_s011
                                 + 0.748973 w_k1_KO_s011
                                 + 0.525049 w_k1_MAT_s011
                                 + 0.909286 w_k1_MSFT_s011
                                 + 0.893306 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.05717 w_k1_ADM_s011
                                 + 1.33006 w_k1_AXP_s011 + 1.1203 w_k1_DIS_s011
                                 + 0.99306 w_k1_FDX_s011
                                 + 1.41481 w_k1_IBM_s011
                                 + 1.00111 w_k1_INTC_s011
                                 + 1.33121 w_k1_KO_s011 + 2.10177 w_k1_MAT_s011
                                 + 0.961088 w_k1_MSFT_s011
                                 + 1.08718 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.98983 w_k1_ADM_s011
                                 + 0.785058 w_k1_AXP_s011
                                 + 0.749032 w_k1_DIS_s011
                                 + 0.826356 w_k1_FDX_s011
                                 + 1.08076 w_k1_IBM_s011
                                 + 1.01938 w_k1_INTC_s011
                                 + 1.02152 w_k1_KO_s011 + 1.07761 w_k1_MAT_s011
                                 + 0.859349 w_k1_MSFT_s011
                                 + 1.01958 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.0135 w_k1_ADM_s011
                                 + 0.811641 w_k1_AXP_s011
                                 + 1.11692 w_k1_DIS_s011
                                 + 0.94806 w_k1_FDX_s011
                                 + 1.20873 w_k1_IBM_s011
                                 + 0.998642 w_k1_INTC_s011
                                 + 1.18034 w_k1_KO_s011
                                 + 0.932056 w_k1_MAT_s011
                                 + 0.92036 w_k1_MSFT_s011
                                 + 1.05771 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.42052 w_k1_ADM_s011
                                 + 1.13014 w_k1_AXP_s011
                                 + 1.66371 w_k1_DIS_s011
                                 + 1.41945 w_k1_FDX_s011
                                 + 1.09156 w_k1_IBM_s011
                                 + 1.32315 w_k1_INTC_s011
                                 + 1.31437 w_k1_KO_s011 + 1.36537 w_k1_MAT_s011
                                 + 1.47932 w_k1_MSFT_s011
                                 + 1.24341 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.28893 w_k1_ADM_s011
                                 + 1.02553 w_k1_AXP_s011
                                 + 0.937405 w_k1_DIS_s011
                                 + 1.02413 w_k1_FDX_s011
                                 + 0.840194 w_k1_IBM_s011
                                 + 1.05431 w_k1_INTC_s011
                                 + 0.948706 w_k1_KO_s011
                                 + 0.962231 w_k1_MAT_s011
                                 + 1.07264 w_k1_MSFT_s011
                                 + 1.08196 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.09856 w_k1_ADM_s012
                                 + 1.20512 w_k1_AXP_s012
                                 + 1.71757 w_k1_DIS_s012
                                 + 1.25498 w_k1_FDX_s012 + 1.2256 w_k1_IBM_s012
                                 + 1.07278 w_k1_INTC_s012
                                 + 1.27596 w_k1_KO_s012 + 1.49439 w_k1_MAT_s012
                                 + 1.25696 w_k1_MSFT_s012
                                 + 1.16335 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.00644 w_k1_ADM_s012
                                 + 1.50032 w_k1_AXP_s012
                                 + 1.16514 w_k1_DIS_s012
                                 + 1.06196 w_k1_FDX_s012
                                 + 2.00819 w_k1_IBM_s012
                                 + 1.05006 w_k1_INTC_s012 + 1.2293 w_k1_KO_s012
                                 + 1.8592 w_k1_MAT_s012 + 1.0386 w_k1_MSFT_s012
                                 + 0.962086 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.784835 w_k1_ADM_s012
                                 + 0.86384 w_k1_AXP_s012
                                 + 0.967648 w_k1_DIS_s012
                                 + 1.01412 w_k1_FDX_s012
                                 + 0.714762 w_k1_IBM_s012
                                 + 1.08453 w_k1_INTC_s012
                                 + 1.06395 w_k1_KO_s012
                                 + 0.972992 w_k1_MAT_s012
                                 + 1.19449 w_k1_MSFT_s012
                                 + 0.979563 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08736 w_k1_ADM_s012
                                 + 0.945616 w_k1_AXP_s012
                                 + 0.971336 w_k1_DIS_s012
                                 + 0.998774 w_k1_FDX_s012
                                 + 1.29163 w_k1_IBM_s012
                                 + 0.919626 w_k1_INTC_s012
                                 + 1.1446 w_k1_KO_s012 + 1.15866 w_k1_MAT_s012
                                 + 1.01977 w_k1_MSFT_s012
                                 + 1.02804 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.618477 w_k1_ADM_s012
                                 + 0.706051 w_k1_AXP_s012
                                 + 0.595503 w_k1_DIS_s012
                                 + 0.693683 w_k1_FDX_s012
                                 + 0.712871 w_k1_IBM_s012
                                 + 0.861241 w_k1_INTC_s012
                                 + 0.746127 w_k1_KO_s012
                                 + 0.670088 w_k1_MAT_s012
                                 + 0.847178 w_k1_MSFT_s012
                                 + 0.789504 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.856553 w_k1_ADM_s012
                                 + 0.803104 w_k1_AXP_s012
                                 + 0.861183 w_k1_DIS_s012
                                 + 0.89848 w_k1_FDX_s012
                                 + 0.946331 w_k1_IBM_s012
                                 + 0.768967 w_k1_INTC_s012
                                 + 0.9674 w_k1_KO_s012 + 1.08058 w_k1_MAT_s012
                                 + 1.03993 w_k1_MSFT_s012
                                 + 0.911258 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.02089 w_k1_ADM_s012
                                 + 0.931669 w_k1_AXP_s012
                                 + 1.38534 w_k1_DIS_s012
                                 + 1.20802 w_k1_FDX_s012
                                 + 0.763486 w_k1_IBM_s012
                                 + 1.22497 w_k1_INTC_s012
                                 + 1.09701 w_k1_KO_s012 + 0.9832 w_k1_MAT_s012
                                 + 1.62019 w_k1_MSFT_s012
                                 + 1.05889 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.19213 w_k1_ADM_s012
                                 + 1.23247 w_k1_AXP_s012 + 1.0944 w_k1_DIS_s012
                                 + 1.31179 w_k1_FDX_s012
                                 + 0.932736 w_k1_IBM_s012
                                 + 1.21189 w_k1_INTC_s012 + 1.1779 w_k1_KO_s012
                                 + 1.23238 w_k1_MAT_s012
                                 + 1.4809 w_k1_MSFT_s012 + 1.1614 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.874151 w_k1_ADM_s013
                                 + 0.927942 w_k1_AXP_s013
                                 + 0.718091 w_k1_DIS_s013
                                 + 0.778705 w_k1_FDX_s013
                                 + 1.04505 w_k1_IBM_s013
                                 + 0.986871 w_k1_INTC_s013
                                 + 0.785972 w_k1_KO_s013
                                 + 0.717206 w_k1_MAT_s013
                                 + 0.880132 w_k1_MSFT_s013
                                 + 0.847945 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.01187 w_k1_ADM_s013
                                 + 1.24859 w_k1_AXP_s013
                                 + 1.14059 w_k1_DIS_s013
                                 + 0.970113 w_k1_FDX_s013
                                 + 1.62099 w_k1_IBM_s013
                                 + 0.998089 w_k1_INTC_s013
                                 + 1.12676 w_k1_KO_s013 + 1.47101 w_k1_MAT_s013
                                 + 1.11186 w_k1_MSFT_s013
                                 + 1.06598 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.08169 w_k1_ADM_s013
                                 + 1.17713 w_k1_AXP_s013
                                 + 0.911986 w_k1_DIS_s013
                                 + 1.10864 w_k1_FDX_s013
                                 + 0.701042 w_k1_IBM_s013
                                 + 1.02282 w_k1_INTC_s013
                                 + 0.709316 w_k1_KO_s013
                                 + 0.762715 w_k1_MAT_s013
                                 + 1.13911 w_k1_MSFT_s013
                                 + 0.997438 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.723482 w_k1_ADM_s013
                                 + 0.871783 w_k1_AXP_s013
                                 + 1.03729 w_k1_DIS_s013
                                 + 0.931413 w_k1_FDX_s013
                                 + 0.871415 w_k1_IBM_s013
                                 + 1.06238 w_k1_INTC_s013
                                 + 0.822937 w_k1_KO_s013
                                 + 0.621754 w_k1_MAT_s013
                                 + 1.11148 w_k1_MSFT_s013
                                 + 0.862816 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.96528 w_k1_ADM_s013
                                 + 1.18016 w_k1_AXP_s013
                                 + 1.15333 w_k1_DIS_s013
                                 + 1.02135 w_k1_FDX_s013
                                 + 1.06087 w_k1_IBM_s013
                                 + 1.15255 w_k1_INTC_s013
                                 + 0.913832 w_k1_KO_s013
                                 + 1.07056 w_k1_MAT_s013
                                 + 1.10102 w_k1_MSFT_s013
                                 + 1.10931 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.807919 w_k1_ADM_s013
                                 + 0.936593 w_k1_AXP_s013
                                 + 0.674329 w_k1_DIS_s013
                                 + 0.958211 w_k1_FDX_s013
                                 + 0.70918 w_k1_IBM_s013
                                 + 0.925296 w_k1_INTC_s013
                                 + 0.729422 w_k1_KO_s013
                                 + 0.925315 w_k1_MAT_s013
                                 + 0.871741 w_k1_MSFT_s013
                                 + 0.93974 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.690263 w_k1_ADM_s013
                                 + 0.706006 w_k1_AXP_s013
                                 + 0.671725 w_k1_DIS_s013
                                 + 0.795774 w_k1_FDX_s013
                                 + 0.623408 w_k1_IBM_s013
                                 + 0.888111 w_k1_INTC_s013
                                 + 0.71562 w_k1_KO_s013
                                 + 0.662033 w_k1_MAT_s013
                                 + 0.870408 w_k1_MSFT_s013
                                 + 0.780881 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.902322 w_k1_ADM_s013
                                 + 1.40797 w_k1_AXP_s013
                                 + 1.50873 w_k1_DIS_s013
                                 + 1.24907 w_k1_FDX_s013
                                 + 1.07457 w_k1_IBM_s013
                                 + 1.25595 w_k1_INTC_s013
                                 + 1.11447 w_k1_KO_s013
                                 + 0.845938 w_k1_MAT_s013
                                 + 1.19899 w_k1_MSFT_s013
                                 + 1.11574 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.04982 w_k1_ADM_s014
                                 + 0.9615 w_k1_AXP_s014
                                 + 0.832741 w_k1_DIS_s014
                                 + 0.929535 w_k1_FDX_s014
                                 + 0.710947 w_k1_IBM_s014
                                 + 1.19463 w_k1_INTC_s014
                                 + 0.915456 w_k1_KO_s014
                                 + 0.612824 w_k1_MAT_s014
                                 + 0.937015 w_k1_MSFT_s014
                                 + 0.91601 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.789648 w_k1_ADM_s014
                                 + 0.697689 w_k1_AXP_s014
                                 + 0.55685 w_k1_DIS_s014
                                 + 0.663272 w_k1_FDX_s014
                                 + 0.642786 w_k1_IBM_s014
                                 + 0.827131 w_k1_INTC_s014
                                 + 0.570886 w_k1_KO_s014
                                 + 0.791261 w_k1_MAT_s014
                                 + 0.761746 w_k1_MSFT_s014
                                 + 0.658662 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.975596 w_k1_ADM_s014
                                 + 0.801567 w_k1_AXP_s014
                                 + 0.700091 w_k1_DIS_s014
                                 + 0.804093 w_k1_FDX_s014
                                 + 1.02973 w_k1_IBM_s014
                                 + 0.969699 w_k1_INTC_s014
                                 + 0.836132 w_k1_KO_s014
                                 + 0.866667 w_k1_MAT_s014
                                 + 0.832166 w_k1_MSFT_s014
                                 + 0.656948 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.24318 w_k1_ADM_s014
                                 + 1.47477 w_k1_AXP_s014 + 1.4353 w_k1_DIS_s014
                                 + 1.22773 w_k1_FDX_s014
                                 + 1.05134 w_k1_IBM_s014
                                 + 1.50332 w_k1_INTC_s014
                                 + 1.13681 w_k1_KO_s014
                                 + 0.906333 w_k1_MAT_s014
                                 + 1.19823 w_k1_MSFT_s014
                                 + 0.889911 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.28486 w_k1_ADM_s014
                                 + 1.52166 w_k1_AXP_s014 + 1.1843 w_k1_DIS_s014
                                 + 1.28578 w_k1_FDX_s014
                                 + 1.31363 w_k1_IBM_s014
                                 + 1.06399 w_k1_INTC_s014
                                 + 1.12407 w_k1_KO_s014 + 1.15168 w_k1_MAT_s014
                                 + 1.09637 w_k1_MSFT_s014
                                 + 1.09718 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.976067 w_k1_ADM_s014
                                 + 0.935403 w_k1_AXP_s014
                                 + 0.929258 w_k1_DIS_s014
                                 + 0.90576 w_k1_FDX_s014
                                 + 0.984124 w_k1_IBM_s014
                                 + 1.2788 w_k1_INTC_s014
                                 + 0.955649 w_k1_KO_s014
                                 + 0.870669 w_k1_MAT_s014
                                 + 1.04793 w_k1_MSFT_s014
                                 + 0.867969 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.977088 w_k1_ADM_s014
                                 + 1.08199 w_k1_AXP_s014
                                 + 1.35003 w_k1_DIS_s014
                                 + 1.05068 w_k1_FDX_s014
                                 + 1.36501 w_k1_IBM_s014
                                 + 1.24753 w_k1_INTC_s014 + 1.1851 w_k1_KO_s014
                                 + 1.18959 w_k1_MAT_s014
                                 + 1.17495 w_k1_MSFT_s014 + 0.9896 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.30041 w_k1_ADM_s014
                                 + 1.50392 w_k1_AXP_s014
                                 + 1.73131 w_k1_DIS_s014
                                 + 1.39419 w_k1_FDX_s014
                                 + 1.94919 w_k1_IBM_s014
                                 + 1.61556 w_k1_INTC_s014 + 1.5824 w_k1_KO_s014
                                 + 1.58182 w_k1_MAT_s014
                                 + 1.44408 w_k1_MSFT_s014
                                 + 1.27254 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.4333 w_k1_ADM_s015
                                 + 1.24622 w_k1_AXP_s015
                                 + 1.04169 w_k1_DIS_s015
                                 + 1.23204 w_k1_FDX_s015
                                 + 0.965998 w_k1_IBM_s015
                                 + 0.935263 w_k1_INTC_s015
                                 + 1.0792 w_k1_KO_s015 + 1.33077 w_k1_MAT_s015
                                 + 1.12601 w_k1_MSFT_s015
                                 + 1.17764 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.759143 w_k1_ADM_s015
                                 + 0.764756 w_k1_AXP_s015
                                 + 0.698807 w_k1_DIS_s015
                                 + 0.793062 w_k1_FDX_s015
                                 + 0.94701 w_k1_IBM_s015
                                 + 0.692697 w_k1_INTC_s015
                                 + 0.82069 w_k1_KO_s015 + 1.14519 w_k1_MAT_s015
                                 + 0.694352 w_k1_MSFT_s015
                                 + 0.931265 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.17045 w_k1_ADM_s015
                                 + 1.1151 w_k1_AXP_s015 + 1.07501 w_k1_DIS_s015
                                 + 1.18937 w_k1_FDX_s015
                                 + 0.810368 w_k1_IBM_s015
                                 + 0.929404 w_k1_INTC_s015
                                 + 0.882273 w_k1_KO_s015
                                 + 1.18449 w_k1_MAT_s015
                                 + 1.09162 w_k1_MSFT_s015
                                 + 1.11082 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.43339 w_k1_ADM_s015
                                 + 0.907325 w_k1_AXP_s015
                                 + 0.89762 w_k1_DIS_s015 + 0.877 w_k1_FDX_s015
                                 + 1.27394 w_k1_IBM_s015
                                 + 0.838548 w_k1_INTC_s015
                                 + 1.07321 w_k1_KO_s015
                                 + 0.966273 w_k1_MAT_s015
                                 + 0.924047 w_k1_MSFT_s015
                                 + 1.05714 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.813635 w_k1_ADM_s015
                                 + 0.691715 w_k1_AXP_s015
                                 + 0.590614 w_k1_DIS_s015
                                 + 0.747295 w_k1_FDX_s015
                                 + 0.663145 w_k1_IBM_s015
                                 + 0.78141 w_k1_INTC_s015
                                 + 0.648879 w_k1_KO_s015
                                 + 0.643875 w_k1_MAT_s015
                                 + 0.771405 w_k1_MSFT_s015
                                 + 0.846854 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.21804 w_k1_ADM_s015
                                 + 1.42764 w_k1_AXP_s015
                                 + 1.67855 w_k1_DIS_s015
                                 + 1.49695 w_k1_FDX_s015
                                 + 1.07298 w_k1_IBM_s015
                                 + 1.49095 w_k1_INTC_s015
                                 + 1.48046 w_k1_KO_s015 + 1.34244 w_k1_MAT_s015
                                 + 1.42289 w_k1_MSFT_s015
                                 + 1.28432 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.988631 w_k1_ADM_s015
                                 + 0.798015 w_k1_AXP_s015
                                 + 0.954419 w_k1_DIS_s015
                                 + 0.945117 w_k1_FDX_s015
                                 + 1.02889 w_k1_IBM_s015
                                 + 0.780244 w_k1_INTC_s015
                                 + 1.10248 w_k1_KO_s015 + 1.37893 w_k1_MAT_s015
                                 + 0.982974 w_k1_MSFT_s015
                                 + 1.11605 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.04516 w_k1_ADM_s015
                                 + 0.837128 w_k1_AXP_s015
                                 + 0.785561 w_k1_DIS_s015
                                 + 0.918095 w_k1_FDX_s015
                                 + 0.648218 w_k1_IBM_s015
                                 + 0.902831 w_k1_INTC_s015
                                 + 0.829502 w_k1_KO_s015
                                 + 0.850403 w_k1_MAT_s015
                                 + 0.946265 w_k1_MSFT_s015
                                 + 1.10282 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.3774 w_k1_ADM_s016
                                 + 1.19813 w_k1_AXP_s016
                                 + 1.70404 w_k1_DIS_s016
                                 + 1.40194 w_k1_FDX_s016
                                 + 1.00453 w_k1_IBM_s016
                                 + 1.25453 w_k1_INTC_s016
                                 + 1.12534 w_k1_KO_s016
                                 + 0.886214 w_k1_MAT_s016
                                 + 1.21783 w_k1_MSFT_s016
                                 + 1.15995 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.720323 w_k1_ADM_s016
                                 + 0.693023 w_k1_AXP_s016
                                 + 0.640481 w_k1_DIS_s016
                                 + 0.87371 w_k1_FDX_s016
                                 + 0.53292 w_k1_IBM_s016
                                 + 0.938045 w_k1_INTC_s016
                                 + 0.555061 w_k1_KO_s016
                                 + 0.647762 w_k1_MAT_s016
                                 + 0.964143 w_k1_MSFT_s016
                                 + 0.761935 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.996828 w_k1_ADM_s016
                                 + 0.743492 w_k1_AXP_s016
                                 + 0.564601 w_k1_DIS_s016
                                 + 0.760078 w_k1_FDX_s016
                                 + 0.7266 w_k1_IBM_s016
                                 + 0.83134 w_k1_INTC_s016
                                 + 0.622336 w_k1_KO_s016
                                 + 0.534267 w_k1_MAT_s016
                                 + 0.777108 w_k1_MSFT_s016
                                 + 0.802558 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.0935 w_k1_ADM_s016
                                 + 1.40703 w_k1_AXP_s016
                                 + 1.34152 w_k1_DIS_s016
                                 + 1.08909 w_k1_FDX_s016
                                 + 1.49113 w_k1_IBM_s016
                                 + 1.04393 w_k1_INTC_s016
                                 + 1.14812 w_k1_KO_s016 + 1.49002 w_k1_MAT_s016
                                 + 0.931115 w_k1_MSFT_s016
                                 + 1.1369 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.736204 w_k1_ADM_s016
                                 + 0.817806 w_k1_AXP_s016
                                 + 0.696495 w_k1_DIS_s016
                                 + 0.790779 w_k1_FDX_s016
                                 + 1.05106 w_k1_IBM_s016
                                 + 0.890524 w_k1_INTC_s016
                                 + 0.80583 w_k1_KO_s016 + 1.32469 w_k1_MAT_s016
                                 + 0.839901 w_k1_MSFT_s016
                                 + 0.803681 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.847212 w_k1_ADM_s016
                                 + 0.946244 w_k1_AXP_s016
                                 + 1.07805 w_k1_DIS_s016
                                 + 0.899858 w_k1_FDX_s016 + 1.233 w_k1_IBM_s016
                                 + 1.07446 w_k1_INTC_s016
                                 + 0.956263 w_k1_KO_s016
                                 + 0.968858 w_k1_MAT_s016
                                 + 0.960324 w_k1_MSFT_s016
                                 + 1.06718 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.05461 w_k1_ADM_s016
                                 + 0.92647 w_k1_AXP_s016
                                 + 1.01997 w_k1_DIS_s016
                                 + 1.05543 w_k1_FDX_s016
                                 + 0.827146 w_k1_IBM_s016
                                 + 1.00359 w_k1_INTC_s016
                                 + 0.767344 w_k1_KO_s016
                                 + 0.811819 w_k1_MAT_s016
                                 + 1.04855 w_k1_MSFT_s016
                                 + 1.00841 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.952765 w_k1_ADM_s016
                                 + 1.28909 w_k1_AXP_s016
                                 + 0.968766 w_k1_DIS_s016
                                 + 0.846247 w_k1_FDX_s016
                                 + 2.10777 w_k1_IBM_s016
                                 + 0.961638 w_k1_INTC_s016
                                 + 1.24894 w_k1_KO_s016 + 2.58529 w_k1_MAT_s016
                                 + 0.919632 w_k1_MSFT_s016
                                 + 1.07292 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.846473 w_k1_ADM_s017
                                 + 0.925143 w_k1_AXP_s017
                                 + 0.657245 w_k1_DIS_s017
                                 + 0.79679 w_k1_FDX_s017
                                 + 0.957561 w_k1_IBM_s017
                                 + 0.750269 w_k1_INTC_s017
                                 + 0.69019 w_k1_KO_s017 + 1.04047 w_k1_MAT_s017
                                 + 0.783584 w_k1_MSFT_s017
                                 + 0.896158 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.00689 w_k1_ADM_s017
                                 + 1.18574 w_k1_AXP_s017 + 1.7245 w_k1_DIS_s017
                                 + 1.23178 w_k1_FDX_s017
                                 + 1.25919 w_k1_IBM_s017
                                 + 1.15993 w_k1_INTC_s017
                                 + 1.43118 w_k1_KO_s017 + 1.46616 w_k1_MAT_s017
                                 + 1.22206 w_k1_MSFT_s017
                                 + 1.25046 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.19945 w_k1_ADM_s017
                                 + 1.43898 w_k1_AXP_s017
                                 + 1.84004 w_k1_DIS_s017
                                 + 1.53519 w_k1_FDX_s017
                                 + 1.09414 w_k1_IBM_s017
                                 + 1.54976 w_k1_INTC_s017
                                 + 1.11348 w_k1_KO_s017 + 1.12062 w_k1_MAT_s017
                                 + 1.31941 w_k1_MSFT_s017
                                 + 1.30774 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.851431 w_k1_ADM_s017
                                 + 0.940399 w_k1_AXP_s017
                                 + 0.857267 w_k1_DIS_s017
                                 + 0.883323 w_k1_FDX_s017
                                 + 1.19292 w_k1_IBM_s017
                                 + 1.01621 w_k1_INTC_s017
                                 + 1.00364 w_k1_KO_s017 + 1.32141 w_k1_MAT_s017
                                 + 0.906989 w_k1_MSFT_s017
                                 + 1.1227 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.09289 w_k1_ADM_s017
                                 + 1.15838 w_k1_AXP_s017
                                 + 1.12943 w_k1_DIS_s017
                                 + 1.07044 w_k1_FDX_s017
                                 + 1.05019 w_k1_IBM_s017
                                 + 1.25665 w_k1_INTC_s017
                                 + 1.18892 w_k1_KO_s017 + 1.09801 w_k1_MAT_s017
                                 + 1.15201 w_k1_MSFT_s017
                                 + 1.14664 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.954643 w_k1_ADM_s017
                                 + 1.2692 w_k1_AXP_s017 + 1.11218 w_k1_DIS_s017
                                 + 1.08783 w_k1_FDX_s017
                                 + 1.67061 w_k1_IBM_s017
                                 + 1.0873 w_k1_INTC_s017 + 1.33187 w_k1_KO_s017
                                 + 1.93836 w_k1_MAT_s017
                                 + 1.06725 w_k1_MSFT_s017
                                 + 1.23069 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.979906 w_k1_ADM_s017
                                 + 0.819509 w_k1_AXP_s017
                                 + 1.40761 w_k1_DIS_s017 + 1.1826 w_k1_FDX_s017
                                 + 0.79056 w_k1_IBM_s017
                                 + 1.1385 w_k1_INTC_s017 + 1.00742 w_k1_KO_s017
                                 + 0.719799 w_k1_MAT_s017
                                 + 1.31616 w_k1_MSFT_s017 + 1.0587 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.67471 w_k1_ADM_s017
                                 + 0.767542 w_k1_AXP_s017
                                 + 0.770988 w_k1_DIS_s017
                                 + 0.844695 w_k1_FDX_s017
                                 + 0.761405 w_k1_IBM_s017
                                 + 1.12702 w_k1_INTC_s017
                                 + 0.724137 w_k1_KO_s017
                                 + 0.820571 w_k1_MAT_s017
                                 + 0.869421 w_k1_MSFT_s017
                                 + 0.926236 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.825351 w_k1_ADM_s100
                                 + 0.945068 w_k1_AXP_s100
                                 + 0.710625 w_k1_DIS_s100
                                 + 0.871196 w_k1_FDX_s100
                                 + 1.26605 w_k1_IBM_s100
                                 + 0.837174 w_k1_INTC_s100
                                 + 0.958717 w_k1_KO_s100
                                 + 1.50714 w_k1_MAT_s100
                                 + 0.861711 w_k1_MSFT_s100
                                 + 0.980126 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.702751 w_k1_ADM_s100
                                 + 0.621236 w_k1_AXP_s100
                                 + 0.457863 w_k1_DIS_s100
                                 + 0.691046 w_k1_FDX_s100
                                 + 0.674718 w_k1_IBM_s100
                                 + 0.755308 w_k1_INTC_s100
                                 + 0.72465 w_k1_KO_s100
                                 + 0.885615 w_k1_MAT_s100
                                 + 0.73946 w_k1_MSFT_s100
                                 + 0.792995 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.37715 w_k1_ADM_s100
                                 + 1.33043 w_k1_AXP_s100
                                 + 1.49545 w_k1_DIS_s100
                                 + 1.17909 w_k1_FDX_s100
                                 + 1.53691 w_k1_IBM_s100
                                 + 1.13895 w_k1_INTC_s100 + 1.4395 w_k1_KO_s100
                                 + 1.93108 w_k1_MAT_s100
                                 + 1.18247 w_k1_MSFT_s100
                                 + 1.22591 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.30435 w_k1_ADM_s100
                                 + 1.50273 w_k1_AXP_s100
                                 + 1.39958 w_k1_DIS_s100
                                 + 1.43446 w_k1_FDX_s100
                                 + 1.01235 w_k1_IBM_s100
                                 + 1.44742 w_k1_INTC_s100
                                 + 1.25625 w_k1_KO_s100
                                 + 0.886957 w_k1_MAT_s100 + 1.35 w_k1_MSFT_s100
                                 + 1.12361 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.736717 w_k1_ADM_s100
                                 + 0.703804 w_k1_AXP_s100
                                 + 0.643796 w_k1_DIS_s100
                                 + 0.762678 w_k1_FDX_s100
                                 + 0.933355 w_k1_IBM_s100
                                 + 0.946306 w_k1_INTC_s100
                                 + 0.958291 w_k1_KO_s100
                                 + 1.23582 w_k1_MAT_s100
                                 + 0.798748 w_k1_MSFT_s100
                                 + 0.816031 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.989338 w_k1_ADM_s100
                                 + 1.19084 w_k1_AXP_s100
                                 + 1.04294 w_k1_DIS_s100
                                 + 1.15664 w_k1_FDX_s100
                                 + 0.828421 w_k1_IBM_s100
                                 + 1.17119 w_k1_INTC_s100
                                 + 0.993329 w_k1_KO_s100
                                 + 1.10758 w_k1_MAT_s100
                                 + 1.17911 w_k1_MSFT_s100
                                 + 1.04679 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.19238 w_k1_ADM_s100
                                 + 1.46165 w_k1_AXP_s100
                                 + 1.11167 w_k1_DIS_s100
                                 + 1.17341 w_k1_FDX_s100
                                 + 1.27122 w_k1_IBM_s100
                                 + 0.927852 w_k1_INTC_s100
                                 + 1.16032 w_k1_KO_s100 + 1.60495 w_k1_MAT_s100
                                 + 1.09427 w_k1_MSFT_s100
                                 + 1.09072 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.911567 w_k1_ADM_s100
                                 + 0.966241 w_k1_AXP_s100
                                 + 0.770613 w_k1_DIS_s100
                                 + 0.920894 w_k1_FDX_s100
                                 + 0.855076 w_k1_IBM_s100
                                 + 0.998318 w_k1_INTC_s100
                                 + 0.797729 w_k1_KO_s100
                                 + 0.885404 w_k1_MAT_s100
                                 + 0.944403 w_k1_MSFT_s100
                                 + 0.891691 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.837713 w_k1_ADM_s101
                                 + 0.767595 w_k1_AXP_s101
                                 + 0.881752 w_k1_DIS_s101
                                 + 1.00168 w_k1_FDX_s101
                                 + 0.651011 w_k1_IBM_s101
                                 + 1.19302 w_k1_INTC_s101
                                 + 0.892613 w_k1_KO_s101
                                 + 0.613649 w_k1_MAT_s101
                                 + 0.938174 w_k1_MSFT_s101
                                 + 0.946564 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.877509 w_k1_ADM_s101
                                 + 0.980937 w_k1_AXP_s101
                                 + 0.749851 w_k1_DIS_s101
                                 + 0.931125 w_k1_FDX_s101
                                 + 0.857149 w_k1_IBM_s101
                                 + 1.06606 w_k1_INTC_s101
                                 + 0.850086 w_k1_KO_s101
                                 + 1.00258 w_k1_MAT_s101
                                 + 0.830111 w_k1_MSFT_s101
                                 + 0.917086 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12879 w_k1_ADM_s101
                                 + 1.06074 w_k1_AXP_s101
                                 + 1.48965 w_k1_DIS_s101
                                 + 1.27532 w_k1_FDX_s101
                                 + 1.37609 w_k1_IBM_s101
                                 + 1.16929 w_k1_INTC_s101
                                 + 1.31601 w_k1_KO_s101 + 1.38371 w_k1_MAT_s101
                                 + 1.27277 w_k1_MSFT_s101
                                 + 1.12831 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.14325 w_k1_ADM_s101
                                 + 0.983498 w_k1_AXP_s101
                                 + 1.03925 w_k1_DIS_s101
                                 + 1.22762 w_k1_FDX_s101
                                 + 0.774842 w_k1_IBM_s101
                                 + 0.971406 w_k1_INTC_s101
                                 + 0.988266 w_k1_KO_s101
                                 + 0.917196 w_k1_MAT_s101
                                 + 1.05238 w_k1_MSFT_s101
                                 + 0.992378 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.04192 w_k1_ADM_s101
                                 + 1.22334 w_k1_AXP_s101
                                 + 1.20579 w_k1_DIS_s101
                                 + 1.25096 w_k1_FDX_s101
                                 + 1.08931 w_k1_IBM_s101
                                 + 1.0618 w_k1_INTC_s101 + 1.056 w_k1_KO_s101
                                 + 1.43517 w_k1_MAT_s101
                                 + 1.03215 w_k1_MSFT_s101
                                 + 1.14632 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.688801 w_k1_ADM_s101
                                 + 0.667733 w_k1_AXP_s101
                                 + 0.533792 w_k1_DIS_s101
                                 + 0.727691 w_k1_FDX_s101
                                 + 0.686127 w_k1_IBM_s101
                                 + 0.802885 w_k1_INTC_s101
                                 + 0.711774 w_k1_KO_s101
                                 + 0.555046 w_k1_MAT_s101
                                 + 0.676483 w_k1_MSFT_s101
                                 + 0.747002 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.06188 w_k1_ADM_s101
                                 + 0.938426 w_k1_AXP_s101
                                 + 1.07443 w_k1_DIS_s101
                                 + 1.09841 w_k1_FDX_s101
                                 + 1.02629 w_k1_IBM_s101
                                 + 0.970353 w_k1_INTC_s101
                                 + 1.07931 w_k1_KO_s101
                                 + 0.779986 w_k1_MAT_s101
                                 + 1.05209 w_k1_MSFT_s101
                                 + 1.00444 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.831961 w_k1_ADM_s101
                                 + 0.743449 w_k1_AXP_s101
                                 + 0.823756 w_k1_DIS_s101
                                 + 0.816764 w_k1_FDX_s101
                                 + 1.08365 w_k1_IBM_s101
                                 + 0.926326 w_k1_INTC_s101
                                 + 0.947062 w_k1_KO_s101
                                 + 1.08274 w_k1_MAT_s101
                                 + 0.915476 w_k1_MSFT_s101
                                 + 0.914829 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.2655 w_k1_ADM_s102
                                 + 1.26184 w_k1_AXP_s102
                                 + 1.31913 w_k1_DIS_s102
                                 + 1.24684 w_k1_FDX_s102
                                 + 1.21049 w_k1_IBM_s102
                                 + 1.31751 w_k1_INTC_s102
                                 + 1.20463 w_k1_KO_s102 + 1.166 w_k1_MAT_s102
                                 + 1.11795 w_k1_MSFT_s102 + 1.2043 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.01452 w_k1_ADM_s102
                                 + 0.953163 w_k1_AXP_s102
                                 + 0.877919 w_k1_DIS_s102
                                 + 1.02393 w_k1_FDX_s102
                                 + 0.905105 w_k1_IBM_s102
                                 + 1.06437 w_k1_INTC_s102
                                 + 1.03303 w_k1_KO_s102
                                 + 0.891236 w_k1_MAT_s102
                                 + 0.937746 w_k1_MSFT_s102
                                 + 1.03657 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.28732 w_k1_ADM_s102
                                 + 1.23947 w_k1_AXP_s102 + 1.0439 w_k1_DIS_s102
                                 + 1.03671 w_k1_FDX_s102
                                 + 1.31323 w_k1_IBM_s102
                                 + 0.932767 w_k1_INTC_s102
                                 + 1.30205 w_k1_KO_s102 + 1.40105 w_k1_MAT_s102
                                 + 1.03871 w_k1_MSFT_s102
                                 + 1.15856 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.13389 w_k1_ADM_s102
                                 + 1.49948 w_k1_AXP_s102
                                 + 1.32263 w_k1_DIS_s102
                                 + 1.22134 w_k1_FDX_s102
                                 + 1.87485 w_k1_IBM_s102
                                 + 1.19157 w_k1_INTC_s102 + 1.4723 w_k1_KO_s102
                                 + 1.76072 w_k1_MAT_s102
                                 + 1.0961 w_k1_MSFT_s102 + 1.21464 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.801774 w_k1_ADM_s102
                                 + 0.798061 w_k1_AXP_s102
                                 + 0.660763 w_k1_DIS_s102
                                 + 0.812042 w_k1_FDX_s102
                                 + 0.748009 w_k1_IBM_s102
                                 + 0.981472 w_k1_INTC_s102
                                 + 0.817672 w_k1_KO_s102
                                 + 0.700091 w_k1_MAT_s102
                                 + 0.80364 w_k1_MSFT_s102 + 0.9038 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04715 w_k1_ADM_s102
                                 + 0.964077 w_k1_AXP_s102
                                 + 0.871361 w_k1_DIS_s102
                                 + 0.842566 w_k1_FDX_s102
                                 + 1.26688 w_k1_IBM_s102
                                 + 0.94967 w_k1_INTC_s102
                                 + 1.17963 w_k1_KO_s102
                                 + 0.844143 w_k1_MAT_s102
                                 + 0.918204 w_k1_MSFT_s102
                                 + 1.05086 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.41363 w_k1_ADM_s102
                                 + 1.29561 w_k1_AXP_s102
                                 + 1.35083 w_k1_DIS_s102
                                 + 1.27612 w_k1_FDX_s102
                                 + 0.81597 w_k1_IBM_s102
                                 + 1.23601 w_k1_INTC_s102
                                 + 1.17254 w_k1_KO_s102
                                 + 0.829827 w_k1_MAT_s102
                                 + 1.18264 w_k1_MSFT_s102 + 1.2571 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.859281 w_k1_ADM_s102
                                 + 0.78555 w_k1_AXP_s102
                                 + 0.722933 w_k1_DIS_s102
                                 + 0.947271 w_k1_FDX_s102
                                 + 0.487848 w_k1_IBM_s102
                                 + 0.936881 w_k1_INTC_s102
                                 + 0.642293 w_k1_KO_s102
                                 + 0.400579 w_k1_MAT_s102
                                 + 0.932449 w_k1_MSFT_s102
                                 + 0.826175 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.11002 w_k1_ADM_s103
                                 + 1.48628 w_k1_AXP_s103
                                 + 0.949449 w_k1_DIS_s103
                                 + 1.04256 w_k1_FDX_s103 + 1.2466 w_k1_IBM_s103
                                 + 1.12643 w_k1_INTC_s103
                                 + 1.11746 w_k1_KO_s103 + 1.91025 w_k1_MAT_s103
                                 + 1.00984 w_k1_MSFT_s103 + 1.0448 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.05651 w_k1_ADM_s103
                                 + 1.32626 w_k1_AXP_s103
                                 + 0.924035 w_k1_DIS_s103
                                 + 1.08109 w_k1_FDX_s103
                                 + 0.842607 w_k1_IBM_s103
                                 + 1.08484 w_k1_INTC_s103
                                 + 0.956007 w_k1_KO_s103
                                 + 1.00322 w_k1_MAT_s103
                                 + 0.919069 w_k1_MSFT_s103
                                 + 1.07882 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.803535 w_k1_ADM_s103
                                 + 0.810398 w_k1_AXP_s103
                                 + 0.757384 w_k1_DIS_s103
                                 + 0.807068 w_k1_FDX_s103
                                 + 0.932335 w_k1_IBM_s103
                                 + 0.806621 w_k1_INTC_s103
                                 + 0.873024 w_k1_KO_s103
                                 + 1.16754 w_k1_MAT_s103
                                 + 0.870432 w_k1_MSFT_s103
                                 + 0.81389 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.786215 w_k1_ADM_s103
                                 + 0.771782 w_k1_AXP_s103
                                 + 0.729123 w_k1_DIS_s103
                                 + 0.786648 w_k1_FDX_s103
                                 + 0.665946 w_k1_IBM_s103
                                 + 0.799815 w_k1_INTC_s103
                                 + 0.760861 w_k1_KO_s103
                                 + 0.769889 w_k1_MAT_s103
                                 + 0.82652 w_k1_MSFT_s103
                                 + 0.883923 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.01539 w_k1_ADM_s103
                                 + 1.17364 w_k1_AXP_s103
                                 + 1.47961 w_k1_DIS_s103
                                 + 1.22376 w_k1_FDX_s103
                                 + 1.10916 w_k1_IBM_s103
                                 + 1.18405 w_k1_INTC_s103
                                 + 1.18585 w_k1_KO_s103 + 1.09974 w_k1_MAT_s103
                                 + 1.17649 w_k1_MSFT_s103
                                 + 1.06135 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.993616 w_k1_ADM_s103
                                 + 0.93003 w_k1_AXP_s103
                                 + 0.980457 w_k1_DIS_s103
                                 + 0.999245 w_k1_FDX_s103
                                 + 0.620591 w_k1_IBM_s103
                                 + 0.987234 w_k1_INTC_s103
                                 + 0.88696 w_k1_KO_s103
                                 + 0.552038 w_k1_MAT_s103
                                 + 0.908813 w_k1_MSFT_s103
                                 + 0.891055 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.796019 w_k1_ADM_s103
                                 + 1.3936 w_k1_AXP_s103 + 1.04079 w_k1_DIS_s103
                                 + 0.952598 w_k1_FDX_s103
                                 + 1.69536 w_k1_IBM_s103 + 1.053 w_k1_INTC_s103
                                 + 1.15052 w_k1_KO_s103 + 2.03724 w_k1_MAT_s103
                                 + 0.908307 w_k1_MSFT_s103
                                 + 1.06074 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.753843 w_k1_ADM_s103
                                 + 0.964012 w_k1_AXP_s103
                                 + 0.843571 w_k1_DIS_s103
                                 + 0.786928 w_k1_FDX_s103
                                 + 1.12394 w_k1_IBM_s103
                                 + 0.958501 w_k1_INTC_s103
                                 + 0.876168 w_k1_KO_s103
                                 + 1.15401 w_k1_MAT_s103
                                 + 0.863946 w_k1_MSFT_s103
                                 + 0.879328 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.909629 w_k1_ADM_s104
                                 + 1.16254 w_k1_AXP_s104
                                 + 1.17656 w_k1_DIS_s104
                                 + 1.13904 w_k1_FDX_s104
                                 + 1.04228 w_k1_IBM_s104
                                 + 0.991391 w_k1_INTC_s104
                                 + 1.03297 w_k1_KO_s104 + 1.06649 w_k1_MAT_s104
                                 + 1.05901 w_k1_MSFT_s104
                                 + 1.03639 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.846273 w_k1_ADM_s104
                                 + 1.03611 w_k1_AXP_s104
                                 + 0.950413 w_k1_DIS_s104
                                 + 0.916995 w_k1_FDX_s104
                                 + 1.45899 w_k1_IBM_s104
                                 + 0.990323 w_k1_INTC_s104
                                 + 1.13781 w_k1_KO_s104 + 1.65505 w_k1_MAT_s104
                                 + 1.00888 w_k1_MSFT_s104
                                 + 1.04832 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.860987 w_k1_ADM_s104
                                 + 1.05655 w_k1_AXP_s104
                                 + 0.728741 w_k1_DIS_s104
                                 + 0.86999 w_k1_FDX_s104
                                 + 1.28613 w_k1_IBM_s104
                                 + 0.799547 w_k1_INTC_s104
                                 + 0.958599 w_k1_KO_s104
                                 + 1.66107 w_k1_MAT_s104
                                 + 0.892626 w_k1_MSFT_s104
                                 + 0.953806 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.951114 w_k1_ADM_s104
                                 + 1.38247 w_k1_AXP_s104 + 1.611 w_k1_DIS_s104
                                 + 1.2543 w_k1_FDX_s104 + 1.39778 w_k1_IBM_s104
                                 + 1.19472 w_k1_INTC_s104
                                 + 1.23945 w_k1_KO_s104 + 1.66337 w_k1_MAT_s104
                                 + 1.12144 w_k1_MSFT_s104
                                 + 1.11179 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.00284 w_k1_ADM_s104
                                 + 1.03092 w_k1_AXP_s104
                                 + 1.00705 w_k1_DIS_s104
                                 + 1.24561 w_k1_FDX_s104
                                 + 0.675524 w_k1_IBM_s104
                                 + 1.05754 w_k1_INTC_s104
                                 + 0.764077 w_k1_KO_s104
                                 + 0.934854 w_k1_MAT_s104
                                 + 1.15261 w_k1_MSFT_s104
                                 + 0.914738 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.835135 w_k1_ADM_s104
                                 + 0.938574 w_k1_AXP_s104
                                 + 0.896845 w_k1_DIS_s104
                                 + 0.936649 w_k1_FDX_s104
                                 + 0.892282 w_k1_IBM_s104
                                 + 0.932002 w_k1_INTC_s104
                                 + 0.978702 w_k1_KO_s104
                                 + 1.26015 w_k1_MAT_s104
                                 + 1.03175 w_k1_MSFT_s104
                                 + 0.973701 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.780555 w_k1_ADM_s104
                                 + 0.701015 w_k1_AXP_s104
                                 + 0.691023 w_k1_DIS_s104
                                 + 0.779711 w_k1_FDX_s104
                                 + 0.852485 w_k1_IBM_s104
                                 + 0.753416 w_k1_INTC_s104
                                 + 0.840218 w_k1_KO_s104
                                 + 0.744263 w_k1_MAT_s104
                                 + 0.854027 w_k1_MSFT_s104
                                 + 0.844593 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.3531 w_k1_ADM_s104
                                 + 1.31172 w_k1_AXP_s104
                                 + 1.29691 w_k1_DIS_s104
                                 + 1.38382 w_k1_FDX_s104
                                 + 0.785605 w_k1_IBM_s104
                                 + 1.33978 w_k1_INTC_s104
                                 + 1.07288 w_k1_KO_s104
                                 + 0.650278 w_k1_MAT_s104
                                 + 1.28971 w_k1_MSFT_s104
                                 + 1.17697 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.06223 w_k1_ADM_s105
                                 + 1.16848 w_k1_AXP_s105
                                 + 0.970689 w_k1_DIS_s105
                                 + 1.12538 w_k1_FDX_s105
                                 + 0.875818 w_k1_IBM_s105
                                 + 1.00206 w_k1_INTC_s105
                                 + 0.90843 w_k1_KO_s105 + 1.02826 w_k1_MAT_s105
                                 + 1.01108 w_k1_MSFT_s105
                                 + 1.02998 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.43352 w_k1_ADM_s105
                                 + 1.81973 w_k1_AXP_s105
                                 + 1.91616 w_k1_DIS_s105
                                 + 1.54455 w_k1_FDX_s105
                                 + 1.50974 w_k1_IBM_s105 + 1.517 w_k1_INTC_s105
                                 + 1.4972 w_k1_KO_s105 + 1.79805 w_k1_MAT_s105
                                 + 1.53074 w_k1_MSFT_s105
                                 + 1.22447 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.40056 w_k1_ADM_s105
                                 + 0.98656 w_k1_AXP_s105
                                 + 0.832163 w_k1_DIS_s105
                                 + 1.08644 w_k1_FDX_s105
                                 + 0.657159 w_k1_IBM_s105
                                 + 0.77873 w_k1_INTC_s105
                                 + 0.716896 w_k1_KO_s105
                                 + 0.645893 w_k1_MAT_s105
                                 + 0.918694 w_k1_MSFT_s105
                                 + 1.00504 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.09505 w_k1_ADM_s105
                                 + 1.03721 w_k1_AXP_s105
                                 + 1.22804 w_k1_DIS_s105
                                 + 1.16402 w_k1_FDX_s105
                                 + 1.08036 w_k1_IBM_s105
                                 + 1.11811 w_k1_INTC_s105
                                 + 1.05166 w_k1_KO_s105
                                 + 0.948644 w_k1_MAT_s105
                                 + 1.28647 w_k1_MSFT_s105
                                 + 1.15047 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.11 w_k1_ADM_s105
                                 + 1.19736 w_k1_AXP_s105
                                 + 1.14866 w_k1_DIS_s105
                                 + 1.02853 w_k1_FDX_s105
                                 + 1.54456 w_k1_IBM_s105
                                 + 0.952716 w_k1_INTC_s105
                                 + 1.27105 w_k1_KO_s105 + 1.41281 w_k1_MAT_s105
                                 + 1.03622 w_k1_MSFT_s105
                                 + 1.01201 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.71604 w_k1_ADM_s105
                                 + 1.52705 w_k1_AXP_s105
                                 + 1.52749 w_k1_DIS_s105
                                 + 1.23064 w_k1_FDX_s105
                                 + 1.21177 w_k1_IBM_s105
                                 + 1.13794 w_k1_INTC_s105 + 1.2758 w_k1_KO_s105
                                 + 1.23721 w_k1_MAT_s105
                                 + 1.2319 w_k1_MSFT_s105 + 1.14841 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.849299 w_k1_ADM_s105
                                 + 0.901929 w_k1_AXP_s105
                                 + 0.664582 w_k1_DIS_s105
                                 + 0.802046 w_k1_FDX_s105
                                 + 0.98629 w_k1_IBM_s105
                                 + 0.898063 w_k1_INTC_s105
                                 + 0.797543 w_k1_KO_s105
                                 + 0.947531 w_k1_MAT_s105
                                 + 0.813695 w_k1_MSFT_s105
                                 + 0.831308 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.90305 w_k1_ADM_s105
                                 + 0.671189 w_k1_AXP_s105
                                 + 0.623855 w_k1_DIS_s105
                                 + 0.731932 w_k1_FDX_s105
                                 + 0.684422 w_k1_IBM_s105
                                 + 0.71537 w_k1_INTC_s105
                                 + 0.692789 w_k1_KO_s105
                                 + 0.54962 w_k1_MAT_s105
                                 + 0.716207 w_k1_MSFT_s105
                                 + 0.748819 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.845985 w_k1_ADM_s106
                                 + 0.911644 w_k1_AXP_s106
                                 + 0.864688 w_k1_DIS_s106
                                 + 0.971922 w_k1_FDX_s106
                                 + 0.745728 w_k1_IBM_s106
                                 + 1.05964 w_k1_INTC_s106
                                 + 0.790245 w_k1_KO_s106
                                 + 0.846443 w_k1_MAT_s106
                                 + 0.980402 w_k1_MSFT_s106
                                 + 0.873302 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.10548 w_k1_ADM_s106
                                 + 1.26587 w_k1_AXP_s106
                                 + 1.17136 w_k1_DIS_s106
                                 + 1.29594 w_k1_FDX_s106
                                 + 0.911579 w_k1_IBM_s106
                                 + 1.1181 w_k1_INTC_s106
                                 + 0.855857 w_k1_KO_s106
                                 + 1.16169 w_k1_MAT_s106
                                 + 1.31074 w_k1_MSFT_s106
                                 + 1.04503 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.984202 w_k1_ADM_s106
                                 + 1.28409 w_k1_AXP_s106
                                 + 1.23512 w_k1_DIS_s106
                                 + 1.00138 w_k1_FDX_s106
                                 + 1.82469 w_k1_IBM_s106
                                 + 1.0696 w_k1_INTC_s106 + 1.27443 w_k1_KO_s106
                                 + 1.99614 w_k1_MAT_s106
                                 + 1.00991 w_k1_MSFT_s106
                                 + 1.16072 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.87113 w_k1_ADM_s106
                                 + 1.05334 w_k1_AXP_s106
                                 + 1.19524 w_k1_DIS_s106
                                 + 1.10218 w_k1_FDX_s106
                                 + 1.16715 w_k1_IBM_s106
                                 + 1.32194 w_k1_INTC_s106 + 1.0615 w_k1_KO_s106
                                 + 1.3091 w_k1_MAT_s106
                                 + 1.20593 w_k1_MSFT_s106
                                 + 0.993067 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.02503 w_k1_ADM_s106
                                 + 1.55643 w_k1_AXP_s106
                                 + 1.69402 w_k1_DIS_s106
                                 + 1.39733 w_k1_FDX_s106 + 1.2158 w_k1_IBM_s106
                                 + 1.74636 w_k1_INTC_s106 + 1.1874 w_k1_KO_s106
                                 + 1.09653 w_k1_MAT_s106
                                 + 1.5801 w_k1_MSFT_s106 + 1.23066 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.804781 w_k1_ADM_s106
                                 + 0.691752 w_k1_AXP_s106
                                 + 0.583195 w_k1_DIS_s106
                                 + 0.730493 w_k1_FDX_s106
                                 + 0.910805 w_k1_IBM_s106
                                 + 1.05688 w_k1_INTC_s106
                                 + 0.718385 w_k1_KO_s106
                                 + 0.705211 w_k1_MAT_s106
                                 + 0.79638 w_k1_MSFT_s106
                                 + 0.814771 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.08806 w_k1_ADM_s106
                                 + 0.942739 w_k1_AXP_s106
                                 + 0.979279 w_k1_DIS_s106
                                 + 0.916744 w_k1_FDX_s106
                                 + 1.33768 w_k1_IBM_s106
                                 + 0.83544 w_k1_INTC_s106
                                 + 1.02889 w_k1_KO_s106 + 1.15618 w_k1_MAT_s106
                                 + 0.967943 w_k1_MSFT_s106
                                 + 1.00613 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.724019 w_k1_ADM_s106
                                 + 0.963877 w_k1_AXP_s106
                                 + 0.762421 w_k1_DIS_s106
                                 + 0.764747 w_k1_FDX_s106
                                 + 1.27884 w_k1_IBM_s106
                                 + 0.93103 w_k1_INTC_s106
                                 + 0.866358 w_k1_KO_s106
                                 + 1.28705 w_k1_MAT_s106
                                 + 0.880952 w_k1_MSFT_s106
                                 + 0.885896 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.77954 w_k1_ADM_s107
                                 + 1.76526 w_k1_AXP_s107
                                 + 1.39849 w_k1_DIS_s107
                                 + 1.47441 w_k1_FDX_s107
                                 + 1.02926 w_k1_IBM_s107
                                 + 1.21679 w_k1_INTC_s107
                                 + 0.952563 w_k1_KO_s107
                                 + 0.899801 w_k1_MAT_s107
                                 + 1.09992 w_k1_MSFT_s107
                                 + 1.13464 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.07788 w_k1_ADM_s107
                                 + 1.18478 w_k1_AXP_s107
                                 + 0.870843 w_k1_DIS_s107
                                 + 0.960469 w_k1_FDX_s107
                                 + 1.11294 w_k1_IBM_s107
                                 + 1.07339 w_k1_INTC_s107
                                 + 1.06128 w_k1_KO_s107 + 1.26753 w_k1_MAT_s107
                                 + 0.929726 w_k1_MSFT_s107
                                 + 1.05659 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.20435 w_k1_ADM_s107
                                 + 1.31498 w_k1_AXP_s107
                                 + 1.20588 w_k1_DIS_s107
                                 + 1.09146 w_k1_FDX_s107
                                 + 1.58471 w_k1_IBM_s107
                                 + 0.976435 w_k1_INTC_s107
                                 + 1.38313 w_k1_KO_s107 + 1.43259 w_k1_MAT_s107
                                 + 1.01642 w_k1_MSFT_s107
                                 + 1.13206 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.993394 w_k1_ADM_s107
                                 + 0.742249 w_k1_AXP_s107
                                 + 0.688389 w_k1_DIS_s107
                                 + 0.831744 w_k1_FDX_s107
                                 + 0.784838 w_k1_IBM_s107
                                 + 0.921896 w_k1_INTC_s107
                                 + 0.793143 w_k1_KO_s107
                                 + 0.560693 w_k1_MAT_s107
                                 + 0.883316 w_k1_MSFT_s107
                                 + 0.828414 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.17232 w_k1_ADM_s107
                                 + 0.995916 w_k1_AXP_s107
                                 + 1.01506 w_k1_DIS_s107
                                 + 1.06939 w_k1_FDX_s107
                                 + 0.816583 w_k1_IBM_s107
                                 + 1.02136 w_k1_INTC_s107
                                 + 0.92422 w_k1_KO_s107 + 0.77263 w_k1_MAT_s107
                                 + 1.06718 w_k1_MSFT_s107
                                 + 0.955996 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.34552 w_k1_ADM_s107
                                 + 1.26023 w_k1_AXP_s107
                                 + 1.01239 w_k1_DIS_s107
                                 + 1.10883 w_k1_FDX_s107
                                 + 1.06238 w_k1_IBM_s107
                                 + 1.35346 w_k1_INTC_s107
                                 + 0.988253 w_k1_KO_s107
                                 + 0.869014 w_k1_MAT_s107
                                 + 1.02629 w_k1_MSFT_s107
                                 + 1.03784 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.49968 w_k1_ADM_s107
                                 + 1.63658 w_k1_AXP_s107
                                 + 1.45319 w_k1_DIS_s107
                                 + 1.26597 w_k1_FDX_s107
                                 + 1.43376 w_k1_IBM_s107
                                 + 1.34942 w_k1_INTC_s107
                                 + 1.36372 w_k1_KO_s107 + 1.28818 w_k1_MAT_s107
                                 + 1.04763 w_k1_MSFT_s107 + 1.1425 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.00963 w_k1_ADM_s107
                                 + 0.861426 w_k1_AXP_s107
                                 + 0.945264 w_k1_DIS_s107
                                 + 0.890545 w_k1_FDX_s107
                                 + 1.45938 w_k1_IBM_s107
                                 + 1.07455 w_k1_INTC_s107
                                 + 1.18269 w_k1_KO_s107
                                 + 0.883249 w_k1_MAT_s107
                                 + 0.949424 w_k1_MSFT_s107
                                 + 0.981032 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.0235 w_k1_ADM_s110
                                 + 0.871993 w_k1_AXP_s110
                                 + 0.916385 w_k1_DIS_s110
                                 + 0.944297 w_k1_FDX_s110
                                 + 0.721951 w_k1_IBM_s110
                                 + 0.759977 w_k1_INTC_s110
                                 + 0.841471 w_k1_KO_s110
                                 + 0.662872 w_k1_MAT_s110
                                 + 0.916098 w_k1_MSFT_s110
                                 + 0.922195 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.800072 w_k1_ADM_s110
                                 + 0.896063 w_k1_AXP_s110
                                 + 0.741939 w_k1_DIS_s110
                                 + 0.826519 w_k1_FDX_s110
                                 + 0.961675 w_k1_IBM_s110
                                 + 0.802878 w_k1_INTC_s110
                                 + 0.925887 w_k1_KO_s110
                                 + 0.805691 w_k1_MAT_s110
                                 + 0.805458 w_k1_MSFT_s110
                                 + 0.87631 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.11691 w_k1_ADM_s110
                                 + 1.18982 w_k1_AXP_s110
                                 + 1.10676 w_k1_DIS_s110
                                 + 1.02413 w_k1_FDX_s110
                                 + 1.60627 w_k1_IBM_s110
                                 + 0.817826 w_k1_INTC_s110
                                 + 1.27498 w_k1_KO_s110 + 1.63088 w_k1_MAT_s110
                                 + 0.925386 w_k1_MSFT_s110
                                 + 1.10581 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.08797 w_k1_ADM_s110
                                 + 1.35913 w_k1_AXP_s110
                                 + 1.55589 w_k1_DIS_s110
                                 + 1.25929 w_k1_FDX_s110
                                 + 1.18801 w_k1_IBM_s110
                                 + 1.04818 w_k1_INTC_s110
                                 + 1.18926 w_k1_KO_s110 + 1.28468 w_k1_MAT_s110
                                 + 1.19532 w_k1_MSFT_s110
                                 + 1.09621 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.48436 w_k1_ADM_s110
                                 + 1.6783 w_k1_AXP_s110 + 1.04846 w_k1_DIS_s110
                                 + 1.20524 w_k1_FDX_s110
                                 + 1.05814 w_k1_IBM_s110
                                 + 0.927761 w_k1_INTC_s110
                                 + 0.909909 w_k1_KO_s110
                                 + 1.31012 w_k1_MAT_s110
                                 + 0.829527 w_k1_MSFT_s110
                                 + 1.14937 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.20921 w_k1_ADM_s110
                                 + 1.00392 w_k1_AXP_s110
                                 + 1.08777 w_k1_DIS_s110
                                 + 0.996183 w_k1_FDX_s110
                                 + 1.09329 w_k1_IBM_s110
                                 + 0.805444 w_k1_INTC_s110
                                 + 0.991501 w_k1_KO_s110
                                 + 1.00355 w_k1_MAT_s110
                                 + 0.963034 w_k1_MSFT_s110
                                 + 1.0092 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.740048 w_k1_ADM_s110
                                 + 0.579484 w_k1_AXP_s110
                                 + 0.53882 w_k1_DIS_s110
                                 + 0.708585 w_k1_FDX_s110
                                 + 0.685237 w_k1_IBM_s110
                                 + 0.596682 w_k1_INTC_s110
                                 + 0.644205 w_k1_KO_s110
                                 + 0.544498 w_k1_MAT_s110
                                 + 0.707868 w_k1_MSFT_s110
                                 + 0.823737 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.10087 w_k1_ADM_s110
                                 + 0.902254 w_k1_AXP_s110
                                 + 1.29193 w_k1_DIS_s110
                                 + 1.15045 w_k1_FDX_s110
                                 + 0.931998 w_k1_IBM_s110
                                 + 0.915226 w_k1_INTC_s110
                                 + 0.98237 w_k1_KO_s110
                                 + 0.838865 w_k1_MAT_s110
                                 + 1.19585 w_k1_MSFT_s110
                                 + 0.978071 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.690618 w_k1_ADM_s111
                                 + 0.636835 w_k1_AXP_s111
                                 + 0.651369 w_k1_DIS_s111
                                 + 0.57302 w_k1_FDX_s111
                                 + 1.05599 w_k1_IBM_s111
                                 + 0.60933 w_k1_INTC_s111
                                 + 0.911444 w_k1_KO_s111
                                 + 0.960883 w_k1_MAT_s111
                                 + 0.653187 w_k1_MSFT_s111
                                 + 0.880378 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.18527 w_k1_ADM_s111
                                 + 1.0904 w_k1_AXP_s111 + 1.22672 w_k1_DIS_s111
                                 + 1.21194 w_k1_FDX_s111
                                 + 0.789499 w_k1_IBM_s111
                                 + 1.02256 w_k1_INTC_s111
                                 + 1.03469 w_k1_KO_s111
                                 + 0.961329 w_k1_MAT_s111
                                 + 1.24295 w_k1_MSFT_s111
                                 + 1.13608 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.00986 w_k1_ADM_s111
                                 + 1.18969 w_k1_AXP_s111
                                 + 0.824623 w_k1_DIS_s111
                                 + 0.942748 w_k1_FDX_s111
                                 + 0.997699 w_k1_IBM_s111
                                 + 0.625267 w_k1_INTC_s111
                                 + 0.819882 w_k1_KO_s111
                                 + 1.50004 w_k1_MAT_s111
                                 + 0.845117 w_k1_MSFT_s111
                                 + 0.966317 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.01288 w_k1_ADM_s111
                                 + 0.79033 w_k1_AXP_s111
                                 + 0.977224 w_k1_DIS_s111
                                 + 0.908222 w_k1_FDX_s111
                                 + 0.69913 w_k1_IBM_s111
                                 + 0.884485 w_k1_INTC_s111
                                 + 0.902833 w_k1_KO_s111
                                 + 0.660492 w_k1_MAT_s111
                                 + 0.953503 w_k1_MSFT_s111
                                 + 0.884678 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.852073 w_k1_ADM_s111
                                 + 0.667642 w_k1_AXP_s111
                                 + 0.844429 w_k1_DIS_s111
                                 + 0.807929 w_k1_FDX_s111
                                 + 0.979776 w_k1_IBM_s111
                                 + 0.825657 w_k1_INTC_s111
                                 + 1.01306 w_k1_KO_s111 + 1.10167 w_k1_MAT_s111
                                 + 0.896808 w_k1_MSFT_s111
                                 + 0.991736 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.26236 w_k1_ADM_s111
                                 + 1.04965 w_k1_AXP_s111 + 1.0402 w_k1_DIS_s111
                                 + 1.13965 w_k1_FDX_s111
                                 + 0.769713 w_k1_IBM_s111
                                 + 0.884109 w_k1_INTC_s111
                                 + 0.862195 w_k1_KO_s111
                                 + 0.594962 w_k1_MAT_s111
                                 + 0.949934 w_k1_MSFT_s111
                                 + 0.937734 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.09792 w_k1_ADM_s111
                                 + 0.9675 w_k1_AXP_s111 + 1.6154 w_k1_DIS_s111
                                 + 1.15253 w_k1_FDX_s111
                                 + 1.30358 w_k1_IBM_s111
                                 + 1.27865 w_k1_INTC_s111
                                 + 1.39384 w_k1_KO_s111 + 1.26699 w_k1_MAT_s111
                                 + 1.4599 w_k1_MSFT_s111 + 1.16124 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.25049 w_k1_ADM_s111
                                 + 1.42317 w_k1_AXP_s111
                                 + 1.22118 w_k1_DIS_s111
                                 + 1.10453 w_k1_FDX_s111
                                 + 1.47192 w_k1_IBM_s111
                                 + 1.10557 w_k1_INTC_s111
                                 + 1.24382 w_k1_KO_s111 + 1.57333 w_k1_MAT_s111
                                 + 0.983483 w_k1_MSFT_s111
                                 + 1.12308 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.866611 w_k1_ADM_s112
                                 + 0.801162 w_k1_AXP_s112
                                 + 1.00499 w_k1_DIS_s112
                                 + 0.933895 w_k1_FDX_s112
                                 + 0.906971 w_k1_IBM_s112
                                 + 0.936725 w_k1_INTC_s112
                                 + 1.15265 w_k1_KO_s112 + 0.95753 w_k1_MAT_s112
                                 + 0.97914 w_k1_MSFT_s112
                                 + 0.938301 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.611633 w_k1_ADM_s112
                                 + 0.623589 w_k1_AXP_s112
                                 + 0.633046 w_k1_DIS_s112
                                 + 0.722932 w_k1_FDX_s112
                                 + 0.960453 w_k1_IBM_s112
                                 + 0.641021 w_k1_INTC_s112
                                 + 0.830663 w_k1_KO_s112
                                 + 0.991024 w_k1_MAT_s112
                                 + 0.778726 w_k1_MSFT_s112
                                 + 0.902238 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.90766 w_k1_ADM_s112
                                 + 0.957552 w_k1_AXP_s112
                                 + 0.842787 w_k1_DIS_s112
                                 + 0.760279 w_k1_FDX_s112
                                 + 1.4119 w_k1_IBM_s112
                                 + 0.76763 w_k1_INTC_s112
                                 + 1.12015 w_k1_KO_s112 + 1.72355 w_k1_MAT_s112
                                 + 0.733782 w_k1_MSFT_s112
                                 + 0.965049 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.05819 w_k1_ADM_s112
                                 + 1.23174 w_k1_AXP_s112
                                 + 1.39475 w_k1_DIS_s112
                                 + 1.14278 w_k1_FDX_s112
                                 + 1.22533 w_k1_IBM_s112
                                 + 1.18949 w_k1_INTC_s112 + 1.173 w_k1_KO_s112
                                 + 1.7541 w_k1_MAT_s112
                                 + 1.27941 w_k1_MSFT_s112
                                 + 1.25632 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.02803 w_k1_ADM_s112
                                 + 0.94003 w_k1_AXP_s112
                                 + 0.929534 w_k1_DIS_s112
                                 + 1.04879 w_k1_FDX_s112
                                 + 0.867371 w_k1_IBM_s112
                                 + 0.931672 w_k1_INTC_s112
                                 + 0.815373 w_k1_KO_s112
                                 + 0.949901 w_k1_MAT_s112
                                 + 0.910281 w_k1_MSFT_s112
                                 + 0.934643 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.91337 w_k1_ADM_s112
                                 + 0.926284 w_k1_AXP_s112
                                 + 0.935381 w_k1_DIS_s112
                                 + 0.946232 w_k1_FDX_s112
                                 + 1.18967 w_k1_IBM_s112
                                 + 0.981005 w_k1_INTC_s112
                                 + 1.02693 w_k1_KO_s112 + 1.72288 w_k1_MAT_s112
                                 + 0.916043 w_k1_MSFT_s112
                                 + 0.988837 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.01685 w_k1_ADM_s112
                                 + 1.45864 w_k1_AXP_s112 + 1.2371 w_k1_DIS_s112
                                 + 1.16861 w_k1_FDX_s112
                                 + 0.863717 w_k1_IBM_s112
                                 + 1.07323 w_k1_INTC_s112
                                 + 1.17229 w_k1_KO_s112 + 1.56547 w_k1_MAT_s112
                                 + 1.20365 w_k1_MSFT_s112
                                 + 1.12376 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.86744 w_k1_ADM_s112
                                 + 0.606847 w_k1_AXP_s112
                                 + 0.59939 w_k1_DIS_s112
                                 + 0.797092 w_k1_FDX_s112
                                 + 0.640221 w_k1_IBM_s112
                                 + 0.793708 w_k1_INTC_s112
                                 + 0.633854 w_k1_KO_s112
                                 + 0.710263 w_k1_MAT_s112
                                 + 0.799528 w_k1_MSFT_s112
                                 + 0.805447 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.97723 w_k1_ADM_s113
                                 + 1.02597 w_k1_AXP_s113
                                 + 0.842349 w_k1_DIS_s113
                                 + 0.974924 w_k1_FDX_s113
                                 + 0.690166 w_k1_IBM_s113
                                 + 0.965291 w_k1_INTC_s113
                                 + 0.917948 w_k1_KO_s113
                                 + 0.824454 w_k1_MAT_s113
                                 + 0.923796 w_k1_MSFT_s113
                                 + 1.00244 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.10436 w_k1_ADM_s113
                                 + 0.913821 w_k1_AXP_s113
                                 + 1.16537 w_k1_DIS_s113 + 1.1691 w_k1_FDX_s113
                                 + 0.662313 w_k1_IBM_s113
                                 + 0.950092 w_k1_INTC_s113
                                 + 1.06402 w_k1_KO_s113 + 0.85094 w_k1_MAT_s113
                                 + 1.14093 w_k1_MSFT_s113
                                 + 1.09648 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.19215 w_k1_ADM_s113
                                 + 1.27907 w_k1_AXP_s113
                                 + 1.62593 w_k1_DIS_s113
                                 + 1.37826 w_k1_FDX_s113
                                 + 1.09264 w_k1_IBM_s113
                                 + 1.20093 w_k1_INTC_s113
                                 + 1.41567 w_k1_KO_s113 + 1.84184 w_k1_MAT_s113
                                 + 1.42959 w_k1_MSFT_s113
                                 + 1.39068 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.847837 w_k1_ADM_s113
                                 + 0.789272 w_k1_AXP_s113
                                 + 0.779745 w_k1_DIS_s113
                                 + 0.825863 w_k1_FDX_s113
                                 + 1.01866 w_k1_IBM_s113
                                 + 0.86065 w_k1_INTC_s113
                                 + 0.933366 w_k1_KO_s113
                                 + 0.82734 w_k1_MAT_s113
                                 + 0.817376 w_k1_MSFT_s113
                                 + 0.926828 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.974188 w_k1_ADM_s113
                                 + 0.89609 w_k1_AXP_s113
                                 + 1.00308 w_k1_DIS_s113
                                 + 0.958165 w_k1_FDX_s113
                                 + 1.60045 w_k1_IBM_s113
                                 + 1.01504 w_k1_INTC_s113
                                 + 1.35296 w_k1_KO_s113 + 1.8589 w_k1_MAT_s113
                                 + 0.893652 w_k1_MSFT_s113
                                 + 1.02388 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.926575 w_k1_ADM_s113
                                 + 0.717646 w_k1_AXP_s113
                                 + 0.862321 w_k1_DIS_s113
                                 + 0.753613 w_k1_FDX_s113
                                 + 1.10742 w_k1_IBM_s113
                                 + 0.949168 w_k1_INTC_s113
                                 + 1.06093 w_k1_KO_s113
                                 + 0.930413 w_k1_MAT_s113
                                 + 0.887716 w_k1_MSFT_s113
                                 + 0.982381 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08851 w_k1_ADM_s113
                                 + 0.919014 w_k1_AXP_s113
                                 + 1.26726 w_k1_DIS_s113
                                 + 1.00581 w_k1_FDX_s113
                                 + 1.16173 w_k1_IBM_s113
                                 + 0.939782 w_k1_INTC_s113
                                 + 1.14359 w_k1_KO_s113 + 1.18129 w_k1_MAT_s113
                                 + 1.14627 w_k1_MSFT_s113
                                 + 1.14568 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.2437 w_k1_ADM_s113
                                 + 1.46052 w_k1_AXP_s113
                                 + 1.34042 w_k1_DIS_s113
                                 + 1.20505 w_k1_FDX_s113
                                 + 1.49267 w_k1_IBM_s113
                                 + 1.28499 w_k1_INTC_s113
                                 + 1.68525 w_k1_KO_s113 + 2.3256 w_k1_MAT_s113
                                 + 1.19092 w_k1_MSFT_s113
                                 + 1.29169 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.07985 w_k1_ADM_s114
                                 + 0.909086 w_k1_AXP_s114
                                 + 1.22494 w_k1_DIS_s114
                                 + 1.10193 w_k1_FDX_s114
                                 + 0.967163 w_k1_IBM_s114
                                 + 1.08472 w_k1_INTC_s114
                                 + 0.985799 w_k1_KO_s114
                                 + 0.850544 w_k1_MAT_s114
                                 + 1.2411 w_k1_MSFT_s114 + 1.145 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.00467 w_k1_ADM_s114
                                 + 1.20659 w_k1_AXP_s114
                                 + 0.964236 w_k1_DIS_s114
                                 + 0.994118 w_k1_FDX_s114
                                 + 1.15289 w_k1_IBM_s114
                                 + 0.970685 w_k1_INTC_s114
                                 + 1.01009 w_k1_KO_s114 + 1.40946 w_k1_MAT_s114
                                 + 1.09404 w_k1_MSFT_s114
                                 + 1.05175 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.835559 w_k1_ADM_s114
                                 + 0.950242 w_k1_AXP_s114
                                 + 0.922148 w_k1_DIS_s114
                                 + 0.944415 w_k1_FDX_s114
                                 + 1.57438 w_k1_IBM_s114
                                 + 1.03203 w_k1_INTC_s114 + 1.0233 w_k1_KO_s114
                                 + 1.35585 w_k1_MAT_s114
                                 + 0.922987 w_k1_MSFT_s114
                                 + 1.01446 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.31097 w_k1_ADM_s114
                                 + 1.51033 w_k1_AXP_s114
                                 + 1.66498 w_k1_DIS_s114
                                 + 1.45291 w_k1_FDX_s114
                                 + 1.19384 w_k1_IBM_s114
                                 + 1.3242 w_k1_INTC_s114 + 1.31473 w_k1_KO_s114
                                 + 1.30657 w_k1_MAT_s114
                                 + 1.52761 w_k1_MSFT_s114
                                 + 1.19674 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.961692 w_k1_ADM_s114
                                 + 0.754514 w_k1_AXP_s114
                                 + 0.791422 w_k1_DIS_s114
                                 + 0.864463 w_k1_FDX_s114
                                 + 0.835714 w_k1_IBM_s114
                                 + 0.950652 w_k1_INTC_s114
                                 + 0.899285 w_k1_KO_s114
                                 + 0.624934 w_k1_MAT_s114
                                 + 0.987266 w_k1_MSFT_s114
                                 + 1.00559 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.00743 w_k1_ADM_s114
                                 + 0.844435 w_k1_AXP_s114
                                 + 0.954833 w_k1_DIS_s114
                                 + 1.03024 w_k1_FDX_s114
                                 + 0.563799 w_k1_IBM_s114
                                 + 1.05537 w_k1_INTC_s114
                                 + 0.781014 w_k1_KO_s114
                                 + 0.716079 w_k1_MAT_s114
                                 + 1.07925 w_k1_MSFT_s114
                                 + 0.935078 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.796583 w_k1_ADM_s114
                                 + 0.700844 w_k1_AXP_s114
                                 + 0.714223 w_k1_DIS_s114
                                 + 0.727148 w_k1_FDX_s114
                                 + 0.908536 w_k1_IBM_s114
                                 + 0.801162 w_k1_INTC_s114
                                 + 0.782559 w_k1_KO_s114
                                 + 0.571102 w_k1_MAT_s114
                                 + 0.790355 w_k1_MSFT_s114
                                 + 0.8678 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.04574 w_k1_ADM_s114
                                 + 1.11967 w_k1_AXP_s114
                                 + 1.51776 w_k1_DIS_s114
                                 + 1.11119 w_k1_FDX_s114
                                 + 1.45764 w_k1_IBM_s114
                                 + 1.08104 w_k1_INTC_s114
                                 + 1.36421 w_k1_KO_s114 + 1.69137 w_k1_MAT_s114
                                 + 1.20284 w_k1_MSFT_s114
                                 + 1.19271 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.793263 w_k1_ADM_s115
                                 + 0.799546 w_k1_AXP_s115
                                 + 0.606234 w_k1_DIS_s115
                                 + 0.772201 w_k1_FDX_s115
                                 + 0.800867 w_k1_IBM_s115
                                 + 0.8133 w_k1_INTC_s115
                                 + 0.806211 w_k1_KO_s115
                                 + 0.804048 w_k1_MAT_s115
                                 + 0.753603 w_k1_MSFT_s115
                                 + 0.946891 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.845099 w_k1_ADM_s115
                                 + 0.76068 w_k1_AXP_s115
                                 + 1.10379 w_k1_DIS_s115
                                 + 0.932979 w_k1_FDX_s115
                                 + 0.78344 w_k1_IBM_s115
                                 + 1.01945 w_k1_INTC_s115
                                 + 1.07549 w_k1_KO_s115 + 0.75282 w_k1_MAT_s115
                                 + 1.08495 w_k1_MSFT_s115
                                 + 1.06308 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.20981 w_k1_ADM_s115
                                 + 1.00075 w_k1_AXP_s115
                                 + 1.03256 w_k1_DIS_s115 + 1.0232 w_k1_FDX_s115
                                 + 0.990287 w_k1_IBM_s115
                                 + 0.855172 w_k1_INTC_s115
                                 + 1.08794 w_k1_KO_s115
                                 + 0.894979 w_k1_MAT_s115
                                 + 0.995818 w_k1_MSFT_s115
                                 + 1.09072 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.930866 w_k1_ADM_s115
                                 + 1.0212 w_k1_AXP_s115
                                 + 0.764798 w_k1_DIS_s115
                                 + 0.828876 w_k1_FDX_s115
                                 + 1.23315 w_k1_IBM_s115
                                 + 0.870376 w_k1_INTC_s115
                                 + 0.941443 w_k1_KO_s115
                                 + 1.56672 w_k1_MAT_s115
                                 + 0.831235 w_k1_MSFT_s115
                                 + 0.926246 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09485 w_k1_ADM_s115
                                 + 1.26293 w_k1_AXP_s115
                                 + 1.27877 w_k1_DIS_s115
                                 + 1.26481 w_k1_FDX_s115
                                 + 0.766449 w_k1_IBM_s115
                                 + 1.04647 w_k1_INTC_s115
                                 + 1.08646 w_k1_KO_s115 + 1.07785 w_k1_MAT_s115
                                 + 1.18148 w_k1_MSFT_s115
                                 + 1.25522 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.11377 w_k1_ADM_s115
                                 + 1.07483 w_k1_AXP_s115
                                 + 1.21834 w_k1_DIS_s115
                                 + 0.999718 w_k1_FDX_s115
                                 + 1.8967 w_k1_IBM_s115
                                 + 0.990678 w_k1_INTC_s115
                                 + 1.52604 w_k1_KO_s115 + 1.73134 w_k1_MAT_s115
                                 + 1.09497 w_k1_MSFT_s115
                                 + 1.28431 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.1376 w_k1_ADM_s115
                                 + 1.29918 w_k1_AXP_s115
                                 + 1.35843 w_k1_DIS_s115
                                 + 1.14745 w_k1_FDX_s115
                                 + 1.42554 w_k1_IBM_s115
                                 + 1.01899 w_k1_INTC_s115
                                 + 1.24617 w_k1_KO_s115 + 1.25521 w_k1_MAT_s115
                                 + 1.13818 w_k1_MSFT_s115
                                 + 1.09035 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.14342 w_k1_ADM_s115
                                 + 0.910582 w_k1_AXP_s115
                                 + 0.850481 w_k1_DIS_s115
                                 + 1.03912 w_k1_FDX_s115
                                 + 0.682341 w_k1_IBM_s115
                                 + 0.872457 w_k1_INTC_s115
                                 + 0.925149 w_k1_KO_s115
                                 + 0.775823 w_k1_MAT_s115
                                 + 0.971772 w_k1_MSFT_s115
                                 + 1.01947 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.15242 w_k1_ADM_s116
                                 + 1.08524 w_k1_AXP_s116
                                 + 1.02427 w_k1_DIS_s116
                                 + 0.936206 w_k1_FDX_s116
                                 + 1.47462 w_k1_IBM_s116
                                 + 0.8102 w_k1_INTC_s116 + 1.30236 w_k1_KO_s116
                                 + 1.35452 w_k1_MAT_s116
                                 + 0.865311 w_k1_MSFT_s116
                                 + 1.01201 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.87668 w_k1_ADM_s116
                                 + 1.61668 w_k1_AXP_s116
                                 + 1.51032 w_k1_DIS_s116
                                 + 1.29983 w_k1_FDX_s116
                                 + 1.48918 w_k1_IBM_s116
                                 + 1.07331 w_k1_INTC_s116
                                 + 1.58934 w_k1_KO_s116 + 1.38628 w_k1_MAT_s116
                                 + 1.23621 w_k1_MSFT_s116
                                 + 1.17424 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.811919 w_k1_ADM_s116
                                 + 0.83259 w_k1_AXP_s116
                                 + 0.994743 w_k1_DIS_s116
                                 + 0.988164 w_k1_FDX_s116
                                 + 0.986111 w_k1_IBM_s116
                                 + 0.966165 w_k1_INTC_s116
                                 + 1.10159 w_k1_KO_s116 + 1.15425 w_k1_MAT_s116
                                 + 1.0198 w_k1_MSFT_s116
                                 + 0.959409 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.43324 w_k1_ADM_s116
                                 + 1.15722 w_k1_AXP_s116
                                 + 1.13861 w_k1_DIS_s116
                                 + 1.20998 w_k1_FDX_s116
                                 + 0.804048 w_k1_IBM_s116
                                 + 1.09623 w_k1_INTC_s116
                                 + 1.02936 w_k1_KO_s116 + 1.02976 w_k1_MAT_s116
                                 + 1.13476 w_k1_MSFT_s116
                                 + 1.02005 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.932389 w_k1_ADM_s116
                                 + 0.723964 w_k1_AXP_s116
                                 + 0.653241 w_k1_DIS_s116
                                 + 0.743697 w_k1_FDX_s116
                                 + 0.99784 w_k1_IBM_s116
                                 + 0.840159 w_k1_INTC_s116
                                 + 1.09353 w_k1_KO_s116
                                 + 0.995016 w_k1_MAT_s116
                                 + 0.816783 w_k1_MSFT_s116
                                 + 0.770668 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.943788 w_k1_ADM_s116
                                 + 1.05848 w_k1_AXP_s116
                                 + 1.43643 w_k1_DIS_s116
                                 + 1.13994 w_k1_FDX_s116
                                 + 1.08884 w_k1_IBM_s116
                                 + 1.06806 w_k1_INTC_s116
                                 + 1.32838 w_k1_KO_s116 + 1.01692 w_k1_MAT_s116
                                 + 1.08938 w_k1_MSFT_s116
                                 + 1.10798 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.888429 w_k1_ADM_s116
                                 + 0.776655 w_k1_AXP_s116
                                 + 0.717096 w_k1_DIS_s116
                                 + 0.876282 w_k1_FDX_s116
                                 + 0.682464 w_k1_IBM_s116
                                 + 1.04984 w_k1_INTC_s116
                                 + 0.878376 w_k1_KO_s116
                                 + 0.719685 w_k1_MAT_s116
                                 + 0.902679 w_k1_MSFT_s116
                                 + 1.00284 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.3722 w_k1_ADM_s116
                                 + 1.7438 w_k1_AXP_s116 + 1.25688 w_k1_DIS_s116
                                 + 1.31653 w_k1_FDX_s116
                                 + 1.03575 w_k1_IBM_s116
                                 + 1.23879 w_k1_INTC_s116
                                 + 1.06821 w_k1_KO_s116 + 1.28374 w_k1_MAT_s116
                                 + 1.09339 w_k1_MSFT_s116
                                 + 0.957968 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.02359 w_k1_ADM_s117
                                 + 0.98099 w_k1_AXP_s117
                                 + 0.976398 w_k1_DIS_s117
                                 + 1.21573 w_k1_FDX_s117
                                 + 0.747298 w_k1_IBM_s117
                                 + 1.29099 w_k1_INTC_s117
                                 + 0.937323 w_k1_KO_s117
                                 + 0.700563 w_k1_MAT_s117
                                 + 1.14581 w_k1_MSFT_s117
                                 + 1.04793 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.837631 w_k1_ADM_s117
                                 + 0.784205 w_k1_AXP_s117
                                 + 0.81256 w_k1_DIS_s117
                                 + 0.810032 w_k1_FDX_s117
                                 + 1.12529 w_k1_IBM_s117
                                 + 0.977781 w_k1_INTC_s117
                                 + 0.988722 w_k1_KO_s117
                                 + 1.04953 w_k1_MAT_s117
                                 + 0.987668 w_k1_MSFT_s117
                                 + 0.937305 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.781411 w_k1_ADM_s117
                                 + 0.936391 w_k1_AXP_s117
                                 + 1.1908 w_k1_DIS_s117 + 1.02352 w_k1_FDX_s117
                                 + 1.30696 w_k1_IBM_s117
                                 + 1.15461 w_k1_INTC_s117
                                 + 1.20192 w_k1_KO_s117 + 1.30317 w_k1_MAT_s117
                                 + 1.18742 w_k1_MSFT_s117
                                 + 1.00166 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.939938 w_k1_ADM_s117
                                 + 1.33618 w_k1_AXP_s117
                                 + 1.01457 w_k1_DIS_s117
                                 + 1.16262 w_k1_FDX_s117 + 1.4287 w_k1_IBM_s117
                                 + 1.15641 w_k1_INTC_s117
                                 + 1.07202 w_k1_KO_s117 + 1.61638 w_k1_MAT_s117
                                 + 1.08343 w_k1_MSFT_s117
                                 + 1.06229 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.09659 w_k1_ADM_s117
                                 + 0.883596 w_k1_AXP_s117
                                 + 1.37295 w_k1_DIS_s117
                                 + 1.19548 w_k1_FDX_s117
                                 + 1.19318 w_k1_IBM_s117
                                 + 1.20657 w_k1_INTC_s117
                                 + 1.19637 w_k1_KO_s117 + 1.25903 w_k1_MAT_s117
                                 + 1.27676 w_k1_MSFT_s117
                                 + 1.18547 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.96828 w_k1_ADM_s117
                                 + 0.899758 w_k1_AXP_s117
                                 + 0.906623 w_k1_DIS_s117
                                 + 0.957194 w_k1_FDX_s117
                                 + 1.24906 w_k1_IBM_s117
                                 + 0.985442 w_k1_INTC_s117
                                 + 1.12411 w_k1_KO_s117
                                 + 0.962994 w_k1_MAT_s117
                                 + 0.968678 w_k1_MSFT_s117
                                 + 1.03167 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.14265 w_k1_ADM_s117
                                 + 1.33622 w_k1_AXP_s117
                                 + 1.50048 w_k1_DIS_s117
                                 + 1.48026 w_k1_FDX_s117
                                 + 1.29334 w_k1_IBM_s117
                                 + 1.46121 w_k1_INTC_s117 + 1.4571 w_k1_KO_s117
                                 + 1.53363 w_k1_MAT_s117
                                 + 1.33509 w_k1_MSFT_s117
                                 + 1.25206 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.702913 w_k1_ADM_s117
                                 + 0.751267 w_k1_AXP_s117
                                 + 0.705027 w_k1_DIS_s117
                                 + 0.82444 w_k1_FDX_s117
                                 + 0.749317 w_k1_IBM_s117
                                 + 0.850038 w_k1_INTC_s117
                                 + 0.771868 w_k1_KO_s117
                                 + 0.681503 w_k1_MAT_s117
                                 + 0.793347 w_k1_MSFT_s117
                                 + 0.81745 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00666565310925252 w_k1_ADM_s000
                                 + 0.0134327769857769 w_k1_ADM_s001
                                 + 0.0043106932607106 w_k1_ADM_s002
                                 + 0.00950828782595113 w_k1_ADM_s003
                                 + 0.0030722087131537 w_k1_ADM_s004
                                 + 0.0034048155065964 w_k1_ADM_s005
                                 + 0.00389395633907886 w_k1_ADM_s006
                                 + 0.010095960234942 w_k1_ADM_s007
                                 + 0.0164106354263324 w_k1_ADM_s010
                                 + 0.0352554083804388 w_k1_ADM_s011
                                 + 0.0178634987851562 w_k1_ADM_s012
                                 + 0.0335309867226078 w_k1_ADM_s013
                                 + 0.00963153853790533 w_k1_ADM_s014
                                 + 0.0131701649266484 w_k1_ADM_s015
                                 + 0.0296945049001469 w_k1_ADM_s016
                                 + 0.0287505982238027 w_k1_ADM_s017
                                 + 0.0273474845240048 w_k1_ADM_s100
                                 + 0.0757629736203584 w_k1_ADM_s101
                                 + 0.104923848217163 w_k1_ADM_s102
                                 + 0.0314534093598187 w_k1_ADM_s103
                                 + 0.0502551764095498 w_k1_ADM_s104
                                 + 0.0858312114044008 w_k1_ADM_s105
                                 + 0.0812292519458189 w_k1_ADM_s106
                                 + 0.0377258106863621 w_k1_ADM_s107
                                 + 0.0465335730334284 w_k1_ADM_s110
                                 + 0.0215190110608664 w_k1_ADM_s111
                                 + 0.0250694425135844 w_k1_ADM_s112
                                 + 0.0424855349424491 w_k1_ADM_s113
                                 + 0.0597623325364754 w_k1_ADM_s114
                                 + 0.0269737761456652 w_k1_ADM_s115
                                 + 0.0258804466057092 w_k1_ADM_s116
                                 + 0.0257070389204938 w_k1_ADM_s117
                                 + 0.00705016563449773 w_k1_AXP_s000
                                 + 0.0127769632287735 w_k1_AXP_s001
                                 + 0.00357788377419475 w_k1_AXP_s002
                                 + 0.0109810612637467 w_k1_AXP_s003
                                 + 0.00348254935922211 w_k1_AXP_s004
                                 + 0.00331608792905123 w_k1_AXP_s005
                                 + 0.00422976434694188 w_k1_AXP_s006
                                 + 0.0110893956543575 w_k1_AXP_s007
                                 + 0.0153974623807361 w_k1_AXP_s010
                                 + 0.0306952057447101 w_k1_AXP_s011
                                 + 0.0182452605767838 w_k1_AXP_s012
                                 + 0.0404202353256444 w_k1_AXP_s013
                                 + 0.00988877284354386 w_k1_AXP_s014
                                 + 0.0113309778046649 w_k1_AXP_s015
                                 + 0.0305497130658958 w_k1_AXP_s016
                                 + 0.0316158226704935 w_k1_AXP_s017
                                 + 0.0301677542996906 w_k1_AXP_s100
                                 + 0.0724188802346398 w_k1_AXP_s101
                                 + 0.102908190047259 w_k1_AXP_s102
                                 + 0.0375973173712956 w_k1_AXP_s103
                                 + 0.0575347810922663 w_k1_AXP_s104
                                 + 0.0845971723112686 w_k1_AXP_s105
                                 + 0.0929732986788646 w_k1_AXP_s106
                                 + 0.0348501774163318 w_k1_AXP_s107
                                 + 0.047044508664124 w_k1_AXP_s110
                                 + 0.0201561447968834 w_k1_AXP_s111
                                 + 0.0262836691092912 w_k1_AXP_s112
                                 + 0.0406529757740402 w_k1_AXP_s113
                                 + 0.0568450905553027 w_k1_AXP_s114
                                 + 0.0264358846587881 w_k1_AXP_s115
                                 + 0.0238824280945345 w_k1_AXP_s116
                                 + 0.0275603200130203 w_k1_AXP_s117
                                 + 0.00678894693789365 w_k1_DIS_s000
                                 + 0.0127650533890484 w_k1_DIS_s001
                                 + 0.00401357393040319 w_k1_DIS_s002
                                 + 0.0105734006231646 w_k1_DIS_s003
                                 + 0.00324273553044595 w_k1_DIS_s004
                                 + 0.00335212243359732 w_k1_DIS_s005
                                 + 0.00449120500686295 w_k1_DIS_s006
                                 + 0.0117027140964974 w_k1_DIS_s007
                                 + 0.0179212848492473 w_k1_DIS_s010
                                 + 0.0296295925094079 w_k1_DIS_s011
                                 + 0.0199622727598178 w_k1_DIS_s012
                                 + 0.0379934046956749 w_k1_DIS_s013
                                 + 0.00971692210191118 w_k1_DIS_s014
                                 + 0.0115064971192664 w_k1_DIS_s015
                                 + 0.0311170436060621 w_k1_DIS_s016
                                 + 0.0334105855665259 w_k1_DIS_s017
                                 + 0.0263456738625707 w_k1_DIS_s100
                                 + 0.0804477516111799 w_k1_DIS_s101
                                 + 0.0954595916934649 w_k1_DIS_s102
                                 + 0.0347206545289188 w_k1_DIS_s103
                                 + 0.0570098466073806 w_k1_DIS_s104
                                 + 0.07803006736479 w_k1_DIS_s105
                                 + 0.0914225110194095 w_k1_DIS_s106
                                 + 0.0312494768938781 w_k1_DIS_s107
                                 + 0.0464019606974697 w_k1_DIS_s110
                                 + 0.0224345595525155 w_k1_DIS_s111
                                 + 0.0274450791614157 w_k1_DIS_s112
                                 + 0.0435062095125199 w_k1_DIS_s113
                                 + 0.0627925640376418 w_k1_DIS_s114
                                 + 0.0261232506335405 w_k1_DIS_s115
                                 + 0.0237351132541872 w_k1_DIS_s116
                                 + 0.0282288290360321 w_k1_DIS_s117
                                 + 0.00647585133979116 w_k1_FDX_s000
                                 + 0.0124450502450828 w_k1_FDX_s001
                                 + 0.00399811243335968 w_k1_FDX_s002
                                 + 0.00964831802041113 w_k1_FDX_s003
                                 + 0.00331841367392807 w_k1_FDX_s004
                                 + 0.00333632838508418 w_k1_FDX_s005
                                 + 0.00446221196949173 w_k1_FDX_s006
                                 + 0.0111646834691053 w_k1_FDX_s007
                                 + 0.0160093140549689 w_k1_FDX_s010
                                 + 0.0294834288173857 w_k1_FDX_s011
                                 + 0.0193699234608274 w_k1_FDX_s012
                                 + 0.0367323943242339 w_k1_FDX_s013
                                 + 0.00917161693326519 w_k1_FDX_s014
                                 + 0.0120867313014787 w_k1_FDX_s015
                                 + 0.0298968180735781 w_k1_FDX_s016
                                 + 0.0319377419826646 w_k1_FDX_s017
                                 + 0.0285124972672402 w_k1_FDX_s100
                                 + 0.0819396846188597 w_k1_FDX_s101
                                 + 0.100280548947291 w_k1_FDX_s102
                                 + 0.0335984819744286 w_k1_FDX_s103
                                 + 0.0575015969237521 w_k1_FDX_s104
                                 + 0.0794290250286517 w_k1_FDX_s105
                                 + 0.0912731990511085 w_k1_FDX_s106
                                 + 0.0327258847875559 w_k1_FDX_s107
                                 + 0.0447793355628247 w_k1_FDX_s110
                                 + 0.0202590497721624 w_k1_FDX_s111
                                 + 0.0262670159645669 w_k1_FDX_s112
                                 + 0.0421048655929836 w_k1_FDX_s113
                                 + 0.0597548295258971 w_k1_FDX_s114
                                 + 0.025952726914995 w_k1_FDX_s115
                                 + 0.0241748115113903 w_k1_FDX_s116
                                 + 0.0299384217025265 w_k1_FDX_s117
                                 + 0.00638451747814211 w_k1_IBM_s000
                                 + 0.0116061688483934 w_k1_IBM_s001
                                 + 0.00350599577683255 w_k1_IBM_s002
                                 + 0.00988609003775096 w_k1_IBM_s003
                                 + 0.00326947885033004 w_k1_IBM_s004
                                 + 0.00319461704382762 w_k1_IBM_s005
                                 + 0.00420796007413658 w_k1_IBM_s006
                                 + 0.0120961895236472 w_k1_IBM_s007
                                 + 0.0166974289562959 w_k1_IBM_s010
                                 + 0.030384853926845 w_k1_IBM_s011
                                 + 0.0197597108666074 w_k1_IBM_s012
                                 + 0.0384375371317974 w_k1_IBM_s013
                                 + 0.00955360917351203 w_k1_IBM_s014
                                 + 0.0112242970649524 w_k1_IBM_s015
                                 + 0.0328214754700179 w_k1_IBM_s016
                                 + 0.0326748137428947 w_k1_IBM_s017
                                 + 0.0274226220044047 w_k1_IBM_s100
                                 + 0.0813606483297377 w_k1_IBM_s101
                                 + 0.10070917142633 w_k1_IBM_s102
                                 + 0.0346502338415529 w_k1_IBM_s103
                                 + 0.0563851792657725 w_k1_IBM_s104
                                 + 0.0806601643017794 w_k1_IBM_s105
                                 + 0.0940403451440579 w_k1_IBM_s106
                                 + 0.0328321378892698 w_k1_IBM_s107
                                 + 0.0470614051499108 w_k1_IBM_s110
                                 + 0.0219161854578421 w_k1_IBM_s111
                                 + 0.0272539330035744 w_k1_IBM_s112
                                 + 0.0418882779771879 w_k1_IBM_s113
                                 + 0.0600472493977013 w_k1_IBM_s114
                                 + 0.0259223134412574 w_k1_IBM_s115
                                 + 0.0236252443380982 w_k1_IBM_s116
                                 + 0.0303331663354328 w_k1_IBM_s117
                                 + 0.00749372922874998 w_k1_INTC_s000
                                 + 0.0110746742175607 w_k1_INTC_s001
                                 + 0.00374994535679521 w_k1_INTC_s002
                                 + 0.00941191212336982 w_k1_INTC_s003
                                 + 0.00289273434583138 w_k1_INTC_s004
                                 + 0.00341892170379325 w_k1_INTC_s005
                                 + 0.00507538653017849 w_k1_INTC_s006
                                 + 0.0119593431130666 w_k1_INTC_s007
                                 + 0.0150670029311752 w_k1_INTC_s010
                                 + 0.0309957961495471 w_k1_INTC_s011
                                 + 0.0186374831675201 w_k1_INTC_s012
                                 + 0.0399611152864107 w_k1_INTC_s013
                                 + 0.011358043064603 w_k1_INTC_s014
                                 + 0.0106190281784708 w_k1_INTC_s015
                                 + 0.0305557962340704 w_k1_INTC_s016
                                 + 0.0349564325180163 w_k1_INTC_s017
                                 + 0.0289439042968895 w_k1_INTC_s100
                                 + 0.080781630893227 w_k1_INTC_s101
                                 + 0.104360531369952 w_k1_INTC_s102
                                 + 0.0344207885965407 w_k1_INTC_s103
                                 + 0.0544185116962658 w_k1_INTC_s104
                                 + 0.0742700132943276 w_k1_INTC_s105
                                 + 0.100564389730954 w_k1_INTC_s106
                                 + 0.0340659730514181 w_k1_INTC_s107
                                 + 0.037747978362455 w_k1_INTC_s110
                                 + 0.019548461747717 w_k1_INTC_s111
                                 + 0.0257315879865497 w_k1_INTC_s112
                                 + 0.0411010032039189 w_k1_INTC_s113
                                 + 0.0605580604339796 w_k1_INTC_s114
                                 + 0.0234984141415723 w_k1_INTC_s115
                                 + 0.0239869179385669 w_k1_INTC_s116
                                 + 0.0313307823784836 w_k1_INTC_s117
                                 + 0.00617183440107315 w_k1_KO_s000
                                 + 0.0124005329400051 w_k1_KO_s001
                                 + 0.00435170562842456 w_k1_KO_s002
                                 + 0.00996520607094056 w_k1_KO_s003
                                 + 0.0031795851855855 w_k1_KO_s004
                                 + 0.00273577947800992 w_k1_KO_s005
                                 + 0.00429983107562996 w_k1_KO_s006
                                 + 0.0111994410425793 w_k1_KO_s007
                                 + 0.015459220830005 w_k1_KO_s010
                                 + 0.0311311471145502 w_k1_KO_s011
                                 + 0.0204831972243857 w_k1_KO_s012
                                 + 0.0334728429367345 w_k1_KO_s013
                                 + 0.00927743202791594 w_k1_KO_s014
                                 + 0.0118475608523398 w_k1_KO_s015
                                 + 0.0270466946919303 w_k1_KO_s016
                                 + 0.0313869798337502 w_k1_KO_s017
                                 + 0.0277095535220442 w_k1_KO_s100
                                 + 0.0807167418940837 w_k1_KO_s101
                                 + 0.105560088907403 w_k1_KO_s102
                                 + 0.033805337061063 w_k1_KO_s103
                                 + 0.0551880175329144 w_k1_KO_s104
                                 + 0.0757521729308224 w_k1_KO_s105
                                 + 0.0822025855669976 w_k1_KO_s106
                                 + 0.0318277537707719 w_k1_KO_s107
                                 + 0.0444059399480781 w_k1_KO_s110
                                 + 0.0220027974872682 w_k1_KO_s111
                                 + 0.0274510409225847 w_k1_KO_s112
                                 + 0.0463560983716809 w_k1_KO_s113
                                 + 0.0587155506545162 w_k1_KO_s114
                                 + 0.0270034964105503 w_k1_KO_s115
                                 + 0.0264644202494996 w_k1_KO_s116
                                 + 0.029397071181706 w_k1_KO_s117
                                 + 0.00719899182844136 w_k1_MAT_s000
                                 + 0.0133442054455986 w_k1_MAT_s001
                                 + 0.00373896635482937 w_k1_MAT_s002
                                 + 0.00879787014263506 w_k1_MAT_s003
                                 + 0.00434527229464467 w_k1_MAT_s004
                                 + 0.0031160218410129 w_k1_MAT_s005
                                 + 0.00433364234364605 w_k1_MAT_s006
                                 + 0.0153971725922541 w_k1_MAT_s007
                                 + 0.018098736613719 w_k1_MAT_s010
                                 + 0.0352577006123042 w_k1_MAT_s011
                                 + 0.0215605243834848 w_k1_MAT_s012
                                 + 0.0330436698613429 w_k1_MAT_s013
                                 + 0.00840811237920196 w_k1_MAT_s014
                                 + 0.0135464039515376 w_k1_MAT_s015
                                 + 0.0332070198680904 w_k1_MAT_s016
                                 + 0.0352442693637156 w_k1_MAT_s017
                                 + 0.0326381270418685 w_k1_MAT_s100
                                 + 0.0825767415298839 w_k1_MAT_s101
                                 + 0.0934703647689262 w_k1_MAT_s102
                                 + 0.0403549833028002 w_k1_MAT_s103
                                 + 0.0650566391807723 w_k1_MAT_s104
                                 + 0.0800224124843748 w_k1_MAT_s105
                                 + 0.0989076223732025 w_k1_MAT_s106
                                 + 0.0284678416509538 w_k1_MAT_s107
                                 + 0.0451895452329622 w_k1_MAT_s110
                                 + 0.0232853770348896 w_k1_MAT_s111
                                 + 0.0351221055181977 w_k1_MAT_s112
                                 + 0.0481653785509935 w_k1_MAT_s113
                                 + 0.0579499605181861 w_k1_MAT_s114
                                 + 0.0265932203781893 w_k1_MAT_s115
                                 + 0.0252505218782605 w_k1_MAT_s116
                                 + 0.0302109733937382 w_k1_MAT_s117
                                 + 0.00706409818258917 w_k1_MSFT_s000
                                 + 0.0120390304496533 w_k1_MSFT_s001
                                 + 0.00347352248319287 w_k1_MSFT_s002
                                 + 0.0103603591598692 w_k1_MSFT_s003
                                 + 0.00324665364783044 w_k1_MSFT_s004
                                 + 0.00331329920288514 w_k1_MSFT_s005
                                 + 0.004881397971318 w_k1_MSFT_s006
                                 + 0.0111584349512706 w_k1_MSFT_s007
                                 + 0.0172395215124638 w_k1_MSFT_s010
                                 + 0.0302156393801079 w_k1_MSFT_s011
                                 + 0.0215253621558814 w_k1_MSFT_s012
                                 + 0.039607089947844 w_k1_MSFT_s013
                                 + 0.00962771272377154 w_k1_MSFT_s014
                                 + 0.0118278723635555 w_k1_MSFT_s015
                                 + 0.029462128754652 w_k1_MSFT_s016
                                 + 0.0326436832808586 w_k1_MSFT_s017
                                 + 0.0285074199481885 w_k1_MSFT_s100
                                 + 0.0791541291685088 w_k1_MSFT_s101
                                 + 0.0953993829399117 w_k1_MSFT_s102
                                 + 0.0326061012932556 w_k1_MSFT_s103
                                 + 0.0575124537128389 w_k1_MSFT_s104
                                 + 0.0771729155790481 w_k1_MSFT_s105
                                 + 0.0963059741838296 w_k1_MSFT_s106
                                 + 0.031327846175499 w_k1_MSFT_s107
                                 + 0.0422124066351206 w_k1_MSFT_s110
                                 + 0.0209678015293876 w_k1_MSFT_s111
                                 + 0.0268326379044119 w_k1_MSFT_s112
                                 + 0.0424407903388 w_k1_MSFT_s113
                                 + 0.0653361640639597 w_k1_MSFT_s114
                                 + 0.0255192193671053 w_k1_MSFT_s115
                                 + 0.0237332527598628 w_k1_MSFT_s116
                                 + 0.0298525890879493 w_k1_MSFT_s117
                                 + 0.00651839490521104 w_k1_PG_s000
                                 + 0.0113936026005383 w_k1_PG_s001
                                 + 0.00419260329076064 w_k1_PG_s002
                                 + 0.00958023451167379 w_k1_PG_s003
                                 + 0.00333490552528749 w_k1_PG_s004
                                 + 0.0031299008860713 w_k1_PG_s005
                                 + 0.00439063383978764 w_k1_PG_s006
                                 + 0.0109709626452063 w_k1_PG_s007
                                 + 0.0158243121574256 w_k1_PG_s010
                                 + 0.031848255295568 w_k1_PG_s011
                                 + 0.0190391810234684 w_k1_PG_s012
                                 + 0.0365476879705772 w_k1_PG_s013
                                 + 0.00823498746996241 w_k1_PG_s014
                                 + 0.0128661304338202 w_k1_PG_s015
                                 + 0.0300958675465545 w_k1_PG_s016
                                 + 0.0338752998372192 w_k1_PG_s017
                                 + 0.0272283228797971 w_k1_PG_s100
                                 + 0.0779333583348431 w_k1_PG_s101
                                 + 0.10416015847917 w_k1_PG_s102
                                 + 0.0327511081808469 w_k1_PG_s103
                                 + 0.0550707678038547 w_k1_PG_s104
                                 + 0.0754597023185586 w_k1_PG_s105
                                 + 0.0860460466861641 w_k1_PG_s106
                                 + 0.0312774604215385 w_k1_PG_s107
                                 + 0.0440731038592484 w_k1_PG_s110
                                 + 0.0210233793346312 w_k1_PG_s111
                                 + 0.0273281598258342 w_k1_PG_s112
                                 + 0.0447519934548792 w_k1_PG_s113
                                 + 0.0623602455676732 w_k1_PG_s114
                                 + 0.027483548765278 w_k1_PG_s115
                                 + 0.0235500702820495 w_k1_PG_s116
                                 + 0.0285621847001004 w_k1_PG_s117 >= 1.18
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
