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
 w_k1_ADM_s0_return_constraint:  1.14054 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.944792 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.22202 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.864689 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.05635 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.05758 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.735944 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.1682 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.752785 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.01615 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.11832 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.19542 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.36317 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.85158 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.18705 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.978398 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.888912 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.2097 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.11809 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.832162 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.01245 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.31025 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.770481 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.03907 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.21373 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.05833 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.909305 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.771582 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.33523 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.802163 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.705141 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.0579 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.636359 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.45466 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.979328 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.789925 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.869019 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.669609 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23818 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.876697 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.2441 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.80895 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.03028 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.936917 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.884896 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.26893 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.756309 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.884825 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.933825 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.21188 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.57675 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.98536 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.0978 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.888361 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.812521 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.53097 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.1365 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.737502 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.0796 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.6155 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.629043 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.10612 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.26826 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.20936 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.932464 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.752872 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.61355 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.739189 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.787939 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.03431 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.564988 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.56042 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.917475 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.18498 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.981238 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.469286 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30327 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.880329 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.32573 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.832223 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.10059 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.820384 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.66461 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.08908 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.790824 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.927832 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.21019 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.53357 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.65085 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.853914 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.896038 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.857286 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.750021 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.38407 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.27865 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.655655 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.03559 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.55196 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.563925 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.07465 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.25422 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.25442 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.822685 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.545602 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.51355 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.625846 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.735545 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.44901 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.698101 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.83416 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.31046 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.982542 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.930004 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.507045 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18144 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.91189 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.21137 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.894683 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.02457 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.98948 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.852321 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.17178 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.790531 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.03283 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.15065 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.15655 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.20891 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.850976 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.11944 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.773807 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.821384 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.07662 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.1899 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.802048 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.968415 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.31973 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.64284 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.1874 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.08709 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.16678 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.974428 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.714453 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.27487 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.843065 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.779296 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.23763 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.755211 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.46091 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.15096 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.955214 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.998106 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.571936 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18622 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.955858 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.20698 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.914618 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.10198 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.60293 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.684129 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.01893 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.766308 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.588139 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.775908 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.15266 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 2.02278 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.34006 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.603147 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.66766 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.956587 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.94464 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.847689 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.605915 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.24349 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.27165 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.935543 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.686069 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.43806 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.0338 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.876142 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.697319 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.90662 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.587416 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.98256 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.16115 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.659314 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.42163 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.820854 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.37118 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.931235 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.834883 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.14577 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.898607 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03237 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.860528 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.03697 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.972931 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.12427 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.19498 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.976409 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.978035 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.32254 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.26147 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.34939 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.0661 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.00698 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.0023 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.950957 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.25275 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.35819 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.937815 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.13076 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.42037 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.759108 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.28028 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.927817 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.22826 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.979932 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.769912 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.21595 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.837369 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.81191 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.22217 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.911186 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.32617 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.12303 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.05322 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.991107 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.686985 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26362 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.93432 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.09635 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.852239 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.01636 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.800501 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.789673 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.08454 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.832668 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.787692 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.03307 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.31387 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.69322 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.041 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.78063 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.23895 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.91474 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.51716 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.05404 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.673003 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.09032 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.29294 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.848457 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.01792 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.34264 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.21076 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.941787 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.704096 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.69377 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.679241 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.774217 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.30071 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.672453 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.42926 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.02498 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.04446 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.921547 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.634754 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.06714 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.896113 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.53882 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.13442 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.10141 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.702281 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.793178 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.11389 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.706572 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.476594 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.567693 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.35684 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.81965 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.31708 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.586828 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.43356 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.902953 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.74797 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.771725 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.508336 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.924702 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.52359 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.962387 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.826939 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.48588 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.20765 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.733407 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.67165 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.76877 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.685034 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.953196 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.18166 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.542561 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.61341 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.707949 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.16748 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.845619 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.685815 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18625 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.898712 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.08428 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.881426 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.06469 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.981626 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.915027 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13873 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.918385 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.11198 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.20982 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.18985 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16189 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.00849 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.0167 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.87792 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.840181 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.16079 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.2311 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.864847 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.927061 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2914 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.757275 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.10155 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06506 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.1771 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.935538 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.763086 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.29182 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.896885 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.791356 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.32079 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.836461 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.42564 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15238 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.970753 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.900771 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.672446 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10344 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.950931 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.21768 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.00121 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06339 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.894508 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.793244 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04248 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.786598 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.919073 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.929979 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.04485 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.18118 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.952621 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.855446 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.918694 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.859948 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.03887 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.997685 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.765808 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.948835 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.05359 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.75768 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.955758 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.09352 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08694 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.935439 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.708529 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.22315 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.834437 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.945795 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.17533 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.829859 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.25323 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00734 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.03132 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.970008 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.73288 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.218000161477968 z_k1_s0000
                                 - 0.228999617789757 z_k1_s0001
                                 - 0.137000373718457 z_k1_s0002
                                 - 0.331000226785311 z_k1_s0003
                                 - 0.235000597144693 z_k1_s0004
                                 - 0.388000508135627 z_k1_s0005
                                 - 0.246000053456482 z_k1_s0006
                                 - 0.220999648653038 z_k1_s0007
                                 - 0.128999558840401 z_k1_s0010
                                 - 0.197999389512946 z_k1_s0011
                                 - 0.260997957853104 z_k1_s0012
                                 - 0.208999196397542 z_k1_s0013
                                 - 0.133999536187722 z_k1_s0014
                                 - 0.523997052881227 z_k1_s0015
                                 - 0.298998072251769 z_k1_s0016
                                 - 0.633997509719038 z_k1_s0017
                                 - 0.256999837369358 z_k1_s0020
                                 - 0.280000213386085 z_k1_s0021
                                 - 0.258000083413114 z_k1_s0022
                                 - 0.163999672155593 z_k1_s0023
                                 - 0.19500043162547 z_k1_s0024
                                 - 0.219999538733154 z_k1_s0025
                                 - 0.21199933137966 z_k1_s0026
                                 - 0.174999737142078 z_k1_s0027
                                 - 0.182999158500934 z_k1_s0030
                                 - 0.32399893880618 z_k1_s0031
                                 - 0.468997546763909 z_k1_s0032
                                 - 0.149999222563476 z_k1_s0033
                                 - 0.133999355675134 z_k1_s0034
                                 - 0.152999234748614 z_k1_s0035
                                 - 0.136999367860273 z_k1_s0036
                                 - 0.430998184815072 z_k1_s0037
                                 - 0.424998596961295 z_k1_s0040
                                 - 0.198998664414674 z_k1_s0041
                                 - 0.226999785292881 z_k1_s0042
                                 - 0.191999695190855 z_k1_s0043
                                 - 0.0879996272299006 z_k1_s0044
                                 - 0.105999598654043 z_k1_s0045
                                 - 0.355998269503504 z_k1_s0046
                                 - 0.154999549997448 z_k1_s0047
                                 - 0.651996301144651 z_k1_s0050
                                 - 0.195999272082574 z_k1_s0051
                                 - 0.31100002329407 z_k1_s0052
                                 - 0.348998516653837 z_k1_s0053
                                 - 0.67099824581469 z_k1_s0054
                                 - 0.178999236108426 z_k1_s0055
                                 - 0.174999465701876 z_k1_s0056
                                 - 0.164999365187961 z_k1_s0057
                                 - 0.0549999156464122 z_k1_s0060
                                 - 0.0839999076417329 z_k1_s0061
                                 - 0.158999309351275 z_k1_s0062
                                 - 0.064999909427563 z_k1_s0063
                                 - 0.152999373262485 z_k1_s0064
                                 - 0.22699958802257 z_k1_s0065
                                 - 0.199999875623016 z_k1_s0066
                                 - 0.0599998623870704 z_k1_s0067
                                 - 0.197999943752777 z_k1_s0070
                                 - 0.592000107583878 z_k1_s0071
                                 - 0.170999988650101 z_k1_s0072
                                 - 0.493998632545621 z_k1_s0073
                                 - 0.233999883889679 z_k1_s0074
                                 - 0.258999842318082 z_k1_s0075
                                 - 0.420998480934863 z_k1_s0076
                                 - 0.360998580706694 z_k1_s0077
                                 - 0.359998060201293 z_k1_s0100
                                 - 0.194999446741035 z_k1_s0101
                                 - 0.164999531857799 z_k1_s0102
                                 - 0.33599803645495 z_k1_s0103
                                 - 0.163999626534779 z_k1_s0104
                                 - 0.377998652009513 z_k1_s0105
                                 - 0.512998498583461 z_k1_s0106
                                 - 0.18499947511329 z_k1_s0107
                                 - 0.179000439886117 z_k1_s0110
                                 - 0.115000172898859 z_k1_s0111
                                 - 0.152999717734804 z_k1_s0112
                                 - 0.154999521515576 z_k1_s0113
                                 - 0.0850000613860874 z_k1_s0114
                                 - 0.125000282803145 z_k1_s0115
                                 - 0.13200014153606 z_k1_s0116
                                 - 0.147000306392488 z_k1_s0117
                                 - 0.701997942626984 z_k1_s0120
                                 - 0.980999977027874 z_k1_s0121
                                 - 0.299999814226395 z_k1_s0122
                                 - 0.196999985604585 z_k1_s0123
                                 - 0.34899986821531 z_k1_s0124
                                 - 0.47099968216346 z_k1_s0125
                                 - 0.477001789654402 z_k1_s0126
                                 - 0.885001153146588 z_k1_s0127
                                 - 0.101000091663287 z_k1_s0130
                                 - 0.159999488714008 z_k1_s0131
                                 - 0.0999999502963001 z_k1_s0132
                                 - 0.241000762238231 z_k1_s0133
                                 - 0.105000040331131 z_k1_s0134
                                 - 0.387999646781618 z_k1_s0135
                                 - 0.116000053367727 z_k1_s0136
                                 - 0.330999607264715 z_k1_s0137
                                 - 0.282999715158037 z_k1_s0140
                                 - 0.329999448219885 z_k1_s0141
                                 - 0.519998740043928 z_k1_s0142
                                 - 0.378999540858384 z_k1_s0143
                                 - 0.665000174380937 z_k1_s0144
                                 - 0.409997913305191 z_k1_s0145
                                 - 0.998997309757796 z_k1_s0146
                                 - 0.203999913862893 z_k1_s0147
                                 - 0.583999236735743 z_k1_s0150
                                 - 0.146000112784117 z_k1_s0151
                                 - 0.338999962434175 z_k1_s0152
                                 - 0.331998942253509 z_k1_s0153
                                 - 0.425000785789655 z_k1_s0154
                                 - 0.680000650063086 z_k1_s0155
                                 - 0.346000982614842 z_k1_s0156
                                 - 0.184000229475926 z_k1_s0157
                                 - 0.998999720747141 z_k1_s0160
                                 - 0.610001261256312 z_k1_s0161
                                 - 0.32199975180296 z_k1_s0162
                                 - 0.245999903523334 z_k1_s0163
                                 - 0.405001341653534 z_k1_s0164
                                 - 0.247999687043954 z_k1_s0165
                                 - 0.28899967096451 z_k1_s0166
                                 - 0.726001009446278 z_k1_s0167
                                 - 0.344998916533472 z_k1_s0170
                                 - 0.134999660034657 z_k1_s0171
                                 - 0.178999658353126 z_k1_s0172
                                 - 0.269998837070348 z_k1_s0173
                                 - 0.16099973588177 z_k1_s0174
                                 - 0.560998469188716 z_k1_s0175
                                 - 0.171999554336775 z_k1_s0176
                                 - 0.591999757830786 z_k1_s0177
                                 - 1.27400177911167 z_k1_s1000
                                 - 1.66199940892361 z_k1_s1001
                                 - 0.477999174827299 z_k1_s1002
                                 - 1.05499643837644 z_k1_s1003
                                 - 0.467999396491721 z_k1_s1004
                                 - 0.894995737012988 z_k1_s1005
                                 - 0.68399928133383 z_k1_s1006
                                 - 0.563999109310758 z_k1_s1007
                                 - 0.402000632446606 z_k1_s1010
                                 - 0.857002990400361 z_k1_s1011
                                 - 0.528001319584642 z_k1_s1012
                                 - 1.07999954459824 z_k1_s1013
                                 - 0.673003767129555 z_k1_s1014
                                 - 0.675002491379804 z_k1_s1015
                                 - 0.596002915388726 z_k1_s1016
                                 - 0.740999255720652 z_k1_s1017
                                 - 1.58699639553879 z_k1_s1020
                                 - 1.5789962973417 z_k1_s1021
                                 - 0.782997607199528 z_k1_s1022
                                 - 0.814000757830361 z_k1_s1023
                                 - 0.905000951449907 z_k1_s1024
                                 - 0.445998456857844 z_k1_s1025
                                 - 0.767994652962877 z_k1_s1026
                                 - 0.504998257689024 z_k1_s1027
                                 - 1.77700176155178 z_k1_s1030
                                 - 1.05799874937985 z_k1_s1031
                                 - 0.537999715252632 z_k1_s1032
                                 - 0.58999913610571 z_k1_s1033
                                 - 2.67900224837239 z_k1_s1034
                                 - 2.35800357309255 z_k1_s1035
                                 - 0.812999598854235 z_k1_s1036
                                 - 2.25099597200521 z_k1_s1037
                                 - 0.60300161175488 z_k1_s1040
                                 - 1.2500045781359 z_k1_s1041
                                 - 0.987003109550518 z_k1_s1042
                                 - 0.774999881634945 z_k1_s1043
                                 - 1.35400362578461 z_k1_s1044
                                 - 0.608001979508525 z_k1_s1045
                                 - 0.606002101207914 z_k1_s1046
                                 - 0.537001595822016 z_k1_s1047
                                 - 0.418999823390446 z_k1_s1050
                                 - 0.401000409808155 z_k1_s1051
                                 - 0.341000317866779 z_k1_s1052
                                 - 0.383000996225864 z_k1_s1053
                                 - 0.300000152706819 z_k1_s1054
                                 - 0.37699914503136 z_k1_s1055
                                 - 0.361001371847441 z_k1_s1056
                                 - 0.488001157123272 z_k1_s1057
                                 - 0.583000214614569 z_k1_s1060
                                 - 0.403998646971602 z_k1_s1061
                                 - 0.535000223291237 z_k1_s1062
                                 - 0.506000548533334 z_k1_s1063
                                 - 0.564998297868666 z_k1_s1064
                                 - 0.671999878526636 z_k1_s1065
                                 - 0.227999958785823 z_k1_s1066
                                 - 0.347000001274903 z_k1_s1067
                                 - 0.571000259332329 z_k1_s1070
                                 - 0.472998321458551 z_k1_s1071
                                 - 0.416998366672852 z_k1_s1072
                                 - 0.433001235181342 z_k1_s1073
                                 - 0.326999880052123 z_k1_s1074
                                 - 0.353999829438102 z_k1_s1075
                                 - 0.291999706611181 z_k1_s1076
                                 - 1.16699920183761 z_k1_s1077
                                 - 0.527002427144454 z_k1_s1100
                                 - 0.150000705663945 z_k1_s1101
                                 - 0.135000451256823 z_k1_s1102
                                 - 0.152000662984591 z_k1_s1103
                                 - 0.221000914053715 z_k1_s1104
                                 - 0.452002304113392 z_k1_s1105
                                 - 0.193000822161936 z_k1_s1106
                                 - 0.468001043474922 z_k1_s1107
                                 - 0.0820002665025027 z_k1_s1110
                                 - 0.14800018126673 z_k1_s1111
                                 - 0.151001015295736 z_k1_s1112
                                 - 0.13300102714181 z_k1_s1113
                                 - 0.165000727414339 z_k1_s1114
                                 - 0.209001255792836 z_k1_s1115
                                 - 0.0770002976598573 z_k1_s1116
                                 - 0.289001760479186 z_k1_s1117
                                 - 0.0690000366422302 z_k1_s1120
                                 - 0.0330000680811623 z_k1_s1121
                                 - 0.155000241471978 z_k1_s1122
                                 - 0.0750000960025305 z_k1_s1123
                                 - 0.0410000826281071 z_k1_s1124
                                 - 0.134000033710927 z_k1_s1125
                                 - 0.0600001414021467 z_k1_s1126
                                 - 0.079000135576064 z_k1_s1127
                                 - 0.155999850219953 z_k1_s1130
                                 - 0.10100000082825 z_k1_s1131
                                 - 0.446999345364987 z_k1_s1132
                                 - 0.198999352871853 z_k1_s1133
                                 - 0.0959998831199845 z_k1_s1134
                                 - 0.157999961543225 z_k1_s1135
                                 - 0.216999391181817 z_k1_s1136
                                 - 0.231999423106788 z_k1_s1137
                                 - 0.202000346445159 z_k1_s1140
                                 - 0.0890004178708021 z_k1_s1141
                                 - 0.117000685696044 z_k1_s1142
                                 - 0.344001176127524 z_k1_s1143
                                 - 0.204001317009128 z_k1_s1144
                                 - 0.155001218316908 z_k1_s1145
                                 - 0.23500156067251 z_k1_s1146
                                 - 0.134000647446023 z_k1_s1147
                                 - 0.180000354414277 z_k1_s1150
                                 - 0.270001205623537 z_k1_s1151
                                 - 0.247000209245387 z_k1_s1152
                                 - 0.0750002600063024 z_k1_s1153
                                 - 0.140001024546795 z_k1_s1154
                                 - 0.108000347448991 z_k1_s1155
                                 - 0.175000606681372 z_k1_s1156
                                 - 0.152999829450651 z_k1_s1157
                                 - 0.201000627725039 z_k1_s1160
                                 - 0.177000690970331 z_k1_s1161
                                 - 0.129000431659462 z_k1_s1162
                                 - 0.300001138441116 z_k1_s1163
                                 - 0.428002344338704 z_k1_s1164
                                 - 0.119000458011667 z_k1_s1165
                                 - 0.211000601372834 z_k1_s1166
                                 - 0.36400174138991 z_k1_s1167
                                 - 0.0860001392565658 z_k1_s1170
                                 - 0.138000062832395 z_k1_s1171
                                 - 0.263000119745797 z_k1_s1172
                                 - 0.0770001351588009 z_k1_s1173
                                 - 0.0740001337928793 z_k1_s1174
                                 - 0.0880001401671803 z_k1_s1175
                                 - 0.154000070117311 z_k1_s1176
                                 - 0.121000055092173 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.24457 w_k1_ADM_s000
                                 + 1.0646 w_k1_AXP_s000 + 1.26067 w_k1_DIS_s000
                                 + 1.12595 w_k1_FDX_s000
                                 + 1.28996 w_k1_IBM_s000
                                 + 1.05437 w_k1_INTC_s000
                                 + 1.04234 w_k1_KO_s000 + 0.81283 w_k1_MAT_s000
                                 + 1.12599 w_k1_MSFT_s000
                                 + 1.05885 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.76282 w_k1_ADM_s000
                                 + 1.52875 w_k1_AXP_s000
                                 + 1.67461 w_k1_DIS_s000
                                 + 1.42918 w_k1_FDX_s000
                                 + 1.16924 w_k1_IBM_s000
                                 + 1.16609 w_k1_INTC_s000
                                 + 1.16235 w_k1_KO_s000 + 1.11147 w_k1_MAT_s000
                                 + 1.23149 w_k1_MSFT_s000
                                 + 1.31183 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.02213 w_k1_ADM_s000
                                 + 1.00521 w_k1_AXP_s000
                                 + 0.964003 w_k1_DIS_s000
                                 + 0.946648 w_k1_FDX_s000
                                 + 1.16256 w_k1_IBM_s000
                                 + 0.797695 w_k1_INTC_s000
                                 + 0.934586 w_k1_KO_s000
                                 + 0.786554 w_k1_MAT_s000
                                 + 0.882243 w_k1_MSFT_s000
                                 + 1.02793 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.988883 w_k1_ADM_s000
                                 + 0.878609 w_k1_AXP_s000
                                 + 0.827658 w_k1_DIS_s000
                                 + 0.954844 w_k1_FDX_s000
                                 + 0.884601 w_k1_IBM_s000
                                 + 0.997848 w_k1_INTC_s000
                                 + 0.861925 w_k1_KO_s000
                                 + 0.526082 w_k1_MAT_s000
                                 + 0.888787 w_k1_MSFT_s000
                                 + 1.03188 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.00902 w_k1_ADM_s000
                                 + 0.985603 w_k1_AXP_s000
                                 + 0.8728 w_k1_DIS_s000
                                 + 0.815289 w_k1_FDX_s000
                                 + 1.43943 w_k1_IBM_s000
                                 + 0.940821 w_k1_INTC_s000
                                 + 0.924415 w_k1_KO_s000
                                 + 1.10278 w_k1_MAT_s000
                                 + 0.878388 w_k1_MSFT_s000
                                 + 0.903625 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.982041 w_k1_ADM_s000
                                 + 0.719867 w_k1_AXP_s000
                                 + 0.593063 w_k1_DIS_s000
                                 + 0.760301 w_k1_FDX_s000
                                 + 0.732865 w_k1_IBM_s000
                                 + 0.711749 w_k1_INTC_s000
                                 + 0.782485 w_k1_KO_s000
                                 + 0.754956 w_k1_MAT_s000
                                 + 0.717557 w_k1_MSFT_s000
                                 + 0.816874 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.46468 w_k1_ADM_s000
                                 + 1.33758 w_k1_AXP_s000
                                 + 1.11076 w_k1_DIS_s000
                                 + 1.08584 w_k1_FDX_s000
                                 + 1.24586 w_k1_IBM_s000
                                 + 0.915931 w_k1_INTC_s000
                                 + 1.01162 w_k1_KO_s000
                                 + 0.964334 w_k1_MAT_s000
                                 + 0.990553 w_k1_MSFT_s000 + 1.061 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.34579 w_k1_ADM_s000
                                 + 1.02583 w_k1_AXP_s000
                                 + 1.04018 w_k1_DIS_s000
                                 + 1.12254 w_k1_FDX_s000
                                 + 0.618075 w_k1_IBM_s000
                                 + 1.12682 w_k1_INTC_s000
                                 + 0.80376 w_k1_KO_s000 + 0.46422 w_k1_MAT_s000
                                 + 1.02277 w_k1_MSFT_s000
                                 + 0.896469 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.725497 w_k1_ADM_s001
                                 + 1.0022 w_k1_AXP_s001
                                 + 0.968498 w_k1_DIS_s001
                                 + 0.899463 w_k1_FDX_s001
                                 + 1.5014 w_k1_IBM_s001
                                 + 0.887778 w_k1_INTC_s001
                                 + 1.15291 w_k1_KO_s001 + 1.3536 w_k1_MAT_s001
                                 + 0.962375 w_k1_MSFT_s001
                                 + 0.98444 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.710501 w_k1_ADM_s001
                                 + 0.662753 w_k1_AXP_s001
                                 + 0.730742 w_k1_DIS_s001
                                 + 0.699113 w_k1_FDX_s001
                                 + 0.886631 w_k1_IBM_s001
                                 + 0.731461 w_k1_INTC_s001
                                 + 0.750622 w_k1_KO_s001
                                 + 0.53907 w_k1_MAT_s001
                                 + 0.687747 w_k1_MSFT_s001
                                 + 0.826709 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.02819 w_k1_ADM_s001
                                 + 1.50267 w_k1_AXP_s001
                                 + 1.35289 w_k1_DIS_s001
                                 + 1.27167 w_k1_FDX_s001 + 1.2132 w_k1_IBM_s001
                                 + 1.21518 w_k1_INTC_s001 + 1.0664 w_k1_KO_s001
                                 + 1.10348 w_k1_MAT_s001
                                 + 1.07912 w_k1_MSFT_s001
                                 + 1.19182 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.06019 w_k1_ADM_s001
                                 + 1.08969 w_k1_AXP_s001
                                 + 1.10618 w_k1_DIS_s001
                                 + 0.980775 w_k1_FDX_s001
                                 + 1.25025 w_k1_IBM_s001
                                 + 1.09006 w_k1_INTC_s001 + 1.1801 w_k1_KO_s001
                                 + 0.981197 w_k1_MAT_s001
                                 + 1.0185 w_k1_MSFT_s001 + 1.02753 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.16454 w_k1_ADM_s001
                                 + 1.83212 w_k1_AXP_s001
                                 + 1.75361 w_k1_DIS_s001
                                 + 1.39165 w_k1_FDX_s001
                                 + 1.61156 w_k1_IBM_s001
                                 + 1.7529 w_k1_INTC_s001 + 1.5086 w_k1_KO_s001
                                 + 1.55947 w_k1_MAT_s001
                                 + 1.51768 w_k1_MSFT_s001 + 1.256 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.03319 w_k1_ADM_s001
                                 + 1.12282 w_k1_AXP_s001
                                 + 1.23944 w_k1_DIS_s001
                                 + 1.17755 w_k1_FDX_s001
                                 + 0.846482 w_k1_IBM_s001
                                 + 1.00089 w_k1_INTC_s001
                                 + 0.967466 w_k1_KO_s001
                                 + 0.863333 w_k1_MAT_s001
                                 + 1.12217 w_k1_MSFT_s001
                                 + 0.942281 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.790429 w_k1_ADM_s001
                                 + 0.705916 w_k1_AXP_s001
                                 + 0.656348 w_k1_DIS_s001
                                 + 0.803932 w_k1_FDX_s001
                                 + 0.653239 w_k1_IBM_s001
                                 + 0.978876 w_k1_INTC_s001
                                 + 0.806245 w_k1_KO_s001
                                 + 0.604785 w_k1_MAT_s001
                                 + 0.927572 w_k1_MSFT_s001
                                 + 0.754625 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.943205 w_k1_ADM_s001
                                 + 0.944403 w_k1_AXP_s001
                                 + 0.858842 w_k1_DIS_s001
                                 + 0.887519 w_k1_FDX_s001
                                 + 0.976354 w_k1_IBM_s001
                                 + 0.920179 w_k1_INTC_s001
                                 + 0.881841 w_k1_KO_s001
                                 + 1.26048 w_k1_MAT_s001
                                 + 0.818762 w_k1_MSFT_s001
                                 + 0.94081 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.14461 w_k1_ADM_s002
                                 + 1.21738 w_k1_AXP_s002
                                 + 1.11573 w_k1_DIS_s002 + 1.0417 w_k1_FDX_s002
                                 + 1.43429 w_k1_IBM_s002
                                 + 1.18637 w_k1_INTC_s002
                                 + 1.39412 w_k1_KO_s002
                                 + 0.943311 w_k1_MAT_s002
                                 + 1.09064 w_k1_MSFT_s002
                                 + 1.12124 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.750963 w_k1_ADM_s002
                                 + 0.810312 w_k1_AXP_s002
                                 + 0.888471 w_k1_DIS_s002
                                 + 0.840442 w_k1_FDX_s002
                                 + 1.27922 w_k1_IBM_s002
                                 + 0.930413 w_k1_INTC_s002
                                 + 1.1113 w_k1_KO_s002 + 1.18258 w_k1_MAT_s002
                                 + 0.937536 w_k1_MSFT_s002
                                 + 1.08436 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.884142 w_k1_ADM_s002
                                 + 0.777697 w_k1_AXP_s002
                                 + 0.754161 w_k1_DIS_s002
                                 + 0.872571 w_k1_FDX_s002
                                 + 0.71529 w_k1_IBM_s002
                                 + 0.818844 w_k1_INTC_s002
                                 + 0.80496 w_k1_KO_s002
                                 + 0.571966 w_k1_MAT_s002
                                 + 0.81082 w_k1_MSFT_s002
                                 + 0.890375 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.32193 w_k1_ADM_s002
                                 + 1.49564 w_k1_AXP_s002
                                 + 1.51408 w_k1_DIS_s002
                                 + 1.31115 w_k1_FDX_s002
                                 + 1.39471 w_k1_IBM_s002 + 1.363 w_k1_INTC_s002
                                 + 1.31958 w_k1_KO_s002 + 1.48393 w_k1_MAT_s002
                                 + 1.29305 w_k1_MSFT_s002
                                 + 1.27602 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.24253 w_k1_ADM_s002
                                 + 1.11992 w_k1_AXP_s002
                                 + 1.42575 w_k1_DIS_s002
                                 + 1.25543 w_k1_FDX_s002
                                 + 0.881176 w_k1_IBM_s002
                                 + 1.04503 w_k1_INTC_s002
                                 + 1.04743 w_k1_KO_s002
                                 + 0.721162 w_k1_MAT_s002
                                 + 1.23391 w_k1_MSFT_s002
                                 + 1.07313 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.06524 w_k1_ADM_s002
                                 + 0.822195 w_k1_AXP_s002
                                 + 1.06894 w_k1_DIS_s002
                                 + 1.08159 w_k1_FDX_s002
                                 + 0.883426 w_k1_IBM_s002
                                 + 0.930239 w_k1_INTC_s002
                                 + 0.972597 w_k1_KO_s002
                                 + 0.624482 w_k1_MAT_s002
                                 + 0.911804 w_k1_MSFT_s002
                                 + 0.947233 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.716321 w_k1_ADM_s002
                                 + 0.561568 w_k1_AXP_s002
                                 + 0.646377 w_k1_DIS_s002
                                 + 0.717304 w_k1_FDX_s002
                                 + 0.863584 w_k1_IBM_s002
                                 + 0.692269 w_k1_INTC_s002
                                 + 0.79974 w_k1_KO_s002
                                 + 0.702671 w_k1_MAT_s002
                                 + 0.766399 w_k1_MSFT_s002
                                 + 0.767033 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.04388 w_k1_ADM_s002
                                 + 0.797427 w_k1_AXP_s002
                                 + 0.853096 w_k1_DIS_s002
                                 + 0.97894 w_k1_FDX_s002
                                 + 0.921046 w_k1_IBM_s002
                                 + 1.0547 w_k1_INTC_s002
                                 + 0.937027 w_k1_KO_s002
                                 + 0.618529 w_k1_MAT_s002
                                 + 1.00444 w_k1_MSFT_s002
                                 + 1.00428 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.777109 w_k1_ADM_s003
                                 + 0.766168 w_k1_AXP_s003
                                 + 0.776211 w_k1_DIS_s003
                                 + 0.864859 w_k1_FDX_s003
                                 + 0.658146 w_k1_IBM_s003
                                 + 1.07189 w_k1_INTC_s003
                                 + 0.881005 w_k1_KO_s003
                                 + 0.653447 w_k1_MAT_s003
                                 + 0.901103 w_k1_MSFT_s003
                                 + 0.815374 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.7313 w_k1_ADM_s003
                                 + 0.917022 w_k1_AXP_s003
                                 + 0.852682 w_k1_DIS_s003
                                 + 0.81417 w_k1_FDX_s003
                                 + 1.06298 w_k1_IBM_s003
                                 + 0.99642 w_k1_INTC_s003
                                 + 0.881367 w_k1_KO_s003
                                 + 1.00903 w_k1_MAT_s003
                                 + 0.809034 w_k1_MSFT_s003
                                 + 0.91226 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.17178 w_k1_ADM_s003
                                 + 1.31229 w_k1_AXP_s003
                                 + 1.15666 w_k1_DIS_s003
                                 + 1.18964 w_k1_FDX_s003
                                 + 1.04442 w_k1_IBM_s003
                                 + 1.17328 w_k1_INTC_s003
                                 + 1.12801 w_k1_KO_s003 + 1.39607 w_k1_MAT_s003
                                 + 1.12921 w_k1_MSFT_s003
                                 + 1.04013 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.32884 w_k1_ADM_s003
                                 + 2.09505 w_k1_AXP_s003
                                 + 1.69749 w_k1_DIS_s003
                                 + 1.41055 w_k1_FDX_s003
                                 + 1.45634 w_k1_IBM_s003
                                 + 1.36592 w_k1_INTC_s003
                                 + 1.49518 w_k1_KO_s003 + 2.30831 w_k1_MAT_s003
                                 + 1.16684 w_k1_MSFT_s003
                                 + 1.20945 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.08924 w_k1_ADM_s003
                                 + 0.965886 w_k1_AXP_s003
                                 + 0.97187 w_k1_DIS_s003
                                 + 1.12754 w_k1_FDX_s003
                                 + 0.638185 w_k1_IBM_s003
                                 + 1.40832 w_k1_INTC_s003
                                 + 0.997514 w_k1_KO_s003
                                 + 0.802255 w_k1_MAT_s003
                                 + 1.01819 w_k1_MSFT_s003
                                 + 1.02835 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.689425 w_k1_ADM_s003
                                 + 0.762741 w_k1_AXP_s003
                                 + 0.634082 w_k1_DIS_s003
                                 + 0.770935 w_k1_FDX_s003
                                 + 0.875454 w_k1_IBM_s003
                                 + 0.865415 w_k1_INTC_s003
                                 + 0.775193 w_k1_KO_s003
                                 + 1.07911 w_k1_MAT_s003
                                 + 0.741173 w_k1_MSFT_s003
                                 + 0.725024 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.916641 w_k1_ADM_s003
                                 + 0.835874 w_k1_AXP_s003
                                 + 0.894137 w_k1_DIS_s003
                                 + 0.983174 w_k1_FDX_s003
                                 + 0.748082 w_k1_IBM_s003
                                 + 1.00957 w_k1_INTC_s003
                                 + 0.808426 w_k1_KO_s003
                                 + 0.430821 w_k1_MAT_s003
                                 + 0.875593 w_k1_MSFT_s003
                                 + 0.830552 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.879123 w_k1_ADM_s003
                                 + 1.08437 w_k1_AXP_s003
                                 + 1.03075 w_k1_DIS_s003
                                 + 0.967019 w_k1_FDX_s003
                                 + 1.29135 w_k1_IBM_s003
                                 + 0.909283 w_k1_INTC_s003
                                 + 1.0636 w_k1_KO_s003 + 1.60871 w_k1_MAT_s003
                                 + 0.985369 w_k1_MSFT_s003
                                 + 0.971332 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.714864 w_k1_ADM_s004
                                 + 0.790403 w_k1_AXP_s004
                                 + 0.777095 w_k1_DIS_s004
                                 + 0.843713 w_k1_FDX_s004
                                 + 0.67981 w_k1_IBM_s004
                                 + 0.863285 w_k1_INTC_s004
                                 + 0.729581 w_k1_KO_s004
                                 + 0.61012 w_k1_MAT_s004
                                 + 0.848802 w_k1_MSFT_s004
                                 + 0.871714 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.915114 w_k1_ADM_s004
                                 + 1.08373 w_k1_AXP_s004 + 1.1451 w_k1_DIS_s004
                                 + 0.988706 w_k1_FDX_s004
                                 + 1.13807 w_k1_IBM_s004
                                 + 0.76537 w_k1_INTC_s004
                                 + 1.11097 w_k1_KO_s004 + 1.69679 w_k1_MAT_s004
                                 + 0.991878 w_k1_MSFT_s004
                                 + 1.14613 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01428 w_k1_ADM_s004
                                 + 1.32338 w_k1_AXP_s004 + 1.0491 w_k1_DIS_s004
                                 + 1.10511 w_k1_FDX_s004
                                 + 0.991583 w_k1_IBM_s004
                                 + 0.97971 w_k1_INTC_s004
                                 + 0.993492 w_k1_KO_s004 + 1.1334 w_k1_MAT_s004
                                 + 1.05173 w_k1_MSFT_s004
                                 + 1.04963 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.998182 w_k1_ADM_s004
                                 + 1.46351 w_k1_AXP_s004 + 1.1554 w_k1_DIS_s004
                                 + 1.29356 w_k1_FDX_s004
                                 + 0.702149 w_k1_IBM_s004
                                 + 1.02121 w_k1_INTC_s004
                                 + 0.929011 w_k1_KO_s004 + 1.1791 w_k1_MAT_s004
                                 + 1.07327 w_k1_MSFT_s004
                                 + 1.04574 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.29242 w_k1_ADM_s004
                                 + 1.80612 w_k1_AXP_s004
                                 + 1.65367 w_k1_DIS_s004
                                 + 1.63349 w_k1_FDX_s004
                                 + 0.995095 w_k1_IBM_s004
                                 + 1.10069 w_k1_INTC_s004 + 1.2438 w_k1_KO_s004
                                 + 1.72798 w_k1_MAT_s004
                                 + 1.31821 w_k1_MSFT_s004
                                 + 1.27345 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.989416 w_k1_ADM_s004
                                 + 1.28727 w_k1_AXP_s004
                                 + 1.55066 w_k1_DIS_s004
                                 + 1.20671 w_k1_FDX_s004
                                 + 1.51778 w_k1_IBM_s004
                                 + 1.17287 w_k1_INTC_s004
                                 + 1.40624 w_k1_KO_s004 + 1.6136 w_k1_MAT_s004
                                 + 1.21718 w_k1_MSFT_s004
                                 + 1.23692 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.01882 w_k1_ADM_s004
                                 + 0.835389 w_k1_AXP_s004
                                 + 0.834303 w_k1_DIS_s004
                                 + 0.807943 w_k1_FDX_s004
                                 + 1.13607 w_k1_IBM_s004
                                 + 0.711261 w_k1_INTC_s004
                                 + 1.09227 w_k1_KO_s004 + 1.48936 w_k1_MAT_s004
                                 + 0.84789 w_k1_MSFT_s004
                                 + 1.05321 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.04021 w_k1_ADM_s004
                                 + 1.40764 w_k1_AXP_s004
                                 + 1.56286 w_k1_DIS_s004
                                 + 1.15917 w_k1_FDX_s004
                                 + 1.06099 w_k1_IBM_s004
                                 + 1.16869 w_k1_INTC_s004
                                 + 1.26056 w_k1_KO_s004 + 1.15695 w_k1_MAT_s004
                                 + 1.22079 w_k1_MSFT_s004
                                 + 1.11824 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.965274 w_k1_ADM_s005
                                 + 0.934161 w_k1_AXP_s005
                                 + 0.994511 w_k1_DIS_s005
                                 + 0.999651 w_k1_FDX_s005
                                 + 1.00586 w_k1_IBM_s005
                                 + 1.14323 w_k1_INTC_s005
                                 + 0.936676 w_k1_KO_s005
                                 + 0.667968 w_k1_MAT_s005
                                 + 1.16078 w_k1_MSFT_s005
                                 + 0.932844 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.67844 w_k1_ADM_s005
                                 + 0.605927 w_k1_AXP_s005
                                 + 0.648415 w_k1_DIS_s005
                                 + 0.729605 w_k1_FDX_s005
                                 + 0.987485 w_k1_IBM_s005
                                 + 0.693095 w_k1_INTC_s005
                                 + 0.804088 w_k1_KO_s005
                                 + 0.648376 w_k1_MAT_s005
                                 + 0.754743 w_k1_MSFT_s005
                                 + 0.88651 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.14011 w_k1_ADM_s005
                                 + 1.57768 w_k1_AXP_s005
                                 + 1.45156 w_k1_DIS_s005
                                 + 1.37134 w_k1_FDX_s005
                                 + 1.16467 w_k1_IBM_s005
                                 + 1.20904 w_k1_INTC_s005
                                 + 1.33097 w_k1_KO_s005 + 1.00793 w_k1_MAT_s005
                                 + 1.40805 w_k1_MSFT_s005
                                 + 1.14075 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.04452 w_k1_ADM_s005
                                 + 1.11856 w_k1_AXP_s005
                                 + 1.01294 w_k1_DIS_s005
                                 + 1.15086 w_k1_FDX_s005
                                 + 0.740157 w_k1_IBM_s005
                                 + 1.05038 w_k1_INTC_s005 + 0.8449 w_k1_KO_s005
                                 + 0.744313 w_k1_MAT_s005
                                 + 1.05473 w_k1_MSFT_s005
                                 + 1.00742 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.940867 w_k1_ADM_s005
                                 + 1.09757 w_k1_AXP_s005
                                 + 0.987222 w_k1_DIS_s005
                                 + 0.981058 w_k1_FDX_s005
                                 + 1.44441 w_k1_IBM_s005
                                 + 0.93433 w_k1_INTC_s005
                                 + 1.34473 w_k1_KO_s005 + 1.28511 w_k1_MAT_s005
                                 + 1.00893 w_k1_MSFT_s005
                                 + 1.07597 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.43649 w_k1_ADM_s005
                                 + 1.54814 w_k1_AXP_s005
                                 + 1.79958 w_k1_DIS_s005
                                 + 1.51837 w_k1_FDX_s005
                                 + 1.00313 w_k1_IBM_s005
                                 + 1.5862 w_k1_INTC_s005 + 1.32304 w_k1_KO_s005
                                 + 0.768104 w_k1_MAT_s005
                                 + 1.48618 w_k1_MSFT_s005
                                 + 1.30791 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.678022 w_k1_ADM_s005
                                 + 0.691936 w_k1_AXP_s005
                                 + 0.527627 w_k1_DIS_s005
                                 + 0.714049 w_k1_FDX_s005
                                 + 0.917152 w_k1_IBM_s005
                                 + 0.693769 w_k1_INTC_s005
                                 + 0.818712 w_k1_KO_s005
                                 + 0.914964 w_k1_MAT_s005
                                 + 0.778709 w_k1_MSFT_s005
                                 + 0.888387 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.810245 w_k1_ADM_s005
                                 + 0.725533 w_k1_AXP_s005
                                 + 0.821764 w_k1_DIS_s005
                                 + 0.899613 w_k1_FDX_s005
                                 + 0.613402 w_k1_IBM_s005
                                 + 1.02961 w_k1_INTC_s005
                                 + 0.797999 w_k1_KO_s005
                                 + 0.494329 w_k1_MAT_s005
                                 + 0.994437 w_k1_MSFT_s005
                                 + 0.84282 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.502775 w_k1_ADM_s006
                                 + 0.697408 w_k1_AXP_s006
                                 + 0.528222 w_k1_DIS_s006
                                 + 0.563065 w_k1_FDX_s006
                                 + 1.10289 w_k1_IBM_s006
                                 + 0.579472 w_k1_INTC_s006
                                 + 0.687491 w_k1_KO_s006
                                 + 1.00784 w_k1_MAT_s006
                                 + 0.584901 w_k1_MSFT_s006
                                 + 0.746495 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.620014 w_k1_ADM_s006
                                 + 0.804834 w_k1_AXP_s006
                                 + 0.735119 w_k1_DIS_s006
                                 + 0.760439 w_k1_FDX_s006
                                 + 1.43921 w_k1_IBM_s006
                                 + 0.859576 w_k1_INTC_s006
                                 + 0.922001 w_k1_KO_s006
                                 + 1.33971 w_k1_MAT_s006
                                 + 0.775266 w_k1_MSFT_s006
                                 + 0.840991 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.557554 w_k1_ADM_s006
                                 + 0.664072 w_k1_AXP_s006
                                 + 0.679786 w_k1_DIS_s006
                                 + 0.813512 w_k1_FDX_s006
                                 + 0.702862 w_k1_IBM_s006
                                 + 0.921806 w_k1_INTC_s006
                                 + 0.729823 w_k1_KO_s006
                                 + 0.821781 w_k1_MAT_s006
                                 + 0.912881 w_k1_MSFT_s006
                                 + 0.89167 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.861364 w_k1_ADM_s006
                                 + 1.08206 w_k1_AXP_s006
                                 + 1.14781 w_k1_DIS_s006
                                 + 0.974227 w_k1_FDX_s006
                                 + 1.65627 w_k1_IBM_s006
                                 + 1.03689 w_k1_INTC_s006
                                 + 1.38459 w_k1_KO_s006 + 2.27745 w_k1_MAT_s006
                                 + 1.19704 w_k1_MSFT_s006
                                 + 1.20485 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.696076 w_k1_ADM_s006
                                 + 0.731912 w_k1_AXP_s006
                                 + 0.89107 w_k1_DIS_s006
                                 + 0.842297 w_k1_FDX_s006
                                 + 1.09232 w_k1_IBM_s006
                                 + 1.0772 w_k1_INTC_s006 + 1.04833 w_k1_KO_s006
                                 + 0.97566 w_k1_MAT_s006
                                 + 1.04237 w_k1_MSFT_s006
                                 + 0.915569 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.961074 w_k1_ADM_s006
                                 + 1.09828 w_k1_AXP_s006
                                 + 1.02984 w_k1_DIS_s006 + 1.0373 w_k1_FDX_s006
                                 + 1.05533 w_k1_IBM_s006
                                 + 1.0472 w_k1_INTC_s006 + 1.15864 w_k1_KO_s006
                                 + 1.10761 w_k1_MAT_s006
                                 + 1.26366 w_k1_MSFT_s006
                                 + 1.15128 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.23557 w_k1_ADM_s006
                                 + 1.20213 w_k1_AXP_s006
                                 + 1.72009 w_k1_DIS_s006
                                 + 1.35003 w_k1_FDX_s006
                                 + 0.947399 w_k1_IBM_s006
                                 + 1.19753 w_k1_INTC_s006
                                 + 1.25388 w_k1_KO_s006 + 1.26015 w_k1_MAT_s006
                                 + 1.67709 w_k1_MSFT_s006
                                 + 1.16554 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.49491 w_k1_ADM_s006
                                 + 0.572274 w_k1_AXP_s006
                                 + 0.526707 w_k1_DIS_s006
                                 + 0.593502 w_k1_FDX_s006
                                 + 0.872948 w_k1_IBM_s006
                                 + 0.700286 w_k1_INTC_s006
                                 + 0.839263 w_k1_KO_s006
                                 + 0.900247 w_k1_MAT_s006
                                 + 0.775281 w_k1_MSFT_s006
                                 + 0.740569 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.34723 w_k1_ADM_s007
                                 + 1.19268 w_k1_AXP_s007
                                 + 1.18299 w_k1_DIS_s007
                                 + 1.08069 w_k1_FDX_s007
                                 + 0.987931 w_k1_IBM_s007
                                 + 0.883683 w_k1_INTC_s007 + 1.152 w_k1_KO_s007
                                 + 1.16699 w_k1_MAT_s007
                                 + 0.854206 w_k1_MSFT_s007
                                 + 1.01214 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.05818 w_k1_ADM_s007
                                 + 0.742372 w_k1_AXP_s007
                                 + 0.723606 w_k1_DIS_s007
                                 + 0.833913 w_k1_FDX_s007
                                 + 0.651107 w_k1_IBM_s007
                                 + 0.929002 w_k1_INTC_s007
                                 + 0.85075 w_k1_KO_s007
                                 + 0.589473 w_k1_MAT_s007
                                 + 0.75875 w_k1_MSFT_s007
                                 + 0.84266 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.971579 w_k1_ADM_s007
                                 + 1.34242 w_k1_AXP_s007
                                 + 1.28162 w_k1_DIS_s007
                                 + 0.987719 w_k1_FDX_s007
                                 + 1.71161 w_k1_IBM_s007
                                 + 1.19083 w_k1_INTC_s007
                                 + 1.58802 w_k1_KO_s007 + 1.98155 w_k1_MAT_s007
                                 + 1.01151 w_k1_MSFT_s007
                                 + 1.05217 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.805711 w_k1_ADM_s007
                                 + 0.754023 w_k1_AXP_s007
                                 + 0.766873 w_k1_DIS_s007
                                 + 0.743158 w_k1_FDX_s007
                                 + 1.04897 w_k1_IBM_s007
                                 + 0.808046 w_k1_INTC_s007
                                 + 0.99133 w_k1_KO_s007 + 1.06597 w_k1_MAT_s007
                                 + 0.740728 w_k1_MSFT_s007
                                 + 0.842257 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.12467 w_k1_ADM_s007
                                 + 1.23675 w_k1_AXP_s007
                                 + 0.930182 w_k1_DIS_s007
                                 + 1.01306 w_k1_FDX_s007
                                 + 0.929655 w_k1_IBM_s007
                                 + 0.670299 w_k1_INTC_s007
                                 + 0.839294 w_k1_KO_s007
                                 + 0.885073 w_k1_MAT_s007
                                 + 0.783029 w_k1_MSFT_s007
                                 + 0.850593 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.17343 w_k1_ADM_s007
                                 + 1.45817 w_k1_AXP_s007 + 1.0557 w_k1_DIS_s007
                                 + 1.09875 w_k1_FDX_s007
                                 + 1.28576 w_k1_IBM_s007
                                 + 0.862222 w_k1_INTC_s007
                                 + 1.04279 w_k1_KO_s007 + 1.17492 w_k1_MAT_s007
                                 + 0.880112 w_k1_MSFT_s007
                                 + 1.00986 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.21437 w_k1_ADM_s007
                                 + 1.04386 w_k1_AXP_s007
                                 + 1.19179 w_k1_DIS_s007
                                 + 1.05352 w_k1_FDX_s007
                                 + 0.857169 w_k1_IBM_s007
                                 + 0.954369 w_k1_INTC_s007
                                 + 1.09534 w_k1_KO_s007 + 1.03182 w_k1_MAT_s007
                                 + 0.924384 w_k1_MSFT_s007
                                 + 0.958642 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.41979 w_k1_ADM_s007
                                 + 1.3223 w_k1_AXP_s007 + 1.53295 w_k1_DIS_s007
                                 + 1.32803 w_k1_FDX_s007
                                 + 1.25667 w_k1_IBM_s007
                                 + 1.14648 w_k1_INTC_s007
                                 + 1.43836 w_k1_KO_s007 + 1.16247 w_k1_MAT_s007
                                 + 1.05301 w_k1_MSFT_s007
                                 + 1.27635 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.853418 w_k1_ADM_s010
                                 + 0.906885 w_k1_AXP_s010
                                 + 0.870401 w_k1_DIS_s010
                                 + 0.839307 w_k1_FDX_s010
                                 + 1.27073 w_k1_IBM_s010
                                 + 0.976305 w_k1_INTC_s010
                                 + 0.960801 w_k1_KO_s010
                                 + 1.19067 w_k1_MAT_s010
                                 + 0.839424 w_k1_MSFT_s010
                                 + 0.97951 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.01504 w_k1_ADM_s010
                                 + 1.11082 w_k1_AXP_s010
                                 + 1.13323 w_k1_DIS_s010
                                 + 0.985024 w_k1_FDX_s010
                                 + 1.59759 w_k1_IBM_s010
                                 + 1.06236 w_k1_INTC_s010
                                 + 1.12803 w_k1_KO_s010 + 1.32586 w_k1_MAT_s010
                                 + 0.868323 w_k1_MSFT_s010
                                 + 1.1357 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.07199 w_k1_ADM_s010
                                 + 0.931767 w_k1_AXP_s010
                                 + 1.10538 w_k1_DIS_s010
                                 + 1.14564 w_k1_FDX_s010
                                 + 0.645811 w_k1_IBM_s010
                                 + 1.18068 w_k1_INTC_s010
                                 + 0.68625 w_k1_KO_s010
                                 + 0.440812 w_k1_MAT_s010
                                 + 0.972281 w_k1_MSFT_s010
                                 + 0.970254 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.36317 w_k1_ADM_s010
                                 + 1.34966 w_k1_AXP_s010
                                 + 1.51919 w_k1_DIS_s010 + 1.39 w_k1_FDX_s010
                                 + 1.06923 w_k1_IBM_s010
                                 + 1.02465 w_k1_INTC_s010
                                 + 1.24247 w_k1_KO_s010 + 1.31311 w_k1_MAT_s010
                                 + 1.30131 w_k1_MSFT_s010
                                 + 1.29953 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.1442 w_k1_ADM_s010
                                 + 0.942317 w_k1_AXP_s010
                                 + 1.51997 w_k1_DIS_s010
                                 + 1.18179 w_k1_FDX_s010
                                 + 1.07136 w_k1_IBM_s010
                                 + 1.16513 w_k1_INTC_s010
                                 + 1.27428 w_k1_KO_s010
                                 + 0.974182 w_k1_MAT_s010
                                 + 1.27069 w_k1_MSFT_s010
                                 + 1.23643 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.915278 w_k1_ADM_s010
                                 + 0.881063 w_k1_AXP_s010
                                 + 1.04028 w_k1_DIS_s010
                                 + 1.01993 w_k1_FDX_s010
                                 + 1.07421 w_k1_IBM_s010
                                 + 0.963589 w_k1_INTC_s010
                                 + 1.05024 w_k1_KO_s010 + 1.21176 w_k1_MAT_s010
                                 + 0.938879 w_k1_MSFT_s010
                                 + 1.03753 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.76712 w_k1_ADM_s010
                                 + 0.692011 w_k1_AXP_s010
                                 + 0.715543 w_k1_DIS_s010
                                 + 0.812224 w_k1_FDX_s010
                                 + 0.713446 w_k1_IBM_s010
                                 + 0.975607 w_k1_INTC_s010
                                 + 0.781139 w_k1_KO_s010
                                 + 0.921401 w_k1_MAT_s010
                                 + 0.905431 w_k1_MSFT_s010
                                 + 0.892964 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.15171 w_k1_ADM_s010
                                 + 1.35956 w_k1_AXP_s010
                                 + 1.17173 w_k1_DIS_s010
                                 + 1.08956 w_k1_FDX_s010 + 1.3691 w_k1_IBM_s010
                                 + 1.00193 w_k1_INTC_s010
                                 + 1.18466 w_k1_KO_s010 + 1.45757 w_k1_MAT_s010
                                 + 0.909422 w_k1_MSFT_s010
                                 + 1.17359 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.602113 w_k1_ADM_s011
                                 + 0.848236 w_k1_AXP_s011
                                 + 0.706014 w_k1_DIS_s011
                                 + 0.794331 w_k1_FDX_s011
                                 + 0.778562 w_k1_IBM_s011
                                 + 0.806254 w_k1_INTC_s011
                                 + 0.725434 w_k1_KO_s011
                                 + 0.688896 w_k1_MAT_s011
                                 + 0.764614 w_k1_MSFT_s011
                                 + 0.852109 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.16008 w_k1_ADM_s011
                                 + 1.67985 w_k1_AXP_s011 + 1.2277 w_k1_DIS_s011
                                 + 1.41111 w_k1_FDX_s011
                                 + 0.937057 w_k1_IBM_s011
                                 + 1.04978 w_k1_INTC_s011
                                 + 0.959706 w_k1_KO_s011
                                 + 1.29188 w_k1_MAT_s011
                                 + 1.11633 w_k1_MSFT_s011
                                 + 1.17366 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.94781 w_k1_ADM_s011
                                 + 0.981764 w_k1_AXP_s011
                                 + 0.929722 w_k1_DIS_s011
                                 + 0.93204 w_k1_FDX_s011
                                 + 1.26951 w_k1_IBM_s011
                                 + 0.856521 w_k1_INTC_s011
                                 + 1.01672 w_k1_KO_s011 + 1.00069 w_k1_MAT_s011
                                 + 0.833008 w_k1_MSFT_s011
                                 + 0.930157 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.31402 w_k1_ADM_s011
                                 + 1.75042 w_k1_AXP_s011
                                 + 1.33708 w_k1_DIS_s011
                                 + 1.32155 w_k1_FDX_s011
                                 + 1.41824 w_k1_IBM_s011
                                 + 1.13953 w_k1_INTC_s011
                                 + 1.04759 w_k1_KO_s011 + 1.17065 w_k1_MAT_s011
                                 + 1.18176 w_k1_MSFT_s011
                                 + 1.09436 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.02508 w_k1_ADM_s011
                                 + 1.06072 w_k1_AXP_s011
                                 + 1.29988 w_k1_DIS_s011
                                 + 1.04077 w_k1_FDX_s011
                                 + 1.22953 w_k1_IBM_s011
                                 + 1.21584 w_k1_INTC_s011
                                 + 1.16617 w_k1_KO_s011
                                 + 0.818569 w_k1_MAT_s011
                                 + 1.29055 w_k1_MSFT_s011
                                 + 1.21231 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.801914 w_k1_ADM_s011
                                 + 0.784522 w_k1_AXP_s011
                                 + 0.844723 w_k1_DIS_s011
                                 + 0.851236 w_k1_FDX_s011
                                 + 0.954774 w_k1_IBM_s011
                                 + 0.839847 w_k1_INTC_s011
                                 + 0.801577 w_k1_KO_s011
                                 + 0.633011 w_k1_MAT_s011
                                 + 0.809749 w_k1_MSFT_s011
                                 + 0.877609 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.975508 w_k1_ADM_s011
                                 + 1.15514 w_k1_AXP_s011
                                 + 1.06779 w_k1_DIS_s011
                                 + 1.03323 w_k1_FDX_s011
                                 + 1.02553 w_k1_IBM_s011
                                 + 1.03967 w_k1_INTC_s011
                                 + 0.919172 w_k1_KO_s011
                                 + 0.988588 w_k1_MAT_s011
                                 + 0.995237 w_k1_MSFT_s011
                                 + 1.10131 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.871689 w_k1_ADM_s011
                                 + 1.12093 w_k1_AXP_s011
                                 + 0.876693 w_k1_DIS_s011
                                 + 1.0221 w_k1_FDX_s011
                                 + 0.689288 w_k1_IBM_s011
                                 + 0.89849 w_k1_INTC_s011
                                 + 0.727517 w_k1_KO_s011
                                 + 0.716851 w_k1_MAT_s011
                                 + 0.843867 w_k1_MSFT_s011
                                 + 0.89329 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.01307 w_k1_ADM_s012
                                 + 1.00753 w_k1_AXP_s012
                                 + 0.926728 w_k1_DIS_s012
                                 + 0.880803 w_k1_FDX_s012
                                 + 1.20835 w_k1_IBM_s012
                                 + 0.855936 w_k1_INTC_s012
                                 + 0.982346 w_k1_KO_s012
                                 + 1.18822 w_k1_MAT_s012
                                 + 0.894192 w_k1_MSFT_s012
                                 + 1.03208 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.817882 w_k1_ADM_s012
                                 + 0.820612 w_k1_AXP_s012
                                 + 0.712743 w_k1_DIS_s012
                                 + 0.805736 w_k1_FDX_s012
                                 + 0.741661 w_k1_IBM_s012
                                 + 0.872682 w_k1_INTC_s012
                                 + 0.847635 w_k1_KO_s012
                                 + 0.710078 w_k1_MAT_s012
                                 + 0.748424 w_k1_MSFT_s012
                                 + 0.831257 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.49929 w_k1_ADM_s012
                                 + 1.7578 w_k1_AXP_s012 + 1.38348 w_k1_DIS_s012
                                 + 1.42934 w_k1_FDX_s012
                                 + 1.06289 w_k1_IBM_s012
                                 + 1.25345 w_k1_INTC_s012 + 1.0712 w_k1_KO_s012
                                 + 1.09081 w_k1_MAT_s012
                                 + 1.26422 w_k1_MSFT_s012
                                 + 1.22924 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.03205 w_k1_ADM_s012
                                 + 1.10026 w_k1_AXP_s012
                                 + 1.30741 w_k1_DIS_s012
                                 + 1.26071 w_k1_FDX_s012
                                 + 0.680451 w_k1_IBM_s012
                                 + 1.14667 w_k1_INTC_s012
                                 + 0.852023 w_k1_KO_s012
                                 + 0.630061 w_k1_MAT_s012
                                 + 1.2314 w_k1_MSFT_s012
                                 + 0.980105 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.08794 w_k1_ADM_s012
                                 + 1.42434 w_k1_AXP_s012
                                 + 0.887959 w_k1_DIS_s012
                                 + 1.01123 w_k1_FDX_s012 + 0.8743 w_k1_IBM_s012
                                 + 1.02485 w_k1_INTC_s012
                                 + 0.750055 w_k1_KO_s012 + 1.0026 w_k1_MAT_s012
                                 + 1.03743 w_k1_MSFT_s012
                                 + 1.03073 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.27982 w_k1_ADM_s012
                                 + 1.33314 w_k1_AXP_s012
                                 + 1.60173 w_k1_DIS_s012
                                 + 1.29618 w_k1_FDX_s012
                                 + 1.13891 w_k1_IBM_s012 + 1.315 w_k1_INTC_s012
                                 + 1.2384 w_k1_KO_s012 + 1.07867 w_k1_MAT_s012
                                 + 1.3749 w_k1_MSFT_s012 + 1.04705 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.1002 w_k1_ADM_s012
                                 + 1.55876 w_k1_AXP_s012
                                 + 1.21539 w_k1_DIS_s012
                                 + 1.06091 w_k1_FDX_s012
                                 + 1.50637 w_k1_IBM_s012
                                 + 1.1148 w_k1_INTC_s012 + 1.02004 w_k1_KO_s012
                                 + 1.53438 w_k1_MAT_s012
                                 + 1.02809 w_k1_MSFT_s012
                                 + 0.976864 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.99109 w_k1_ADM_s012
                                 + 0.941293 w_k1_AXP_s012
                                 + 1.04318 w_k1_DIS_s012
                                 + 1.02645 w_k1_FDX_s012
                                 + 0.931796 w_k1_IBM_s012
                                 + 0.93086 w_k1_INTC_s012
                                 + 0.998623 w_k1_KO_s012
                                 + 0.966872 w_k1_MAT_s012
                                 + 1.11504 w_k1_MSFT_s012
                                 + 1.02254 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.17239 w_k1_ADM_s013
                                 + 1.62913 w_k1_AXP_s013
                                 + 1.41974 w_k1_DIS_s013
                                 + 1.37626 w_k1_FDX_s013
                                 + 1.07167 w_k1_IBM_s013 + 1.215 w_k1_INTC_s013
                                 + 1.28997 w_k1_KO_s013 + 1.69871 w_k1_MAT_s013
                                 + 1.34949 w_k1_MSFT_s013
                                 + 1.37658 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.38964 w_k1_ADM_s013
                                 + 1.06776 w_k1_AXP_s013
                                 + 1.74746 w_k1_DIS_s013
                                 + 1.23451 w_k1_FDX_s013
                                 + 1.16125 w_k1_IBM_s013
                                 + 1.01903 w_k1_INTC_s013
                                 + 1.44532 w_k1_KO_s013
                                 + 0.846907 w_k1_MAT_s013
                                 + 1.19101 w_k1_MSFT_s013
                                 + 1.44951 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.61148 w_k1_ADM_s013
                                 + 1.343 w_k1_AXP_s013 + 1.50474 w_k1_DIS_s013
                                 + 1.22356 w_k1_FDX_s013
                                 + 1.68863 w_k1_IBM_s013
                                 + 0.980274 w_k1_INTC_s013
                                 + 1.44284 w_k1_KO_s013 + 1.71812 w_k1_MAT_s013
                                 + 0.939789 w_k1_MSFT_s013
                                 + 1.52864 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.770296 w_k1_ADM_s013
                                 + 0.698096 w_k1_AXP_s013
                                 + 0.614401 w_k1_DIS_s013
                                 + 0.834868 w_k1_FDX_s013
                                 + 0.74172 w_k1_IBM_s013
                                 + 0.654977 w_k1_INTC_s013
                                 + 0.692431 w_k1_KO_s013
                                 + 0.84616 w_k1_MAT_s013
                                 + 0.831207 w_k1_MSFT_s013
                                 + 0.996005 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.13574 w_k1_ADM_s013
                                 + 1.02942 w_k1_AXP_s013
                                 + 1.29545 w_k1_DIS_s013
                                 + 1.20398 w_k1_FDX_s013
                                 + 0.743391 w_k1_IBM_s013
                                 + 1.12224 w_k1_INTC_s013
                                 + 0.849438 w_k1_KO_s013
                                 + 0.66564 w_k1_MAT_s013
                                 + 1.26906 w_k1_MSFT_s013
                                 + 1.21037 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.882655 w_k1_ADM_s013
                                 + 0.856862 w_k1_AXP_s013
                                 + 0.977778 w_k1_DIS_s013
                                 + 0.961012 w_k1_FDX_s013
                                 + 0.825496 w_k1_IBM_s013
                                 + 0.959097 w_k1_INTC_s013
                                 + 0.982781 w_k1_KO_s013
                                 + 1.01171 w_k1_MAT_s013
                                 + 0.934025 w_k1_MSFT_s013
                                 + 1.15792 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.015 w_k1_ADM_s013
                                 + 0.939494 w_k1_AXP_s013
                                 + 1.40223 w_k1_DIS_s013
                                 + 1.26294 w_k1_FDX_s013
                                 + 1.00072 w_k1_IBM_s013
                                 + 1.15093 w_k1_INTC_s013
                                 + 0.956727 w_k1_KO_s013
                                 + 1.09351 w_k1_MAT_s013
                                 + 1.24056 w_k1_MSFT_s013
                                 + 1.10385 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.843567 w_k1_ADM_s013
                                 + 0.978387 w_k1_AXP_s013
                                 + 0.983259 w_k1_DIS_s013
                                 + 0.871143 w_k1_FDX_s013
                                 + 1.28217 w_k1_IBM_s013
                                 + 0.957507 w_k1_INTC_s013
                                 + 0.928425 w_k1_KO_s013
                                 + 1.67229 w_k1_MAT_s013
                                 + 0.892961 w_k1_MSFT_s013
                                 + 1.08027 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.16741 w_k1_ADM_s014
                                 + 1.35675 w_k1_AXP_s014 + 1.7548 w_k1_DIS_s014
                                 + 1.32204 w_k1_FDX_s014
                                 + 1.15962 w_k1_IBM_s014
                                 + 1.17264 w_k1_INTC_s014
                                 + 1.40645 w_k1_KO_s014 + 1.24358 w_k1_MAT_s014
                                 + 1.31938 w_k1_MSFT_s014
                                 + 1.24692 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.87143 w_k1_ADM_s014
                                 + 0.775005 w_k1_AXP_s014
                                 + 0.906669 w_k1_DIS_s014
                                 + 0.998603 w_k1_FDX_s014
                                 + 0.675497 w_k1_IBM_s014
                                 + 0.872737 w_k1_INTC_s014
                                 + 0.804195 w_k1_KO_s014
                                 + 0.708064 w_k1_MAT_s014
                                 + 0.930009 w_k1_MSFT_s014
                                 + 0.907796 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.09335 w_k1_ADM_s014
                                 + 1.20977 w_k1_AXP_s014
                                 + 1.34143 w_k1_DIS_s014
                                 + 1.10957 w_k1_FDX_s014
                                 + 1.33774 w_k1_IBM_s014
                                 + 0.992132 w_k1_INTC_s014
                                 + 1.40589 w_k1_KO_s014 + 1.48952 w_k1_MAT_s014
                                 + 1.06636 w_k1_MSFT_s014
                                 + 1.26888 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.23005 w_k1_ADM_s014
                                 + 1.628 w_k1_AXP_s014 + 1.22976 w_k1_DIS_s014
                                 + 1.36388 w_k1_FDX_s014
                                 + 0.908023 w_k1_IBM_s014
                                 + 1.04153 w_k1_INTC_s014
                                 + 1.03588 w_k1_KO_s014 + 1.28898 w_k1_MAT_s014
                                 + 1.04266 w_k1_MSFT_s014
                                 + 1.11012 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.868951 w_k1_ADM_s014
                                 + 0.766309 w_k1_AXP_s014
                                 + 0.652656 w_k1_DIS_s014
                                 + 0.760307 w_k1_FDX_s014
                                 + 0.863544 w_k1_IBM_s014
                                 + 0.664271 w_k1_INTC_s014
                                 + 0.942393 w_k1_KO_s014
                                 + 0.948668 w_k1_MAT_s014
                                 + 0.779399 w_k1_MSFT_s014
                                 + 0.853267 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.936091 w_k1_ADM_s014
                                 + 1.01131 w_k1_AXP_s014
                                 + 1.02981 w_k1_DIS_s014
                                 + 0.941469 w_k1_FDX_s014
                                 + 1.39085 w_k1_IBM_s014
                                 + 0.791389 w_k1_INTC_s014
                                 + 1.09237 w_k1_KO_s014 + 1.63742 w_k1_MAT_s014
                                 + 0.954879 w_k1_MSFT_s014
                                 + 1.16743 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.00276 w_k1_ADM_s014
                                 + 0.958474 w_k1_AXP_s014
                                 + 0.958396 w_k1_DIS_s014
                                 + 0.932162 w_k1_FDX_s014
                                 + 0.998927 w_k1_IBM_s014
                                 + 0.783283 w_k1_INTC_s014
                                 + 1.0647 w_k1_KO_s014 + 1.44438 w_k1_MAT_s014
                                 + 0.945359 w_k1_MSFT_s014
                                 + 0.998343 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.16375 w_k1_ADM_s014
                                 + 1.04081 w_k1_AXP_s014
                                 + 1.07061 w_k1_DIS_s014
                                 + 1.10034 w_k1_FDX_s014
                                 + 0.724405 w_k1_IBM_s014
                                 + 0.715863 w_k1_INTC_s014
                                 + 0.882423 w_k1_KO_s014
                                 + 0.715303 w_k1_MAT_s014
                                 + 0.993008 w_k1_MSFT_s014
                                 + 1.02039 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.30443 w_k1_ADM_s015
                                 + 1.09065 w_k1_AXP_s015
                                 + 1.14483 w_k1_DIS_s015
                                 + 1.13674 w_k1_FDX_s015
                                 + 0.786472 w_k1_IBM_s015
                                 + 1.06225 w_k1_INTC_s015
                                 + 0.960739 w_k1_KO_s015
                                 + 0.743226 w_k1_MAT_s015
                                 + 1.00691 w_k1_MSFT_s015
                                 + 1.02248 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.764853 w_k1_ADM_s015
                                 + 0.528696 w_k1_AXP_s015
                                 + 0.560194 w_k1_DIS_s015
                                 + 0.652085 w_k1_FDX_s015
                                 + 1.04508 w_k1_IBM_s015
                                 + 0.722215 w_k1_INTC_s015
                                 + 0.80041 w_k1_KO_s015 + 1.05858 w_k1_MAT_s015
                                 + 0.748009 w_k1_MSFT_s015
                                 + 0.76485 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.20605 w_k1_ADM_s015
                                 + 1.36301 w_k1_AXP_s015
                                 + 1.23428 w_k1_DIS_s015
                                 + 1.09287 w_k1_FDX_s015
                                 + 1.47564 w_k1_IBM_s015
                                 + 1.30155 w_k1_INTC_s015
                                 + 1.33524 w_k1_KO_s015 + 1.41491 w_k1_MAT_s015
                                 + 1.09918 w_k1_MSFT_s015
                                 + 1.22822 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.26731 w_k1_ADM_s015
                                 + 1.69174 w_k1_AXP_s015
                                 + 1.60043 w_k1_DIS_s015
                                 + 1.41421 w_k1_FDX_s015
                                 + 1.11765 w_k1_IBM_s015
                                 + 1.08139 w_k1_INTC_s015
                                 + 1.24183 w_k1_KO_s015 + 1.63764 w_k1_MAT_s015
                                 + 1.06407 w_k1_MSFT_s015
                                 + 1.14089 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.887383 w_k1_ADM_s015
                                 + 0.988267 w_k1_AXP_s015
                                 + 0.926265 w_k1_DIS_s015
                                 + 0.876616 w_k1_FDX_s015
                                 + 1.28127 w_k1_IBM_s015
                                 + 0.931133 w_k1_INTC_s015
                                 + 1.12147 w_k1_KO_s015 + 1.23319 w_k1_MAT_s015
                                 + 0.858494 w_k1_MSFT_s015
                                 + 0.877641 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.00019 w_k1_ADM_s015
                                 + 0.824296 w_k1_AXP_s015
                                 + 0.795664 w_k1_DIS_s015
                                 + 0.85419 w_k1_FDX_s015
                                 + 0.833663 w_k1_IBM_s015
                                 + 0.784899 w_k1_INTC_s015
                                 + 0.771853 w_k1_KO_s015
                                 + 0.977373 w_k1_MAT_s015
                                 + 0.831246 w_k1_MSFT_s015
                                 + 0.921081 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.17388 w_k1_ADM_s015
                                 + 1.20276 w_k1_AXP_s015
                                 + 1.43405 w_k1_DIS_s015
                                 + 1.17056 w_k1_FDX_s015
                                 + 1.08557 w_k1_IBM_s015
                                 + 1.09946 w_k1_INTC_s015
                                 + 1.17794 w_k1_KO_s015 + 0.93212 w_k1_MAT_s015
                                 + 1.08127 w_k1_MSFT_s015
                                 + 1.15746 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.850531 w_k1_ADM_s015
                                 + 0.576658 w_k1_AXP_s015
                                 + 0.576841 w_k1_DIS_s015
                                 + 0.717308 w_k1_FDX_s015
                                 + 0.710876 w_k1_IBM_s015
                                 + 0.825695 w_k1_INTC_s015
                                 + 0.664555 w_k1_KO_s015
                                 + 0.506483 w_k1_MAT_s015
                                 + 0.681125 w_k1_MSFT_s015
                                 + 0.776262 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.935974 w_k1_ADM_s016
                                 + 0.824233 w_k1_AXP_s016
                                 + 0.97005 w_k1_DIS_s016
                                 + 0.957773 w_k1_FDX_s016
                                 + 0.837522 w_k1_IBM_s016
                                 + 0.793569 w_k1_INTC_s016
                                 + 0.939858 w_k1_KO_s016
                                 + 0.900243 w_k1_MAT_s016
                                 + 1.06395 w_k1_MSFT_s016
                                 + 0.982249 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.729495 w_k1_ADM_s016
                                 + 0.836963 w_k1_AXP_s016
                                 + 0.729687 w_k1_DIS_s016
                                 + 0.741861 w_k1_FDX_s016
                                 + 0.967524 w_k1_IBM_s016
                                 + 0.856136 w_k1_INTC_s016
                                 + 0.848309 w_k1_KO_s016
                                 + 1.16695 w_k1_MAT_s016
                                 + 0.909776 w_k1_MSFT_s016
                                 + 0.933162 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.934618 w_k1_ADM_s016
                                 + 1.02278 w_k1_AXP_s016
                                 + 0.677831 w_k1_DIS_s016
                                 + 0.886852 w_k1_FDX_s016
                                 + 0.671075 w_k1_IBM_s016
                                 + 0.857251 w_k1_INTC_s016
                                 + 0.67562 w_k1_KO_s016 + 1.07608 w_k1_MAT_s016
                                 + 0.851323 w_k1_MSFT_s016
                                 + 0.841625 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.31636 w_k1_ADM_s016
                                 + 1.26304 w_k1_AXP_s016
                                 + 1.59756 w_k1_DIS_s016
                                 + 1.38466 w_k1_FDX_s016
                                 + 1.13436 w_k1_IBM_s016
                                 + 1.06937 w_k1_INTC_s016
                                 + 1.39895 w_k1_KO_s016 + 1.71525 w_k1_MAT_s016
                                 + 1.33562 w_k1_MSFT_s016
                                 + 1.16311 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.01576 w_k1_ADM_s016
                                 + 1.00581 w_k1_AXP_s016
                                 + 0.915174 w_k1_DIS_s016
                                 + 0.850878 w_k1_FDX_s016
                                 + 1.4851 w_k1_IBM_s016
                                 + 0.844922 w_k1_INTC_s016
                                 + 1.16628 w_k1_KO_s016 + 1.72743 w_k1_MAT_s016
                                 + 0.901564 w_k1_MSFT_s016
                                 + 0.940418 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.25897 w_k1_ADM_s016
                                 + 1.3311 w_k1_AXP_s016 + 1.07251 w_k1_DIS_s016
                                 + 1.12242 w_k1_FDX_s016
                                 + 1.20249 w_k1_IBM_s016
                                 + 1.16654 w_k1_INTC_s016
                                 + 1.22539 w_k1_KO_s016 + 1.97673 w_k1_MAT_s016
                                 + 1.40127 w_k1_MSFT_s016
                                 + 1.16409 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.46472 w_k1_ADM_s016
                                 + 1.59479 w_k1_AXP_s016
                                 + 1.66244 w_k1_DIS_s016
                                 + 1.26947 w_k1_FDX_s016
                                 + 1.54224 w_k1_IBM_s016
                                 + 0.969514 w_k1_INTC_s016
                                 + 1.26837 w_k1_KO_s016 + 2.21022 w_k1_MAT_s016
                                 + 1.30827 w_k1_MSFT_s016
                                 + 1.17222 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.08632 w_k1_ADM_s016
                                 + 1.20484 w_k1_AXP_s016
                                 + 1.16804 w_k1_DIS_s016
                                 + 1.02166 w_k1_FDX_s016
                                 + 1.07381 w_k1_IBM_s016
                                 + 1.03202 w_k1_INTC_s016
                                 + 1.10472 w_k1_KO_s016 + 1.23291 w_k1_MAT_s016
                                 + 1.17957 w_k1_MSFT_s016
                                 + 1.05556 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.984508 w_k1_ADM_s017
                                 + 1.33662 w_k1_AXP_s017
                                 + 1.45858 w_k1_DIS_s017
                                 + 1.19056 w_k1_FDX_s017
                                 + 1.31546 w_k1_IBM_s017
                                 + 0.983029 w_k1_INTC_s017
                                 + 1.21736 w_k1_KO_s017 + 1.56626 w_k1_MAT_s017
                                 + 1.15954 w_k1_MSFT_s017
                                 + 1.24837 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.31403 w_k1_ADM_s017
                                 + 1.05528 w_k1_AXP_s017
                                 + 0.998091 w_k1_DIS_s017
                                 + 1.23529 w_k1_FDX_s017
                                 + 0.716432 w_k1_IBM_s017
                                 + 1.18688 w_k1_INTC_s017
                                 + 0.806559 w_k1_KO_s017
                                 + 0.485171 w_k1_MAT_s017
                                 + 0.994252 w_k1_MSFT_s017
                                 + 1.0652 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.17443 w_k1_ADM_s017
                                 + 1.19652 w_k1_AXP_s017
                                 + 0.959325 w_k1_DIS_s017
                                 + 0.903233 w_k1_FDX_s017
                                 + 1.52098 w_k1_IBM_s017
                                 + 0.733648 w_k1_INTC_s017
                                 + 1.1472 w_k1_KO_s017 + 1.22361 w_k1_MAT_s017
                                 + 0.791298 w_k1_MSFT_s017
                                 + 1.05315 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00521 w_k1_ADM_s017
                                 + 0.915794 w_k1_AXP_s017
                                 + 0.991094 w_k1_DIS_s017
                                 + 0.915999 w_k1_FDX_s017
                                 + 1.13726 w_k1_IBM_s017
                                 + 0.828755 w_k1_INTC_s017
                                 + 1.09579 w_k1_KO_s017 + 1.12878 w_k1_MAT_s017
                                 + 0.959884 w_k1_MSFT_s017
                                 + 1.1572 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.10855 w_k1_ADM_s017
                                 + 1.25694 w_k1_AXP_s017
                                 + 1.14631 w_k1_DIS_s017
                                 + 1.18224 w_k1_FDX_s017
                                 + 0.964468 w_k1_IBM_s017
                                 + 1.12238 w_k1_INTC_s017
                                 + 0.998153 w_k1_KO_s017
                                 + 0.809327 w_k1_MAT_s017
                                 + 1.0183 w_k1_MSFT_s017 + 1.15713 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.960696 w_k1_ADM_s017
                                 + 0.906947 w_k1_AXP_s017
                                 + 0.95543 w_k1_DIS_s017
                                 + 1.01759 w_k1_FDX_s017
                                 + 0.884647 w_k1_IBM_s017
                                 + 0.995291 w_k1_INTC_s017
                                 + 0.915653 w_k1_KO_s017
                                 + 0.689565 w_k1_MAT_s017
                                 + 1.09241 w_k1_MSFT_s017
                                 + 0.984851 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.06362 w_k1_ADM_s017
                                 + 1.3714 w_k1_AXP_s017 + 1.63385 w_k1_DIS_s017
                                 + 1.40472 w_k1_FDX_s017
                                 + 0.981643 w_k1_IBM_s017
                                 + 1.27726 w_k1_INTC_s017
                                 + 1.15886 w_k1_KO_s017
                                 + 0.985511 w_k1_MAT_s017
                                 + 1.36204 w_k1_MSFT_s017
                                 + 1.24703 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.944418 w_k1_ADM_s017
                                 + 0.82609 w_k1_AXP_s017
                                 + 0.637981 w_k1_DIS_s017
                                 + 0.789453 w_k1_FDX_s017
                                 + 0.861665 w_k1_IBM_s017
                                 + 0.824067 w_k1_INTC_s017
                                 + 0.87735 w_k1_KO_s017
                                 + 0.764597 w_k1_MAT_s017
                                 + 0.779307 w_k1_MSFT_s017
                                 + 1.01888 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.855955 w_k1_ADM_s100
                                 + 1.18455 w_k1_AXP_s100
                                 + 1.02795 w_k1_DIS_s100
                                 + 1.12208 w_k1_FDX_s100
                                 + 0.890401 w_k1_IBM_s100
                                 + 1.08624 w_k1_INTC_s100
                                 + 0.974814 w_k1_KO_s100
                                 + 0.919295 w_k1_MAT_s100
                                 + 1.07042 w_k1_MSFT_s100
                                 + 1.05578 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.808423 w_k1_ADM_s100
                                 + 1.22504 w_k1_AXP_s100
                                 + 0.940899 w_k1_DIS_s100
                                 + 0.91911 w_k1_FDX_s100
                                 + 1.25007 w_k1_IBM_s100
                                 + 0.932129 w_k1_INTC_s100
                                 + 0.982235 w_k1_KO_s100
                                 + 1.42733 w_k1_MAT_s100
                                 + 0.973285 w_k1_MSFT_s100
                                 + 1.08857 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.869464 w_k1_ADM_s100
                                 + 0.919361 w_k1_AXP_s100
                                 + 0.830687 w_k1_DIS_s100
                                 + 0.979039 w_k1_FDX_s100
                                 + 0.706588 w_k1_IBM_s100
                                 + 1.09242 w_k1_INTC_s100
                                 + 0.862536 w_k1_KO_s100
                                 + 0.824594 w_k1_MAT_s100
                                 + 0.975602 w_k1_MSFT_s100
                                 + 0.985237 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.12639 w_k1_ADM_s100
                                 + 1.5685 w_k1_AXP_s100 + 1.52118 w_k1_DIS_s100
                                 + 1.42089 w_k1_FDX_s100
                                 + 1.19037 w_k1_IBM_s100
                                 + 1.28342 w_k1_INTC_s100
                                 + 1.34964 w_k1_KO_s100 + 1.32122 w_k1_MAT_s100
                                 + 1.27367 w_k1_MSFT_s100
                                 + 1.31464 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.836346 w_k1_ADM_s100
                                 + 1.29778 w_k1_AXP_s100
                                 + 0.800972 w_k1_DIS_s100
                                 + 0.988217 w_k1_FDX_s100
                                 + 0.815347 w_k1_IBM_s100
                                 + 0.976168 w_k1_INTC_s100
                                 + 0.805307 w_k1_KO_s100
                                 + 0.823071 w_k1_MAT_s100
                                 + 0.82983 w_k1_MSFT_s100
                                 + 1.05076 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.77267 w_k1_ADM_s100
                                 + 0.731357 w_k1_AXP_s100
                                 + 0.602868 w_k1_DIS_s100
                                 + 0.814211 w_k1_FDX_s100
                                 + 0.560171 w_k1_IBM_s100
                                 + 0.868449 w_k1_INTC_s100
                                 + 0.652604 w_k1_KO_s100
                                 + 0.411502 w_k1_MAT_s100
                                 + 0.768481 w_k1_MSFT_s100
                                 + 0.803255 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.66912 w_k1_ADM_s100
                                 + 0.80056 w_k1_AXP_s100
                                 + 0.687914 w_k1_DIS_s100
                                 + 0.783763 w_k1_FDX_s100
                                 + 0.883003 w_k1_IBM_s100
                                 + 0.814226 w_k1_INTC_s100
                                 + 0.755928 w_k1_KO_s100
                                 + 0.775492 w_k1_MAT_s100
                                 + 0.822558 w_k1_MSFT_s100
                                 + 0.877126 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.885245 w_k1_ADM_s100
                                 + 1.20432 w_k1_AXP_s100
                                 + 1.26572 w_k1_DIS_s100
                                 + 1.03237 w_k1_FDX_s100
                                 + 1.75327 w_k1_IBM_s100
                                 + 1.09794 w_k1_INTC_s100
                                 + 1.20346 w_k1_KO_s100 + 1.59953 w_k1_MAT_s100
                                 + 1.0221 w_k1_MSFT_s100 + 1.24712 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.06334 w_k1_ADM_s101
                                 + 0.870561 w_k1_AXP_s101
                                 + 1.13517 w_k1_DIS_s101
                                 + 0.924154 w_k1_FDX_s101
                                 + 0.933791 w_k1_IBM_s101
                                 + 0.816644 w_k1_INTC_s101
                                 + 0.930336 w_k1_KO_s101
                                 + 0.827152 w_k1_MAT_s101
                                 + 1.08747 w_k1_MSFT_s101
                                 + 0.960305 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.0211 w_k1_ADM_s101
                                 + 1.31096 w_k1_AXP_s101
                                 + 1.13348 w_k1_DIS_s101
                                 + 1.09082 w_k1_FDX_s101
                                 + 1.30514 w_k1_IBM_s101
                                 + 1.15668 w_k1_INTC_s101
                                 + 1.05084 w_k1_KO_s101 + 1.26679 w_k1_MAT_s101
                                 + 1.13439 w_k1_MSFT_s101
                                 + 1.09903 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.29467 w_k1_ADM_s101
                                 + 1.49214 w_k1_AXP_s101
                                 + 1.93134 w_k1_DIS_s101
                                 + 1.34288 w_k1_FDX_s101
                                 + 1.47198 w_k1_IBM_s101
                                 + 1.19728 w_k1_INTC_s101
                                 + 1.32593 w_k1_KO_s101 + 1.20692 w_k1_MAT_s101
                                 + 1.46191 w_k1_MSFT_s101
                                 + 1.21252 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.942212 w_k1_ADM_s101
                                 + 0.864357 w_k1_AXP_s101
                                 + 0.777994 w_k1_DIS_s101
                                 + 0.869445 w_k1_FDX_s101
                                 + 0.933848 w_k1_IBM_s101
                                 + 0.924154 w_k1_INTC_s101
                                 + 0.806864 w_k1_KO_s101
                                 + 0.801768 w_k1_MAT_s101
                                 + 1.02963 w_k1_MSFT_s101
                                 + 0.935126 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.16077 w_k1_ADM_s101
                                 + 1.10597 w_k1_AXP_s101
                                 + 0.941621 w_k1_DIS_s101
                                 + 1.12685 w_k1_FDX_s101
                                 + 0.728207 w_k1_IBM_s101
                                 + 0.884662 w_k1_INTC_s101
                                 + 0.744846 w_k1_KO_s101
                                 + 0.581188 w_k1_MAT_s101
                                 + 1.04506 w_k1_MSFT_s101
                                 + 0.979655 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.985821 w_k1_ADM_s101
                                 + 0.936018 w_k1_AXP_s101
                                 + 1.23618 w_k1_DIS_s101
                                 + 1.17115 w_k1_FDX_s101
                                 + 0.921841 w_k1_IBM_s101
                                 + 0.855626 w_k1_INTC_s101
                                 + 1.00819 w_k1_KO_s101
                                 + 0.878411 w_k1_MAT_s101
                                 + 1.17271 w_k1_MSFT_s101
                                 + 1.13745 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.789392 w_k1_ADM_s101
                                 + 0.848374 w_k1_AXP_s101
                                 + 0.966453 w_k1_DIS_s101
                                 + 0.844082 w_k1_FDX_s101
                                 + 1.35986 w_k1_IBM_s101
                                 + 0.945537 w_k1_INTC_s101
                                 + 1.10923 w_k1_KO_s101
                                 + 0.938191 w_k1_MAT_s101
                                 + 0.994142 w_k1_MSFT_s101
                                 + 0.963765 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.774929 w_k1_ADM_s101
                                 + 0.576704 w_k1_AXP_s101
                                 + 0.665652 w_k1_DIS_s101
                                 + 0.69749 w_k1_FDX_s101
                                 + 0.838777 w_k1_IBM_s101
                                 + 0.680925 w_k1_INTC_s101
                                 + 0.8028 w_k1_KO_s101 + 0.694906 w_k1_MAT_s101
                                 + 0.830002 w_k1_MSFT_s101
                                 + 0.828686 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.835699 w_k1_ADM_s102
                                 + 1.01188 w_k1_AXP_s102
                                 + 0.913441 w_k1_DIS_s102
                                 + 0.952958 w_k1_FDX_s102
                                 + 1.10522 w_k1_IBM_s102
                                 + 1.22718 w_k1_INTC_s102 + 1.0851 w_k1_KO_s102
                                 + 1.16969 w_k1_MAT_s102
                                 + 1.10858 w_k1_MSFT_s102 + 1.117 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.674899 w_k1_ADM_s102
                                 + 0.721227 w_k1_AXP_s102
                                 + 0.617568 w_k1_DIS_s102
                                 + 0.831408 w_k1_FDX_s102
                                 + 0.812353 w_k1_IBM_s102
                                 + 0.930623 w_k1_INTC_s102
                                 + 0.719538 w_k1_KO_s102
                                 + 0.853659 w_k1_MAT_s102
                                 + 0.846106 w_k1_MSFT_s102
                                 + 0.829277 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.904456 w_k1_ADM_s102
                                 + 1.34952 w_k1_AXP_s102
                                 + 1.29088 w_k1_DIS_s102
                                 + 1.19983 w_k1_FDX_s102
                                 + 1.45764 w_k1_IBM_s102
                                 + 1.16711 w_k1_INTC_s102 + 1.2524 w_k1_KO_s102
                                 + 2.08783 w_k1_MAT_s102
                                 + 1.05209 w_k1_MSFT_s102
                                 + 1.16256 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.08 w_k1_ADM_s102
                                 + 1.15286 w_k1_AXP_s102
                                 + 1.45343 w_k1_DIS_s102
                                 + 1.26626 w_k1_FDX_s102
                                 + 0.942274 w_k1_IBM_s102
                                 + 1.52197 w_k1_INTC_s102
                                 + 1.06504 w_k1_KO_s102
                                 + 0.983966 w_k1_MAT_s102
                                 + 1.28626 w_k1_MSFT_s102
                                 + 1.10273 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.709573 w_k1_ADM_s102
                                 + 0.68353 w_k1_AXP_s102
                                 + 0.643013 w_k1_DIS_s102
                                 + 0.650178 w_k1_FDX_s102
                                 + 1.13239 w_k1_IBM_s102
                                 + 0.884583 w_k1_INTC_s102
                                 + 0.97306 w_k1_KO_s102
                                 + 0.872671 w_k1_MAT_s102
                                 + 0.777051 w_k1_MSFT_s102
                                 + 0.888658 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.88959 w_k1_ADM_s102
                                 + 1.1146 w_k1_AXP_s102
                                 + 0.859842 w_k1_DIS_s102
                                 + 1.07536 w_k1_FDX_s102
                                 + 0.698749 w_k1_IBM_s102
                                 + 1.1426 w_k1_INTC_s102
                                 + 0.738317 w_k1_KO_s102
                                 + 0.726578 w_k1_MAT_s102
                                 + 0.958247 w_k1_MSFT_s102
                                 + 0.99054 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.19581 w_k1_ADM_s102
                                 + 1.50856 w_k1_AXP_s102
                                 + 1.91585 w_k1_DIS_s102
                                 + 1.62004 w_k1_FDX_s102
                                 + 1.07982 w_k1_IBM_s102
                                 + 1.67534 w_k1_INTC_s102
                                 + 1.36827 w_k1_KO_s102 + 1.3647 w_k1_MAT_s102
                                 + 1.70779 w_k1_MSFT_s102
                                 + 1.22405 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.810031 w_k1_ADM_s102
                                 + 1.13995 w_k1_AXP_s102 + 1.1152 w_k1_DIS_s102
                                 + 1.13332 w_k1_FDX_s102
                                 + 0.880366 w_k1_IBM_s102
                                 + 1.58063 w_k1_INTC_s102
                                 + 1.01686 w_k1_KO_s102 + 1.09782 w_k1_MAT_s102
                                 + 1.24085 w_k1_MSFT_s102
                                 + 1.15385 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.733224 w_k1_ADM_s103
                                 + 0.720923 w_k1_AXP_s103
                                 + 0.635196 w_k1_DIS_s103
                                 + 0.677822 w_k1_FDX_s103
                                 + 1.01909 w_k1_IBM_s103
                                 + 0.664842 w_k1_INTC_s103
                                 + 0.775068 w_k1_KO_s103
                                 + 0.79314 w_k1_MAT_s103
                                 + 0.615764 w_k1_MSFT_s103
                                 + 0.791474 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.978604 w_k1_ADM_s103
                                 + 0.718141 w_k1_AXP_s103
                                 + 0.887549 w_k1_DIS_s103
                                 + 0.854589 w_k1_FDX_s103
                                 + 1.09368 w_k1_IBM_s103
                                 + 0.676779 w_k1_INTC_s103
                                 + 0.999774 w_k1_KO_s103
                                 + 0.700662 w_k1_MAT_s103
                                 + 0.911824 w_k1_MSFT_s103
                                 + 0.927276 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.984074 w_k1_ADM_s103
                                 + 1.05017 w_k1_AXP_s103
                                 + 1.30948 w_k1_DIS_s103
                                 + 1.03304 w_k1_FDX_s103
                                 + 1.72166 w_k1_IBM_s103
                                 + 0.954152 w_k1_INTC_s103
                                 + 1.43218 w_k1_KO_s103 + 1.64408 w_k1_MAT_s103
                                 + 1.03465 w_k1_MSFT_s103 + 1.1179 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.38344 w_k1_ADM_s103
                                 + 1.82329 w_k1_AXP_s103
                                 + 1.17106 w_k1_DIS_s103
                                 + 1.37227 w_k1_FDX_s103
                                 + 1.05148 w_k1_IBM_s103
                                 + 1.02193 w_k1_INTC_s103
                                 + 1.00274 w_k1_KO_s103 + 1.38926 w_k1_MAT_s103
                                 + 1.02763 w_k1_MSFT_s103
                                 + 1.01181 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.986686 w_k1_ADM_s103
                                 + 0.893951 w_k1_AXP_s103
                                 + 0.791756 w_k1_DIS_s103
                                 + 0.913808 w_k1_FDX_s103
                                 + 0.835069 w_k1_IBM_s103
                                 + 0.841809 w_k1_INTC_s103
                                 + 0.726685 w_k1_KO_s103
                                 + 0.766282 w_k1_MAT_s103
                                 + 0.804942 w_k1_MSFT_s103
                                 + 0.886812 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.43268 w_k1_ADM_s103
                                 + 1.35562 w_k1_AXP_s103
                                 + 1.64771 w_k1_DIS_s103
                                 + 1.34312 w_k1_FDX_s103
                                 + 1.05907 w_k1_IBM_s103
                                 + 1.15385 w_k1_INTC_s103
                                 + 1.25574 w_k1_KO_s103
                                 + 0.893619 w_k1_MAT_s103
                                 + 1.24104 w_k1_MSFT_s103
                                 + 1.12695 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.06834 w_k1_ADM_s103
                                 + 0.899858 w_k1_AXP_s103
                                 + 1.08965 w_k1_DIS_s103
                                 + 1.07078 w_k1_FDX_s103
                                 + 0.701128 w_k1_IBM_s103
                                 + 0.958471 w_k1_INTC_s103
                                 + 0.860613 w_k1_KO_s103
                                 + 0.499785 w_k1_MAT_s103
                                 + 1.16647 w_k1_MSFT_s103
                                 + 0.950761 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.05985 w_k1_ADM_s103
                                 + 1.11066 w_k1_AXP_s103
                                 + 1.05486 w_k1_DIS_s103
                                 + 0.989327 w_k1_FDX_s103
                                 + 1.2445 w_k1_IBM_s103
                                 + 0.908058 w_k1_INTC_s103
                                 + 1.06165 w_k1_KO_s103 + 1.06126 w_k1_MAT_s103
                                 + 0.928145 w_k1_MSFT_s103
                                 + 1.10527 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.06687 w_k1_ADM_s104
                                 + 0.922697 w_k1_AXP_s104
                                 + 0.783215 w_k1_DIS_s104
                                 + 0.906473 w_k1_FDX_s104
                                 + 0.918909 w_k1_IBM_s104
                                 + 0.896834 w_k1_INTC_s104
                                 + 0.827157 w_k1_KO_s104
                                 + 1.16161 w_k1_MAT_s104
                                 + 0.912031 w_k1_MSFT_s104
                                 + 0.954906 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.96274 w_k1_ADM_s104
                                 + 1.02267 w_k1_AXP_s104
                                 + 0.946129 w_k1_DIS_s104
                                 + 0.939625 w_k1_FDX_s104
                                 + 1.11114 w_k1_IBM_s104
                                 + 0.969782 w_k1_INTC_s104
                                 + 1.04291 w_k1_KO_s104 + 1.39419 w_k1_MAT_s104
                                 + 0.87848 w_k1_MSFT_s104
                                 + 0.910496 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.825425 w_k1_ADM_s104
                                 + 0.930955 w_k1_AXP_s104
                                 + 0.790808 w_k1_DIS_s104
                                 + 0.885781 w_k1_FDX_s104
                                 + 0.662014 w_k1_IBM_s104
                                 + 0.830532 w_k1_INTC_s104
                                 + 0.744864 w_k1_KO_s104
                                 + 0.708882 w_k1_MAT_s104
                                 + 0.925865 w_k1_MSFT_s104
                                 + 0.89511 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.11672 w_k1_ADM_s104
                                 + 1.19392 w_k1_AXP_s104
                                 + 1.33075 w_k1_DIS_s104 + 1.02 w_k1_FDX_s104
                                 + 1.46841 w_k1_IBM_s104
                                 + 1.15693 w_k1_INTC_s104
                                 + 1.26387 w_k1_KO_s104 + 1.79296 w_k1_MAT_s104
                                 + 1.06213 w_k1_MSFT_s104
                                 + 1.05547 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.08698 w_k1_ADM_s104
                                 + 1.17608 w_k1_AXP_s104
                                 + 1.21187 w_k1_DIS_s104
                                 + 1.14724 w_k1_FDX_s104
                                 + 0.955112 w_k1_IBM_s104
                                 + 1.13437 w_k1_INTC_s104
                                 + 1.07493 w_k1_KO_s104 + 1.33284 w_k1_MAT_s104
                                 + 1.11452 w_k1_MSFT_s104
                                 + 1.07993 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.82442 w_k1_ADM_s104
                                 + 0.69359 w_k1_AXP_s104
                                 + 0.682697 w_k1_DIS_s104
                                 + 0.706363 w_k1_FDX_s104
                                 + 1.04791 w_k1_IBM_s104
                                 + 0.791822 w_k1_INTC_s104
                                 + 0.841858 w_k1_KO_s104
                                 + 1.05668 w_k1_MAT_s104
                                 + 0.80845 w_k1_MSFT_s104
                                 + 0.841795 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.39344 w_k1_ADM_s104
                                 + 1.43179 w_k1_AXP_s104
                                 + 1.56918 w_k1_DIS_s104
                                 + 1.25535 w_k1_FDX_s104
                                 + 1.24484 w_k1_IBM_s104
                                 + 1.29889 w_k1_INTC_s104
                                 + 1.24524 w_k1_KO_s104 + 1.46478 w_k1_MAT_s104
                                 + 1.38382 w_k1_MSFT_s104
                                 + 1.27842 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.782684 w_k1_ADM_s104
                                 + 0.758172 w_k1_AXP_s104
                                 + 1.12575 w_k1_DIS_s104
                                 + 0.869957 w_k1_FDX_s104
                                 + 0.830346 w_k1_IBM_s104
                                 + 1.03072 w_k1_INTC_s104
                                 + 0.889934 w_k1_KO_s104
                                 + 0.761277 w_k1_MAT_s104
                                 + 0.970544 w_k1_MSFT_s104
                                 + 0.870175 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.971573 w_k1_ADM_s105
                                 + 0.995499 w_k1_AXP_s105
                                 + 1.22721 w_k1_DIS_s105
                                 + 1.08501 w_k1_FDX_s105
                                 + 0.844719 w_k1_IBM_s105
                                 + 1.06584 w_k1_INTC_s105
                                 + 0.927642 w_k1_KO_s105
                                 + 0.837756 w_k1_MAT_s105
                                 + 1.01534 w_k1_MSFT_s105
                                 + 1.02942 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.842203 w_k1_ADM_s105
                                 + 0.737843 w_k1_AXP_s105
                                 + 0.636855 w_k1_DIS_s105
                                 + 0.88812 w_k1_FDX_s105
                                 + 0.617133 w_k1_IBM_s105
                                 + 1.03958 w_k1_INTC_s105
                                 + 0.779952 w_k1_KO_s105
                                 + 0.576374 w_k1_MAT_s105
                                 + 0.88124 w_k1_MSFT_s105
                                 + 0.883441 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.23145 w_k1_ADM_s105
                                 + 1.63113 w_k1_AXP_s105
                                 + 1.22668 w_k1_DIS_s105
                                 + 1.31802 w_k1_FDX_s105
                                 + 0.988694 w_k1_IBM_s105
                                 + 1.20216 w_k1_INTC_s105
                                 + 1.10975 w_k1_KO_s105 + 1.58729 w_k1_MAT_s105
                                 + 1.06232 w_k1_MSFT_s105 + 1.2959 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.931743 w_k1_ADM_s105
                                 + 0.953721 w_k1_AXP_s105
                                 + 0.979478 w_k1_DIS_s105
                                 + 0.941573 w_k1_FDX_s105
                                 + 0.974944 w_k1_IBM_s105
                                 + 1.09206 w_k1_INTC_s105
                                 + 1.03743 w_k1_KO_s105 + 1.04666 w_k1_MAT_s105
                                 + 0.93625 w_k1_MSFT_s105
                                 + 0.938052 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.14035 w_k1_ADM_s105
                                 + 1.11301 w_k1_AXP_s105
                                 + 0.927376 w_k1_DIS_s105
                                 + 0.954829 w_k1_FDX_s105
                                 + 1.29002 w_k1_IBM_s105
                                 + 0.999944 w_k1_INTC_s105
                                 + 1.10111 w_k1_KO_s105 + 1.31303 w_k1_MAT_s105
                                 + 0.926844 w_k1_MSFT_s105
                                 + 1.0323 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.17306 w_k1_ADM_s105
                                 + 1.10265 w_k1_AXP_s105
                                 + 1.39897 w_k1_DIS_s105
                                 + 1.17184 w_k1_FDX_s105
                                 + 1.12787 w_k1_IBM_s105
                                 + 1.2717 w_k1_INTC_s105 + 1.22946 w_k1_KO_s105
                                 + 0.972794 w_k1_MAT_s105
                                 + 1.21466 w_k1_MSFT_s105
                                 + 1.13498 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.835847 w_k1_ADM_s105
                                 + 0.671977 w_k1_AXP_s105
                                 + 0.69353 w_k1_DIS_s105
                                 + 0.769633 w_k1_FDX_s105
                                 + 0.875613 w_k1_IBM_s105
                                 + 0.882196 w_k1_INTC_s105
                                 + 0.936983 w_k1_KO_s105
                                 + 0.832165 w_k1_MAT_s105
                                 + 0.731078 w_k1_MSFT_s105
                                 + 0.903608 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.20617 w_k1_ADM_s105
                                 + 1.38372 w_k1_AXP_s105
                                 + 1.60987 w_k1_DIS_s105
                                 + 1.30915 w_k1_FDX_s105
                                 + 1.54234 w_k1_IBM_s105
                                 + 1.12088 w_k1_INTC_s105
                                 + 1.35256 w_k1_KO_s105 + 1.50892 w_k1_MAT_s105
                                 + 1.2362 w_k1_MSFT_s105 + 1.35675 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.729591 w_k1_ADM_s106
                                 + 0.793657 w_k1_AXP_s106
                                 + 0.630718 w_k1_DIS_s106
                                 + 0.734843 w_k1_FDX_s106
                                 + 1.06028 w_k1_IBM_s106
                                 + 0.812098 w_k1_INTC_s106
                                 + 0.80668 w_k1_KO_s106 + 1.40584 w_k1_MAT_s106
                                 + 0.740356 w_k1_MSFT_s106
                                 + 0.901766 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.97449 w_k1_ADM_s106
                                 + 0.977387 w_k1_AXP_s106
                                 + 0.720364 w_k1_DIS_s106
                                 + 0.868155 w_k1_FDX_s106
                                 + 0.66212 w_k1_IBM_s106
                                 + 0.877901 w_k1_INTC_s106
                                 + 0.664112 w_k1_KO_s106
                                 + 0.768233 w_k1_MAT_s106
                                 + 0.838298 w_k1_MSFT_s106
                                 + 0.868721 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.905443 w_k1_ADM_s106
                                 + 1.34411 w_k1_AXP_s106
                                 + 1.14907 w_k1_DIS_s106
                                 + 1.21699 w_k1_FDX_s106
                                 + 0.830483 w_k1_IBM_s106
                                 + 1.3739 w_k1_INTC_s106
                                 + 0.931503 w_k1_KO_s106
                                 + 1.08996 w_k1_MAT_s106
                                 + 1.06802 w_k1_MSFT_s106
                                 + 0.95479 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.679173 w_k1_ADM_s106
                                 + 0.569915 w_k1_AXP_s106
                                 + 0.531203 w_k1_DIS_s106
                                 + 0.61653 w_k1_FDX_s106
                                 + 0.909492 w_k1_IBM_s106
                                 + 0.79593 w_k1_INTC_s106
                                 + 0.763413 w_k1_KO_s106
                                 + 0.84956 w_k1_MAT_s106
                                 + 0.620318 w_k1_MSFT_s106
                                 + 0.752502 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.955764 w_k1_ADM_s106
                                 + 1.11211 w_k1_AXP_s106
                                 + 1.39056 w_k1_DIS_s106
                                 + 1.09903 w_k1_FDX_s106
                                 + 1.00765 w_k1_IBM_s106
                                 + 1.21977 w_k1_INTC_s106
                                 + 1.17159 w_k1_KO_s106 + 1.26979 w_k1_MAT_s106
                                 + 1.22673 w_k1_MSFT_s106
                                 + 1.13601 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.821489 w_k1_ADM_s106
                                 + 1.2627 w_k1_AXP_s106 + 1.26639 w_k1_DIS_s106
                                 + 1.13873 w_k1_FDX_s106
                                 + 1.30919 w_k1_IBM_s106
                                 + 1.1516 w_k1_INTC_s106 + 1.22474 w_k1_KO_s106
                                 + 1.60656 w_k1_MAT_s106
                                 + 0.99682 w_k1_MSFT_s106
                                 + 1.21624 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.19046 w_k1_ADM_s106
                                 + 1.41655 w_k1_AXP_s106
                                 + 1.17219 w_k1_DIS_s106
                                 + 1.38873 w_k1_FDX_s106
                                 + 0.658321 w_k1_IBM_s106
                                 + 1.32914 w_k1_INTC_s106
                                 + 0.824535 w_k1_KO_s106
                                 + 0.837223 w_k1_MAT_s106
                                 + 1.03376 w_k1_MSFT_s106 + 1.0122 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.23351 w_k1_ADM_s106
                                 + 1.18702 w_k1_AXP_s106 + 1.5523 w_k1_DIS_s106
                                 + 1.08312 w_k1_FDX_s106
                                 + 1.75757 w_k1_IBM_s106
                                 + 1.07888 w_k1_INTC_s106
                                 + 1.34037 w_k1_KO_s106 + 2.30114 w_k1_MAT_s106
                                 + 0.998987 w_k1_MSFT_s106
                                 + 1.09815 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.16205 w_k1_ADM_s107
                                 + 1.02124 w_k1_AXP_s107
                                 + 1.17198 w_k1_DIS_s107
                                 + 1.08432 w_k1_FDX_s107
                                 + 0.910688 w_k1_IBM_s107
                                 + 1.16309 w_k1_INTC_s107
                                 + 1.14393 w_k1_KO_s107
                                 + 0.629766 w_k1_MAT_s107
                                 + 1.05361 w_k1_MSFT_s107
                                 + 0.995148 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.02888 w_k1_ADM_s107
                                 + 1.08816 w_k1_AXP_s107
                                 + 0.941996 w_k1_DIS_s107
                                 + 0.91822 w_k1_FDX_s107
                                 + 1.37037 w_k1_IBM_s107
                                 + 1.02681 w_k1_INTC_s107
                                 + 1.27265 w_k1_KO_s107 + 1.20793 w_k1_MAT_s107
                                 + 0.904258 w_k1_MSFT_s107
                                 + 0.956992 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.16455 w_k1_ADM_s107
                                 + 1.47382 w_k1_AXP_s107
                                 + 1.61342 w_k1_DIS_s107
                                 + 1.27572 w_k1_FDX_s107
                                 + 1.63101 w_k1_IBM_s107
                                 + 1.42853 w_k1_INTC_s107
                                 + 1.46279 w_k1_KO_s107 + 1.70866 w_k1_MAT_s107
                                 + 1.14797 w_k1_MSFT_s107
                                 + 1.18483 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.965442 w_k1_ADM_s107
                                 + 1.4514 w_k1_AXP_s107 + 1.06571 w_k1_DIS_s107
                                 + 1.13875 w_k1_FDX_s107
                                 + 1.20632 w_k1_IBM_s107
                                 + 1.20772 w_k1_INTC_s107 + 1.2201 w_k1_KO_s107
                                 + 1.48416 w_k1_MAT_s107
                                 + 1.01844 w_k1_MSFT_s107
                                 + 1.07866 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.36871 w_k1_ADM_s107
                                 + 1.48023 w_k1_AXP_s107
                                 + 1.60517 w_k1_DIS_s107
                                 + 1.44921 w_k1_FDX_s107
                                 + 1.06776 w_k1_IBM_s107
                                 + 1.41719 w_k1_INTC_s107
                                 + 1.29424 w_k1_KO_s107 + 1.10132 w_k1_MAT_s107
                                 + 1.19975 w_k1_MSFT_s107
                                 + 1.23201 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.695732 w_k1_ADM_s107
                                 + 0.744905 w_k1_AXP_s107
                                 + 0.711409 w_k1_DIS_s107
                                 + 0.762007 w_k1_FDX_s107
                                 + 1.11152 w_k1_IBM_s107
                                 + 0.76151 w_k1_INTC_s107
                                 + 1.00293 w_k1_KO_s107 + 0.83022 w_k1_MAT_s107
                                 + 0.700986 w_k1_MSFT_s107
                                 + 0.79513 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.966705 w_k1_ADM_s107
                                 + 0.862112 w_k1_AXP_s107
                                 + 1.26233 w_k1_DIS_s107
                                 + 0.985293 w_k1_FDX_s107
                                 + 1.1612 w_k1_IBM_s107
                                 + 1.11055 w_k1_INTC_s107
                                 + 1.25031 w_k1_KO_s107
                                 + 0.858047 w_k1_MAT_s107
                                 + 1.10149 w_k1_MSFT_s107
                                 + 0.971114 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.799931 w_k1_ADM_s107
                                 + 0.828063 w_k1_AXP_s107
                                 + 0.813119 w_k1_DIS_s107
                                 + 0.865295 w_k1_FDX_s107
                                 + 0.736113 w_k1_IBM_s107
                                 + 0.919779 w_k1_INTC_s107
                                 + 0.960264 w_k1_KO_s107
                                 + 0.55143 w_k1_MAT_s107
                                 + 0.797256 w_k1_MSFT_s107
                                 + 0.882783 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.929858 w_k1_ADM_s110
                                 + 0.916865 w_k1_AXP_s110
                                 + 0.791709 w_k1_DIS_s110
                                 + 0.944568 w_k1_FDX_s110
                                 + 0.805496 w_k1_IBM_s110
                                 + 1.11329 w_k1_INTC_s110
                                 + 0.874737 w_k1_KO_s110
                                 + 0.518114 w_k1_MAT_s110
                                 + 0.949741 w_k1_MSFT_s110
                                 + 0.842711 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.982083 w_k1_ADM_s110
                                 + 0.537184 w_k1_AXP_s110
                                 + 0.618105 w_k1_DIS_s110
                                 + 0.722219 w_k1_FDX_s110
                                 + 0.719116 w_k1_IBM_s110
                                 + 0.695147 w_k1_INTC_s110
                                 + 0.776075 w_k1_KO_s110
                                 + 0.538431 w_k1_MAT_s110
                                 + 0.705046 w_k1_MSFT_s110
                                 + 0.786006 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.32015 w_k1_ADM_s110
                                 + 1.03967 w_k1_AXP_s110
                                 + 1.45494 w_k1_DIS_s110
                                 + 1.36382 w_k1_FDX_s110
                                 + 0.794704 w_k1_IBM_s110
                                 + 1.46449 w_k1_INTC_s110
                                 + 1.08973 w_k1_KO_s110
                                 + 0.675842 w_k1_MAT_s110
                                 + 1.28984 w_k1_MSFT_s110
                                 + 1.10686 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.32818 w_k1_ADM_s110
                                 + 1.2156 w_k1_AXP_s110 + 1.72352 w_k1_DIS_s110
                                 + 1.33901 w_k1_FDX_s110
                                 + 1.28994 w_k1_IBM_s110
                                 + 1.57655 w_k1_INTC_s110
                                 + 1.41587 w_k1_KO_s110
                                 + 0.789608 w_k1_MAT_s110
                                 + 1.45165 w_k1_MSFT_s110 + 1.1079 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.21294 w_k1_ADM_s110
                                 + 1.21866 w_k1_AXP_s110
                                 + 1.00182 w_k1_DIS_s110
                                 + 1.08116 w_k1_FDX_s110
                                 + 1.65245 w_k1_IBM_s110
                                 + 1.00285 w_k1_INTC_s110
                                 + 1.36662 w_k1_KO_s110 + 1.57381 w_k1_MAT_s110
                                 + 1.08479 w_k1_MSFT_s110
                                 + 1.04113 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.09331 w_k1_ADM_s110
                                 + 0.916139 w_k1_AXP_s110
                                 + 0.844184 w_k1_DIS_s110
                                 + 0.875153 w_k1_FDX_s110
                                 + 1.27294 w_k1_IBM_s110
                                 + 0.905674 w_k1_INTC_s110
                                 + 0.996926 w_k1_KO_s110
                                 + 0.912356 w_k1_MAT_s110
                                 + 0.958037 w_k1_MSFT_s110
                                 + 0.943346 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.35421 w_k1_ADM_s110
                                 + 1.27615 w_k1_AXP_s110
                                 + 1.34733 w_k1_DIS_s110
                                 + 1.23346 w_k1_FDX_s110
                                 + 1.30201 w_k1_IBM_s110
                                 + 1.17396 w_k1_INTC_s110
                                 + 1.16728 w_k1_KO_s110 + 0.94092 w_k1_MAT_s110
                                 + 1.20173 w_k1_MSFT_s110
                                 + 1.12262 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.16969 w_k1_ADM_s110
                                 + 1.06409 w_k1_AXP_s110 + 0.9458 w_k1_DIS_s110
                                 + 1.09125 w_k1_FDX_s110
                                 + 0.85867 w_k1_IBM_s110
                                 + 1.31525 w_k1_INTC_s110
                                 + 0.901779 w_k1_KO_s110
                                 + 0.921232 w_k1_MAT_s110
                                 + 1.15522 w_k1_MSFT_s110
                                 + 0.984686 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.817826 w_k1_ADM_s111
                                 + 0.772199 w_k1_AXP_s111
                                 + 0.578926 w_k1_DIS_s111
                                 + 0.724338 w_k1_FDX_s111
                                 + 0.879408 w_k1_IBM_s111
                                 + 0.873156 w_k1_INTC_s111
                                 + 0.566867 w_k1_KO_s111
                                 + 0.508394 w_k1_MAT_s111
                                 + 0.677975 w_k1_MSFT_s111
                                 + 0.674808 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.838855 w_k1_ADM_s111
                                 + 1.18604 w_k1_AXP_s111
                                 + 1.55869 w_k1_DIS_s111
                                 + 1.14676 w_k1_FDX_s111
                                 + 1.15311 w_k1_IBM_s111
                                 + 1.07852 w_k1_INTC_s111
                                 + 1.16974 w_k1_KO_s111
                                 + 0.986886 w_k1_MAT_s111
                                 + 1.14425 w_k1_MSFT_s111
                                 + 1.16259 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.9615 w_k1_ADM_s111
                                 + 0.814249 w_k1_AXP_s111
                                 + 1.20676 w_k1_DIS_s111
                                 + 1.02093 w_k1_FDX_s111
                                 + 1.00928 w_k1_IBM_s111
                                 + 0.921455 w_k1_INTC_s111
                                 + 1.04738 w_k1_KO_s111
                                 + 0.826815 w_k1_MAT_s111
                                 + 0.93618 w_k1_MSFT_s111
                                 + 0.995328 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09077 w_k1_ADM_s111
                                 + 1.27998 w_k1_AXP_s111
                                 + 1.22843 w_k1_DIS_s111
                                 + 1.09281 w_k1_FDX_s111
                                 + 1.53871 w_k1_IBM_s111
                                 + 1.06213 w_k1_INTC_s111
                                 + 1.04044 w_k1_KO_s111 + 1.05659 w_k1_MAT_s111
                                 + 0.992662 w_k1_MSFT_s111
                                 + 1.01852 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.833998 w_k1_ADM_s111
                                 + 0.778994 w_k1_AXP_s111
                                 + 0.800125 w_k1_DIS_s111
                                 + 0.815814 w_k1_FDX_s111
                                 + 0.929539 w_k1_IBM_s111
                                 + 0.894823 w_k1_INTC_s111
                                 + 0.87462 w_k1_KO_s111 + 1.00992 w_k1_MAT_s111
                                 + 0.767452 w_k1_MSFT_s111
                                 + 0.901327 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.2189 w_k1_ADM_s111
                                 + 1.31748 w_k1_AXP_s111
                                 + 1.30512 w_k1_DIS_s111
                                 + 1.31485 w_k1_FDX_s111
                                 + 0.823579 w_k1_IBM_s111
                                 + 1.23184 w_k1_INTC_s111
                                 + 0.997118 w_k1_KO_s111
                                 + 0.707538 w_k1_MAT_s111
                                 + 1.14597 w_k1_MSFT_s111 + 1.1247 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.681453 w_k1_ADM_s111
                                 + 0.953459 w_k1_AXP_s111
                                 + 0.699961 w_k1_DIS_s111
                                 + 0.731248 w_k1_FDX_s111
                                 + 0.705201 w_k1_IBM_s111
                                 + 0.819792 w_k1_INTC_s111
                                 + 0.644333 w_k1_KO_s111
                                 + 0.58828 w_k1_MAT_s111
                                 + 0.749146 w_k1_MSFT_s111
                                 + 0.725107 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.01857 w_k1_ADM_s111
                                 + 1.01152 w_k1_AXP_s111
                                 + 0.887941 w_k1_DIS_s111
                                 + 0.908013 w_k1_FDX_s111
                                 + 1.12959 w_k1_IBM_s111
                                 + 0.880643 w_k1_INTC_s111
                                 + 0.796092 w_k1_KO_s111
                                 + 1.00967 w_k1_MAT_s111
                                 + 0.888097 w_k1_MSFT_s111
                                 + 0.975043 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.39009 w_k1_ADM_s112
                                 + 1.08757 w_k1_AXP_s112
                                 + 1.65838 w_k1_DIS_s112
                                 + 1.40126 w_k1_FDX_s112
                                 + 1.22425 w_k1_IBM_s112
                                 + 1.2607 w_k1_INTC_s112 + 1.49143 w_k1_KO_s112
                                 + 0.804482 w_k1_MAT_s112
                                 + 1.34053 w_k1_MSFT_s112
                                 + 1.08537 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.21779 w_k1_ADM_s112
                                 + 1.12906 w_k1_AXP_s112
                                 + 1.02888 w_k1_DIS_s112
                                 + 1.16395 w_k1_FDX_s112
                                 + 0.648094 w_k1_IBM_s112
                                 + 0.951725 w_k1_INTC_s112
                                 + 0.809804 w_k1_KO_s112
                                 + 0.784928 w_k1_MAT_s112
                                 + 1.08671 w_k1_MSFT_s112
                                 + 0.922588 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.984488 w_k1_ADM_s112
                                 + 0.833809 w_k1_AXP_s112
                                 + 0.703886 w_k1_DIS_s112
                                 + 0.811615 w_k1_FDX_s112
                                 + 0.919249 w_k1_IBM_s112
                                 + 0.700494 w_k1_INTC_s112
                                 + 0.928135 w_k1_KO_s112
                                 + 0.821978 w_k1_MAT_s112
                                 + 0.869298 w_k1_MSFT_s112
                                 + 0.865909 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.15371 w_k1_ADM_s112
                                 + 1.15798 w_k1_AXP_s112
                                 + 1.62859 w_k1_DIS_s112
                                 + 1.09319 w_k1_FDX_s112
                                 + 1.67341 w_k1_IBM_s112
                                 + 0.933653 w_k1_INTC_s112
                                 + 1.36757 w_k1_KO_s112 + 1.66606 w_k1_MAT_s112
                                 + 1.17729 w_k1_MSFT_s112
                                 + 1.04135 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.954688 w_k1_ADM_s112
                                 + 0.959338 w_k1_AXP_s112
                                 + 0.820003 w_k1_DIS_s112
                                 + 1.02298 w_k1_FDX_s112
                                 + 0.550951 w_k1_IBM_s112
                                 + 1.01728 w_k1_INTC_s112
                                 + 0.714738 w_k1_KO_s112
                                 + 0.570153 w_k1_MAT_s112
                                 + 0.881211 w_k1_MSFT_s112
                                 + 0.899455 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.974633 w_k1_ADM_s112
                                 + 0.983591 w_k1_AXP_s112
                                 + 1.07391 w_k1_DIS_s112
                                 + 1.02312 w_k1_FDX_s112
                                 + 1.30956 w_k1_IBM_s112
                                 + 1.03528 w_k1_INTC_s112
                                 + 1.07903 w_k1_KO_s112
                                 + 0.844584 w_k1_MAT_s112
                                 + 1.00976 w_k1_MSFT_s112
                                 + 0.963537 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.40385 w_k1_ADM_s112
                                 + 1.25733 w_k1_AXP_s112
                                 + 0.910586 w_k1_DIS_s112
                                 + 1.01795 w_k1_FDX_s112
                                 + 0.939475 w_k1_IBM_s112
                                 + 0.812264 w_k1_INTC_s112
                                 + 0.976601 w_k1_KO_s112
                                 + 0.92318 w_k1_MAT_s112
                                 + 0.940123 w_k1_MSFT_s112
                                 + 0.965406 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.444 w_k1_ADM_s112
                                 + 1.52736 w_k1_AXP_s112
                                 + 1.23948 w_k1_DIS_s112
                                 + 1.23152 w_k1_FDX_s112
                                 + 1.09075 w_k1_IBM_s112
                                 + 1.00532 w_k1_INTC_s112
                                 + 1.14753 w_k1_KO_s112 + 1.32164 w_k1_MAT_s112
                                 + 1.0574 w_k1_MSFT_s112 + 1.10954 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.03846 w_k1_ADM_s113
                                 + 1.26306 w_k1_AXP_s113
                                 + 1.52272 w_k1_DIS_s113
                                 + 1.27722 w_k1_FDX_s113
                                 + 1.01487 w_k1_IBM_s113
                                 + 1.17497 w_k1_INTC_s113
                                 + 1.20528 w_k1_KO_s113
                                 + 0.850103 w_k1_MAT_s113
                                 + 1.14147 w_k1_MSFT_s113
                                 + 1.29507 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.30212 w_k1_ADM_s113
                                 + 1.67164 w_k1_AXP_s113
                                 + 1.46598 w_k1_DIS_s113
                                 + 1.35879 w_k1_FDX_s113
                                 + 1.58616 w_k1_IBM_s113
                                 + 1.42981 w_k1_INTC_s113
                                 + 1.31788 w_k1_KO_s113 + 1.34495 w_k1_MAT_s113
                                 + 1.36514 w_k1_MSFT_s113 + 1.4111 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.0388 w_k1_ADM_s113
                                 + 0.977674 w_k1_AXP_s113
                                 + 0.94032 w_k1_DIS_s113
                                 + 1.07806 w_k1_FDX_s113
                                 + 0.83213 w_k1_IBM_s113
                                 + 0.889689 w_k1_INTC_s113
                                 + 0.850587 w_k1_KO_s113
                                 + 0.742412 w_k1_MAT_s113
                                 + 1.00222 w_k1_MSFT_s113
                                 + 1.09907 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.647322 w_k1_ADM_s113
                                 + 0.845467 w_k1_AXP_s113
                                 + 0.952121 w_k1_DIS_s113
                                 + 0.793874 w_k1_FDX_s113
                                 + 1.33232 w_k1_IBM_s113
                                 + 0.80618 w_k1_INTC_s113
                                 + 1.00415 w_k1_KO_s113 + 1.06156 w_k1_MAT_s113
                                 + 0.755847 w_k1_MSFT_s113
                                 + 0.991281 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.10129 w_k1_ADM_s113
                                 + 1.70851 w_k1_AXP_s113
                                 + 1.77982 w_k1_DIS_s113 + 1.5727 w_k1_FDX_s113
                                 + 1.06579 w_k1_IBM_s113
                                 + 1.50095 w_k1_INTC_s113
                                 + 1.21143 w_k1_KO_s113
                                 + 0.921587 w_k1_MAT_s113
                                 + 1.26828 w_k1_MSFT_s113
                                 + 1.15754 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.865379 w_k1_ADM_s113
                                 + 1.19438 w_k1_AXP_s113
                                 + 1.05014 w_k1_DIS_s113
                                 + 0.926539 w_k1_FDX_s113
                                 + 1.47894 w_k1_IBM_s113
                                 + 1.0453 w_k1_INTC_s113 + 1.21651 w_k1_KO_s113
                                 + 1.3414 w_k1_MAT_s113
                                 + 0.887999 w_k1_MSFT_s113
                                 + 1.07018 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.624193 w_k1_ADM_s113
                                 + 0.752026 w_k1_AXP_s113
                                 + 0.581246 w_k1_DIS_s113
                                 + 0.697147 w_k1_FDX_s113
                                 + 0.980114 w_k1_IBM_s113
                                 + 0.785406 w_k1_INTC_s113
                                 + 0.756898 w_k1_KO_s113
                                 + 0.910204 w_k1_MAT_s113
                                 + 0.727773 w_k1_MSFT_s113
                                 + 0.87822 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.861013 w_k1_ADM_s113
                                 + 0.843232 w_k1_AXP_s113
                                 + 0.714257 w_k1_DIS_s113
                                 + 0.875183 w_k1_FDX_s113
                                 + 0.762031 w_k1_IBM_s113
                                 + 0.912035 w_k1_INTC_s113
                                 + 0.770649 w_k1_KO_s113
                                 + 0.568567 w_k1_MAT_s113
                                 + 0.848391 w_k1_MSFT_s113
                                 + 1.07008 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12708 w_k1_ADM_s114
                                 + 1.61908 w_k1_AXP_s114
                                 + 1.40901 w_k1_DIS_s114
                                 + 1.39612 w_k1_FDX_s114
                                 + 1.31637 w_k1_IBM_s114
                                 + 1.33924 w_k1_INTC_s114
                                 + 1.42431 w_k1_KO_s114 + 2.17303 w_k1_MAT_s114
                                 + 1.39131 w_k1_MSFT_s114
                                 + 1.50492 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.889234 w_k1_ADM_s114
                                 + 0.963104 w_k1_AXP_s114
                                 + 0.889051 w_k1_DIS_s114
                                 + 0.990333 w_k1_FDX_s114
                                 + 0.653232 w_k1_IBM_s114
                                 + 1.07327 w_k1_INTC_s114
                                 + 0.689363 w_k1_KO_s114
                                 + 0.586317 w_k1_MAT_s114
                                 + 0.966734 w_k1_MSFT_s114
                                 + 1.02782 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.655624 w_k1_ADM_s114
                                 + 0.611849 w_k1_AXP_s114
                                 + 0.63132 w_k1_DIS_s114
                                 + 0.69134 w_k1_FDX_s114
                                 + 0.815249 w_k1_IBM_s114
                                 + 0.738669 w_k1_INTC_s114
                                 + 0.715022 w_k1_KO_s114
                                 + 0.828151 w_k1_MAT_s114
                                 + 0.774904 w_k1_MSFT_s114
                                 + 0.747196 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.05291 w_k1_ADM_s114
                                 + 0.829452 w_k1_AXP_s114
                                 + 0.931773 w_k1_DIS_s114
                                 + 1.03843 w_k1_FDX_s114
                                 + 0.921576 w_k1_IBM_s114
                                 + 0.91612 w_k1_INTC_s114
                                 + 0.857633 w_k1_KO_s114
                                 + 0.89324 w_k1_MAT_s114
                                 + 1.03897 w_k1_MSFT_s114
                                 + 0.97533 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.1902 w_k1_ADM_s114
                                 + 1.23274 w_k1_AXP_s114
                                 + 1.25158 w_k1_DIS_s114
                                 + 1.25627 w_k1_FDX_s114
                                 + 0.937832 w_k1_IBM_s114
                                 + 1.24256 w_k1_INTC_s114
                                 + 0.973922 w_k1_KO_s114
                                 + 0.657249 w_k1_MAT_s114
                                 + 1.22853 w_k1_MSFT_s114
                                 + 1.16313 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.06618 w_k1_ADM_s114
                                 + 1.53282 w_k1_AXP_s114
                                 + 0.914969 w_k1_DIS_s114
                                 + 1.10239 w_k1_FDX_s114
                                 + 1.21978 w_k1_IBM_s114
                                 + 1.13555 w_k1_INTC_s114
                                 + 0.933433 w_k1_KO_s114
                                 + 1.31797 w_k1_MAT_s114
                                 + 0.979178 w_k1_MSFT_s114
                                 + 1.09351 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.06601 w_k1_ADM_s114
                                 + 0.965416 w_k1_AXP_s114
                                 + 0.786185 w_k1_DIS_s114
                                 + 0.91534 w_k1_FDX_s114
                                 + 1.25303 w_k1_IBM_s114
                                 + 0.971216 w_k1_INTC_s114
                                 + 0.982838 w_k1_KO_s114
                                 + 1.48249 w_k1_MAT_s114
                                 + 0.943273 w_k1_MSFT_s114
                                 + 1.02453 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.781784 w_k1_ADM_s114
                                 + 0.676696 w_k1_AXP_s114
                                 + 0.793291 w_k1_DIS_s114
                                 + 0.815049 w_k1_FDX_s114
                                 + 1.30919 w_k1_IBM_s114
                                 + 0.855055 w_k1_INTC_s114
                                 + 1.00969 w_k1_KO_s114
                                 + 0.770525 w_k1_MAT_s114
                                 + 0.817249 w_k1_MSFT_s114
                                 + 0.915072 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02838 w_k1_ADM_s115
                                 + 0.844812 w_k1_AXP_s115
                                 + 0.873114 w_k1_DIS_s115
                                 + 0.77845 w_k1_FDX_s115
                                 + 1.16357 w_k1_IBM_s115
                                 + 0.937534 w_k1_INTC_s115
                                 + 0.979945 w_k1_KO_s115
                                 + 1.20404 w_k1_MAT_s115
                                 + 0.893777 w_k1_MSFT_s115
                                 + 1.03764 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.22659 w_k1_ADM_s115
                                 + 0.645526 w_k1_AXP_s115
                                 + 0.729255 w_k1_DIS_s115
                                 + 0.851489 w_k1_FDX_s115
                                 + 0.718396 w_k1_IBM_s115
                                 + 0.936523 w_k1_INTC_s115
                                 + 0.791977 w_k1_KO_s115
                                 + 0.514388 w_k1_MAT_s115
                                 + 0.882054 w_k1_MSFT_s115
                                 + 0.883472 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.03425 w_k1_ADM_s115
                                 + 0.881678 w_k1_AXP_s115
                                 + 0.908197 w_k1_DIS_s115
                                 + 0.995182 w_k1_FDX_s115
                                 + 0.949477 w_k1_IBM_s115
                                 + 0.867586 w_k1_INTC_s115
                                 + 0.896881 w_k1_KO_s115
                                 + 1.25277 w_k1_MAT_s115
                                 + 1.01521 w_k1_MSFT_s115
                                 + 1.01106 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.35923 w_k1_ADM_s115
                                 + 1.03141 w_k1_AXP_s115
                                 + 1.32769 w_k1_DIS_s115
                                 + 1.02968 w_k1_FDX_s115 + 1.6548 w_k1_IBM_s115
                                 + 0.848846 w_k1_INTC_s115
                                 + 1.34828 w_k1_KO_s115 + 2.019 w_k1_MAT_s115
                                 + 1.06051 w_k1_MSFT_s115
                                 + 1.16614 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.72931 w_k1_ADM_s115
                                 + 1.35134 w_k1_AXP_s115
                                 + 1.52842 w_k1_DIS_s115
                                 + 1.42633 w_k1_FDX_s115
                                 + 1.00762 w_k1_IBM_s115
                                 + 1.03646 w_k1_INTC_s115 + 1.2322 w_k1_KO_s115
                                 + 1.43337 w_k1_MAT_s115
                                 + 1.32373 w_k1_MSFT_s115
                                 + 1.16173 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.14313 w_k1_ADM_s115
                                 + 0.95729 w_k1_AXP_s115 + 1.1717 w_k1_DIS_s115
                                 + 1.0422 w_k1_FDX_s115 + 1.12604 w_k1_IBM_s115
                                 + 1.04772 w_k1_INTC_s115
                                 + 1.30231 w_k1_KO_s115 + 1.52254 w_k1_MAT_s115
                                 + 1.07417 w_k1_MSFT_s115
                                 + 1.06846 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.29373 w_k1_ADM_s115
                                 + 0.962965 w_k1_AXP_s115
                                 + 1.36518 w_k1_DIS_s115
                                 + 1.18569 w_k1_FDX_s115
                                 + 0.858222 w_k1_IBM_s115
                                 + 1.1475 w_k1_INTC_s115 + 1.02758 w_k1_KO_s115
                                 + 1.1832 w_k1_MAT_s115
                                 + 1.25771 w_k1_MSFT_s115
                                 + 1.12686 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.22578 w_k1_ADM_s115
                                 + 1.33253 w_k1_AXP_s115
                                 + 1.19452 w_k1_DIS_s115
                                 + 1.16526 w_k1_FDX_s115
                                 + 1.10328 w_k1_IBM_s115
                                 + 0.991516 w_k1_INTC_s115
                                 + 1.08501 w_k1_KO_s115 + 1.6809 w_k1_MAT_s115
                                 + 1.01225 w_k1_MSFT_s115
                                 + 0.927288 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.71326 w_k1_ADM_s116
                                 + 1.76803 w_k1_AXP_s116
                                 + 1.46043 w_k1_DIS_s116
                                 + 1.33995 w_k1_FDX_s116
                                 + 1.17496 w_k1_IBM_s116
                                 + 1.31053 w_k1_INTC_s116
                                 + 1.16654 w_k1_KO_s116 + 1.06655 w_k1_MAT_s116
                                 + 1.28596 w_k1_MSFT_s116
                                 + 1.21316 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.49307 w_k1_ADM_s116
                                 + 1.1927 w_k1_AXP_s116 + 1.3664 w_k1_DIS_s116
                                 + 1.21805 w_k1_FDX_s116
                                 + 0.957953 w_k1_IBM_s116
                                 + 1.00618 w_k1_INTC_s116
                                 + 0.987897 w_k1_KO_s116
                                 + 0.803929 w_k1_MAT_s116
                                 + 1.30037 w_k1_MSFT_s116
                                 + 1.05526 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.29512 w_k1_ADM_s116
                                 + 1.00335 w_k1_AXP_s116
                                 + 1.29567 w_k1_DIS_s116
                                 + 1.10825 w_k1_FDX_s116
                                 + 1.44833 w_k1_IBM_s116
                                 + 1.13291 w_k1_INTC_s116
                                 + 1.39859 w_k1_KO_s116 + 1.66705 w_k1_MAT_s116
                                 + 1.30724 w_k1_MSFT_s116
                                 + 1.32144 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.20871 w_k1_ADM_s116
                                 + 0.728653 w_k1_AXP_s116
                                 + 1.06629 w_k1_DIS_s116 + 0.9851 w_k1_FDX_s116
                                 + 0.826404 w_k1_IBM_s116
                                 + 0.92836 w_k1_INTC_s116
                                 + 0.959749 w_k1_KO_s116
                                 + 0.814866 w_k1_MAT_s116
                                 + 1.13856 w_k1_MSFT_s116
                                 + 0.973364 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.856645 w_k1_ADM_s116
                                 + 0.660538 w_k1_AXP_s116
                                 + 0.684233 w_k1_DIS_s116
                                 + 0.789751 w_k1_FDX_s116
                                 + 0.792365 w_k1_IBM_s116
                                 + 1.01288 w_k1_INTC_s116
                                 + 0.728403 w_k1_KO_s116
                                 + 0.702389 w_k1_MAT_s116
                                 + 0.871908 w_k1_MSFT_s116
                                 + 0.92419 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.18854 w_k1_ADM_s116
                                 + 1.31076 w_k1_AXP_s116
                                 + 1.13113 w_k1_DIS_s116
                                 + 1.00973 w_k1_FDX_s116
                                 + 1.02539 w_k1_IBM_s116
                                 + 1.02514 w_k1_INTC_s116
                                 + 0.983425 w_k1_KO_s116 + 1.2181 w_k1_MAT_s116
                                 + 1.05329 w_k1_MSFT_s116
                                 + 0.990538 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.15866 w_k1_ADM_s116
                                 + 0.947063 w_k1_AXP_s116
                                 + 0.949721 w_k1_DIS_s116
                                 + 0.862599 w_k1_FDX_s116
                                 + 1.25511 w_k1_IBM_s116
                                 + 0.941945 w_k1_INTC_s116
                                 + 1.06034 w_k1_KO_s116 + 1.10046 w_k1_MAT_s116
                                 + 0.955153 w_k1_MSFT_s116
                                 + 1.01958 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.22707 w_k1_ADM_s116
                                 + 1.08133 w_k1_AXP_s116
                                 + 0.868261 w_k1_DIS_s116
                                 + 1.10189 w_k1_FDX_s116
                                 + 0.848722 w_k1_IBM_s116
                                 + 0.993588 w_k1_INTC_s116
                                 + 0.801518 w_k1_KO_s116
                                 + 0.715793 w_k1_MAT_s116
                                 + 0.967273 w_k1_MSFT_s116
                                 + 0.909833 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.91472 w_k1_ADM_s117
                                 + 1.38265 w_k1_AXP_s117
                                 + 1.40141 w_k1_DIS_s117
                                 + 1.16831 w_k1_FDX_s117
                                 + 1.80217 w_k1_IBM_s117
                                 + 1.27018 w_k1_INTC_s117
                                 + 1.54517 w_k1_KO_s117 + 1.70767 w_k1_MAT_s117
                                 + 1.05908 w_k1_MSFT_s117
                                 + 1.26188 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.842284 w_k1_ADM_s117
                                 + 0.976533 w_k1_AXP_s117
                                 + 0.886469 w_k1_DIS_s117
                                 + 0.852847 w_k1_FDX_s117
                                 + 1.34934 w_k1_IBM_s117
                                 + 0.935732 w_k1_INTC_s117
                                 + 1.19958 w_k1_KO_s117 + 1.38311 w_k1_MAT_s117
                                 + 0.997144 w_k1_MSFT_s117
                                 + 1.03082 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.944907 w_k1_ADM_s117
                                 + 0.825855 w_k1_AXP_s117
                                 + 0.601593 w_k1_DIS_s117
                                 + 0.828264 w_k1_FDX_s117
                                 + 0.817907 w_k1_IBM_s117
                                 + 0.80846 w_k1_INTC_s117
                                 + 0.830158 w_k1_KO_s117
                                 + 0.713495 w_k1_MAT_s117
                                 + 0.759827 w_k1_MSFT_s117
                                 + 0.880033 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.20136 w_k1_ADM_s117
                                 + 1.50344 w_k1_AXP_s117
                                 + 1.73661 w_k1_DIS_s117
                                 + 1.44478 w_k1_FDX_s117
                                 + 1.32648 w_k1_IBM_s117
                                 + 1.15798 w_k1_INTC_s117
                                 + 1.50604 w_k1_KO_s117 + 1.99141 w_k1_MAT_s117
                                 + 1.30516 w_k1_MSFT_s117
                                 + 1.41808 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.958644 w_k1_ADM_s117
                                 + 1.28262 w_k1_AXP_s117
                                 + 0.905414 w_k1_DIS_s117
                                 + 1.06478 w_k1_FDX_s117
                                 + 1.37919 w_k1_IBM_s117
                                 + 1.13422 w_k1_INTC_s117
                                 + 1.27022 w_k1_KO_s117 + 1.60857 w_k1_MAT_s117
                                 + 1.03312 w_k1_MSFT_s117
                                 + 1.13515 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.853748 w_k1_ADM_s117
                                 + 0.93294 w_k1_AXP_s117
                                 + 0.697618 w_k1_DIS_s117
                                 + 0.973528 w_k1_FDX_s117
                                 + 0.611803 w_k1_IBM_s117
                                 + 0.800748 w_k1_INTC_s117
                                 + 0.740844 w_k1_KO_s117
                                 + 0.516849 w_k1_MAT_s117
                                 + 0.902554 w_k1_MSFT_s117
                                 + 0.868926 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.855663 w_k1_ADM_s117
                                 + 1.44539 w_k1_AXP_s117
                                 + 1.25562 w_k1_DIS_s117 + 1.1764 w_k1_FDX_s117
                                 + 1.0229 w_k1_IBM_s117
                                 + 1.17428 w_k1_INTC_s117
                                 + 1.19824 w_k1_KO_s117 + 1.05501 w_k1_MAT_s117
                                 + 1.00654 w_k1_MSFT_s117
                                 + 1.05495 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.00102 w_k1_ADM_s117
                                 + 1.21035 w_k1_AXP_s117
                                 + 0.922412 w_k1_DIS_s117
                                 + 1.22921 w_k1_FDX_s117
                                 + 0.644188 w_k1_IBM_s117
                                 + 1.09613 w_k1_INTC_s117
                                 + 0.907101 w_k1_KO_s117
                                 + 0.678991 w_k1_MAT_s117
                                 + 1.02104 w_k1_MSFT_s117
                                 + 0.920648 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0241823744145413 w_k1_ADM_s000
                                 + 0.0225596460884691 w_k1_ADM_s001
                                 + 0.0176052367406686 w_k1_ADM_s002
                                 + 0.0188395623964686 w_k1_ADM_s003
                                 + 0.0165035505874193 w_k1_ADM_s004
                                 + 0.0262223310851903 w_k1_ADM_s005
                                 + 0.00825843956255687 w_k1_ADM_s006
                                 + 0.0304823520284474 w_k1_ADM_s007
                                 + 0.0228028073108255 w_k1_ADM_s010
                                 + 0.0103415211002375 w_k1_ADM_s011
                                 + 0.0455101472726577 w_k1_ADM_s012
                                 + 0.0154622600391962 w_k1_ADM_s013
                                 + 0.0385348448957771 w_k1_ADM_s014
                                 + 0.0332298108797079 w_k1_ADM_s015
                                 + 0.0394038317517185 w_k1_ADM_s016
                                 + 0.024581385322072 w_k1_ADM_s017
                                 + 0.0607792802378865 w_k1_ADM_s100
                                 + 0.0549506106479162 w_k1_ADM_s101
                                 + 0.0634558196210201 w_k1_ADM_s102
                                 + 0.129598387282972 w_k1_ADM_s103
                                 + 0.067646626246409 w_k1_ADM_s104
                                 + 0.0319629806276776 w_k1_ADM_s105
                                 + 0.0343861681267863 w_k1_ADM_s106
                                 + 0.0396349582080972 w_k1_ADM_s107
                                 + 0.0258847492393445 w_k1_ADM_s110
                                 + 0.0122067616295198 w_k1_ADM_s111
                                 + 0.00743278188348985 w_k1_ADM_s112
                                 + 0.0146433088487168 w_k1_ADM_s113
                                 + 0.0150905862627638 w_k1_ADM_s114
                                 + 0.0165320338786303 w_k1_ADM_s115
                                 + 0.0233754261844442 w_k1_ADM_s116
                                 + 0.00934882193898852 w_k1_ADM_s117
                                 + 0.0207737893794442 w_k1_AXP_s000
                                 + 0.0252412045741651 w_k1_AXP_s001
                                 + 0.0164355331550785 w_k1_AXP_s002
                                 + 0.0219504151825247 w_k1_AXP_s003
                                 + 0.0194395003340231 w_k1_AXP_s004
                                 + 0.0286324940191821 w_k1_AXP_s005
                                 + 0.00917937550739883 w_k1_AXP_s006
                                 + 0.0286155206650458 w_k1_AXP_s007
                                 + 0.0224440910773796 w_k1_AXP_s010
                                 + 0.0127202396882211 w_k1_AXP_s011
                                 + 0.0495797490655592 w_k1_AXP_s012
                                 + 0.015113032965139 w_k1_AXP_s013
                                 + 0.039798699203235 w_k1_AXP_s014
                                 + 0.0324064208613305 w_k1_AXP_s015
                                 + 0.0404707527110208 w_k1_AXP_s016
                                 + 0.0250112303468545 w_k1_AXP_s017
                                 + 0.0812808980231974 w_k1_AXP_s100
                                 + 0.0550392683867367 w_k1_AXP_s101
                                 + 0.0758970847768546 w_k1_AXP_s102
                                 + 0.125047333286422 w_k1_AXP_s103
                                 + 0.0696780129819297 w_k1_AXP_s104
                                 + 0.033019240493806 w_k1_AXP_s105
                                 + 0.0407678546643305 w_k1_AXP_s106
                                 + 0.0430668127036525 w_k1_AXP_s107
                                 + 0.0231661112687946 w_k1_AXP_s110
                                 + 0.013016829779068 w_k1_AXP_s111
                                 + 0.00695626423517892 w_k1_AXP_s112
                                 + 0.0168268711213363 w_k1_AXP_s113
                                 + 0.0157631211527194 w_k1_AXP_s114
                                 + 0.0128646288540624 w_k1_AXP_s115
                                 + 0.0194664289501436 w_k1_AXP_s116
                                 + 0.0113269023539747 w_k1_AXP_s117
                                 + 0.0200267877094129 w_k1_DIS_s000
                                 + 0.0247911085274588 w_k1_DIS_s001
                                 + 0.017779079625593 w_k1_DIS_s002
                                 + 0.02009398503054 w_k1_DIS_s003
                                 + 0.018672642800619 w_k1_DIS_s004
                                 + 0.0279292876969705 w_k1_DIS_s005
                                 + 0.0101922168493585 w_k1_DIS_s006
                                 + 0.028068211695175 w_k1_DIS_s007
                                 + 0.024534948038575 w_k1_DIS_s010
                                 + 0.0110295955706138 w_k1_DIS_s011
                                 + 0.0458963611312663 w_k1_DIS_s012
                                 + 0.0172504852976613 w_k1_DIS_s013
                                 + 0.039915042334755 w_k1_DIS_s014
                                 + 0.0323716714972839 w_k1_DIS_s015
                                 + 0.0399052236334 w_k1_DIS_s016
                                 + 0.0245651998404745 w_k1_DIS_s017
                                 + 0.069741072302085 w_k1_DIS_s100
                                 + 0.0582511354391751 w_k1_DIS_s101
                                 + 0.0761856642731913 w_k1_DIS_s102
                                 + 0.127299871939046 w_k1_DIS_s103
                                 + 0.0707821996145507 w_k1_DIS_s104
                                 + 0.0340462634873006 w_k1_DIS_s105
                                 + 0.0398486185956793 w_k1_DIS_s106
                                 + 0.0434324893762488 w_k1_DIS_s107
                                 + 0.022739899234403 w_k1_DIS_s110
                                 + 0.0133906956934865 w_k1_DIS_s111
                                 + 0.00709706840892026 w_k1_DIS_s112
                                 + 0.016240237920969 w_k1_DIS_s113
                                 + 0.0144634378708556 w_k1_DIS_s114
                                 + 0.0144015556044332 w_k1_DIS_s115
                                 + 0.0196632982834893 w_k1_DIS_s116
                                 + 0.00968161178390346 w_k1_DIS_s117
                                 + 0.02020271885719 w_k1_FDX_s000
                                 + 0.0239791087550946 w_k1_FDX_s001
                                 + 0.0174933313490474 w_k1_FDX_s002
                                 + 0.0201209807886612 w_k1_FDX_s003
                                 + 0.0179350529817648 w_k1_FDX_s004
                                 + 0.028263750427359 w_k1_FDX_s005
                                 + 0.00957471302351955 w_k1_FDX_s006
                                 + 0.0268825138237529 w_k1_FDX_s007
                                 + 0.0234787836569445 w_k1_FDX_s010
                                 + 0.0113341074099503 w_k1_FDX_s011
                                 + 0.0446379538473969 w_k1_FDX_s012
                                 + 0.0159422248190011 w_k1_FDX_s013
                                 + 0.0384486148669764 w_k1_FDX_s014
                                 + 0.0308947087990931 w_k1_FDX_s015
                                 + 0.0376711259944015 w_k1_FDX_s016
                                 + 0.0245667666208053 w_k1_FDX_s017
                                 + 0.0723365533933698 w_k1_FDX_s100
                                 + 0.0552320747703277 w_k1_FDX_s101
                                 + 0.0767985897550516 w_k1_FDX_s102
                                 + 0.121867485697306 w_k1_FDX_s103
                                 + 0.0659665896933625 w_k1_FDX_s104
                                 + 0.0326576092589928 w_k1_FDX_s105
                                 + 0.0382085103712685 w_k1_FDX_s106
                                 + 0.0411966251830465 w_k1_FDX_s107
                                 + 0.023770431649089 w_k1_FDX_s110
                                 + 0.0125676178973922 w_k1_FDX_s111
                                 + 0.0068029506644182 w_k1_FDX_s112
                                 + 0.0162805101457751 w_k1_FDX_s113
                                 + 0.0155974111971111 w_k1_FDX_s114
                                 + 0.0139108743648438 w_k1_FDX_s115
                                 + 0.0196469473109721 w_k1_FDX_s116
                                 + 0.0104161417185963 w_k1_FDX_s117
                                 + 0.0206673617387258 w_k1_IBM_s000
                                 + 0.0242100187430923 w_k1_IBM_s001
                                 + 0.0185051608345376 w_k1_IBM_s002
                                 + 0.0205164127964995 w_k1_IBM_s003
                                 + 0.0169263821477941 w_k1_IBM_s004
                                 + 0.0288035756340976 w_k1_IBM_s005
                                 + 0.0104950452926755 w_k1_IBM_s006
                                 + 0.0275701446527252 w_k1_IBM_s007
                                 + 0.0243583927440336 w_k1_IBM_s010
                                 + 0.0112173864838173 w_k1_IBM_s011
                                 + 0.0441348283085602 w_k1_IBM_s012
                                 + 0.0158048575719231 w_k1_IBM_s013
                                 + 0.0388105563311338 w_k1_IBM_s014
                                 + 0.0310102314052684 w_k1_IBM_s015
                                 + 0.040469921710822 w_k1_IBM_s016
                                 + 0.0246037646487752 w_k1_IBM_s017
                                 + 0.0728132134725551 w_k1_IBM_s100
                                 + 0.0582400093573254 w_k1_IBM_s101
                                 + 0.0755535720308832 w_k1_IBM_s102
                                 + 0.12620485521403 w_k1_IBM_s103
                                 + 0.068650903958928 w_k1_IBM_s104
                                 + 0.0319293061084443 w_k1_IBM_s105
                                 + 0.0399922110193485 w_k1_IBM_s106
                                 + 0.0431140097511973 w_k1_IBM_s107
                                 + 0.0242943561822517 w_k1_IBM_s110
                                 + 0.0130608233651029 w_k1_IBM_s111
                                 + 0.00714458320000606 w_k1_IBM_s112
                                 + 0.0168107718461671 w_k1_IBM_s113
                                 + 0.0158673569933175 w_k1_IBM_s114
                                 + 0.0134371381253198 w_k1_IBM_s115
                                 + 0.0187540458442406 w_k1_IBM_s116
                                 + 0.0104981710346586 w_k1_IBM_s117
                                 + 0.0190805779150556 w_k1_INTC_s000
                                 + 0.0243975739296081 w_k1_INTC_s001
                                 + 0.0173997234891925 w_k1_INTC_s002
                                 + 0.0212547794998016 w_k1_INTC_s003
                                 + 0.0159320665815683 w_k1_INTC_s004
                                 + 0.0282597695696692 w_k1_INTC_s005
                                 + 0.0100208729731472 w_k1_INTC_s006
                                 + 0.0252357480603357 w_k1_INTC_s007
                                 + 0.023388772319955 w_k1_INTC_s010
                                 + 0.010503612577407 w_k1_INTC_s011
                                 + 0.0439150593911847 w_k1_INTC_s012
                                 + 0.0148204362274665 w_k1_INTC_s013
                                 + 0.0322524806573702 w_k1_INTC_s014
                                 + 0.0298784879467803 w_k1_INTC_s015
                                 + 0.0351505166254518 w_k1_INTC_s016
                                 + 0.0230105319015543 w_k1_INTC_s017
                                 + 0.0721952029569644 w_k1_INTC_s100
                                 + 0.0519085920960616 w_k1_INTC_s101
                                 + 0.0896472343002005 w_k1_INTC_s102
                                 + 0.108129901276019 w_k1_INTC_s103
                                 + 0.0682738135739481 w_k1_INTC_s104
                                 + 0.0333653365652529 w_k1_INTC_s105
                                 + 0.0410636020833564 w_k1_INTC_s106
                                 + 0.043991000297506 w_k1_INTC_s107
                                 + 0.0260143364779527 w_k1_INTC_s110
                                 + 0.0123435935992129 w_k1_INTC_s111
                                 + 0.00605588629881498 w_k1_INTC_s112
                                 + 0.0157709825612766 w_k1_INTC_s113
                                 + 0.0153993026871247 w_k1_INTC_s114
                                 + 0.0131035094660878 w_k1_INTC_s115
                                 + 0.0198209271155166 w_k1_INTC_s116
                                 + 0.0100802578571075 w_k1_INTC_s117
                                 + 0.0185407590209387 w_k1_KO_s000
                                 + 0.0233156748484733 w_k1_KO_s001
                                 + 0.0184528730237994 w_k1_KO_s002
                                 + 0.0202152859604205 w_k1_KO_s003
                                 + 0.0177779202162872 w_k1_KO_s004
                                 + 0.0289118860920546 w_k1_KO_s005
                                 + 0.0104583566758772 w_k1_KO_s006
                                 + 0.0293986663188042 w_k1_KO_s007
                                 + 0.0232240549199825 w_k1_KO_s010
                                 + 0.00985751112077088 w_k1_KO_s011
                                 + 0.0422574162083321 w_k1_KO_s012
                                 + 0.0156149627490948 w_k1_KO_s013
                                 + 0.0410527551025228 w_k1_KO_s014
                                 + 0.0307419619222845 w_k1_KO_s015
                                 + 0.0396290612589135 w_k1_KO_s016
                                 + 0.0242318019731431 w_k1_KO_s017
                                 + 0.0686731121385569 w_k1_KO_s100
                                 + 0.0528386766709451 w_k1_KO_s101
                                 + 0.0748001223758507 w_k1_KO_s102
                                 + 0.117944630620959 w_k1_KO_s103
                                 + 0.067169259459664 w_k1_KO_s104
                                 + 0.0326934626254115 w_k1_KO_s105
                                 + 0.037613104970698 w_k1_KO_s106
                                 + 0.0465740233671583 w_k1_KO_s107
                                 + 0.0233968417108089 w_k1_KO_s110
                                 + 0.0114853768186653 w_k1_KO_s111
                                 + 0.00699206939294122 w_k1_KO_s112
                                 + 0.0155270861315643 w_k1_KO_s113
                                 + 0.014373817339729 w_k1_KO_s114
                                 + 0.0137186900066087 w_k1_KO_s115
                                 + 0.0182195077266471 w_k1_KO_s116
                                 + 0.0108620302828998 w_k1_KO_s117
                                 + 0.0160551067228322 w_k1_MAT_s000
                                 + 0.0241574767932175 w_k1_MAT_s001
                                 + 0.0149970558295983 w_k1_MAT_s002
                                 + 0.0247248096274627 w_k1_MAT_s003
                                 + 0.0211326634530686 w_k1_MAT_s004
                                 + 0.0237730304298496 w_k1_MAT_s005
                                 + 0.0115341020800326 w_k1_MAT_s006
                                 + 0.0281092053030468 w_k1_MAT_s007
                                 + 0.0256125315735273 w_k1_MAT_s010
                                 + 0.00991010629627478 w_k1_MAT_s011
                                 + 0.0442762463834954 w_k1_MAT_s012
                                 + 0.0182562157335854 w_k1_MAT_s013
                                 + 0.0473972151981411 w_k1_MAT_s014
                                 + 0.032163738192756 w_k1_MAT_s015
                                 + 0.051033189511647 w_k1_MAT_s016
                                 + 0.0226894601661437 w_k1_MAT_s017
                                 + 0.0751750146078479 w_k1_MAT_s100
                                 + 0.0497948359969849 w_k1_MAT_s101
                                 + 0.0835623342025309 w_k1_MAT_s102
                                 + 0.108101324542082 w_k1_MAT_s103
                                 + 0.0827600955707022 w_k1_MAT_s104
                                 + 0.0332170316291372 w_k1_MAT_s105
                                 + 0.0492939648958786 w_k1_MAT_s106
                                 + 0.0383419404258711 w_k1_MAT_s107
                                 + 0.0193800861355477 w_k1_MAT_s110
                                 + 0.0110473321786918 w_k1_MAT_s111
                                 + 0.00630124687652211 w_k1_MAT_s112
                                 + 0.0144139764535904 w_k1_MAT_s113
                                 + 0.016853100461273 w_k1_MAT_s114
                                 + 0.0164581923268494 w_k1_MAT_s115
                                 + 0.0175451032206928 w_k1_MAT_s116
                                 + 0.0108786397634502 w_k1_MAT_s117
                                 + 0.0189707514715417 w_k1_MSFT_s000
                                 + 0.0234265073222814 w_k1_MSFT_s001
                                 + 0.0174351823994652 w_k1_MSFT_s002
                                 + 0.0192613270446515 w_k1_MSFT_s003
                                 + 0.0173902390687498 w_k1_MSFT_s004
                                 + 0.0295412661084772 w_k1_MSFT_s005
                                 + 0.0114851351898848 w_k1_MSFT_s006
                                 + 0.0233767657876522 w_k1_MSFT_s007
                                 + 0.0226519204900127 w_k1_MSFT_s010
                                 + 0.0104220242441016 w_k1_MSFT_s011
                                 + 0.0447062927055492 w_k1_MSFT_s012
                                 + 0.0155628746639512 w_k1_MSFT_s013
                                 + 0.0368674365132114 w_k1_MSFT_s014
                                 + 0.02852692246749 w_k1_MSFT_s015
                                 + 0.0416764620749352 w_k1_MSFT_s016
                                 + 0.024074795580835 w_k1_MSFT_s017
                                 + 0.0700661007238584 w_k1_MSFT_s100
                                 + 0.0599568347336511 w_k1_MSFT_s101
                                 + 0.0803492052670108 w_k1_MSFT_s102
                                 + 0.113422730626496 w_k1_MSFT_s103
                                 + 0.0674542726892831 w_k1_MSFT_s104
                                 + 0.0310280733661576 w_k1_MSFT_s105
                                 + 0.0360088015992019 w_k1_MSFT_s106
                                 + 0.0384151039537455 w_k1_MSFT_s107
                                 + 0.024464075105968 w_k1_MSFT_s110
                                 + 0.0117881790302418 w_k1_MSFT_s111
                                 + 0.00662776240184699 w_k1_MSFT_s112
                                 + 0.0153116409804447 w_k1_MSFT_s113
                                 + 0.0154873407649154 w_k1_MSFT_s114
                                 + 0.0140563980819295 w_k1_MSFT_s115
                                 + 0.020509965504817 w_k1_MSFT_s116
                                 + 0.00963448074086479 w_k1_MSFT_s117
                                 + 0.0200204268018105 w_k1_PG_s000
                                 + 0.0230065423524207 w_k1_PG_s001
                                 + 0.0178677407039853 w_k1_PG_s002
                                 + 0.0189517278823459 w_k1_PG_s003
                                 + 0.0182904689515897 w_k1_PG_s004
                                 + 0.0273894840321104 w_k1_PG_s005
                                 + 0.0101075384540001 w_k1_PG_s006
                                 + 0.0262019294628258 w_k1_PG_s007
                                 + 0.0244097528065214 w_k1_PG_s010
                                 + 0.0108887280075221 w_k1_PG_s011
                                 + 0.0432563338382364 w_k1_PG_s012
                                 + 0.018258345077485 w_k1_PG_s013
                                 + 0.0398456916653119 w_k1_PG_s014
                                 + 0.030465850098018 w_k1_PG_s015
                                 + 0.0388229778979407 w_k1_PG_s016
                                 + 0.0263190692501728 w_k1_PG_s017
                                 + 0.0752614425511388 w_k1_PG_s100
                                 + 0.055936185763291 w_k1_PG_s101
                                 + 0.0765877678175885 w_k1_PG_s102
                                 + 0.118799545463011 w_k1_PG_s103
                                 + 0.0663145223571243 w_k1_PG_s104
                                 + 0.0331264010847741 w_k1_PG_s105
                                 + 0.0383926762640068 w_k1_PG_s106
                                 + 0.0398013372893497 w_k1_PG_s107
                                 + 0.0221382623673166 w_k1_PG_s110
                                 + 0.012345636327002 w_k1_PG_s111
                                 + 0.00629223517385772 w_k1_PG_s112
                                 + 0.017521429267704 w_k1_PG_s113
                                 + 0.015885697058401 w_k1_PG_s114
                                 + 0.0137962043863116 w_k1_PG_s115
                                 + 0.0195284706102801 w_k1_PG_s116
                                 + 0.0102574391090086 w_k1_PG_s117 >= 1.11
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
