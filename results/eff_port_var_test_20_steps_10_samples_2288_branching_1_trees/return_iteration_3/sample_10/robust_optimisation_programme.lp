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
 w_k1_ADM_s0_return_constraint:  1.074 w_0_ADM - w_k1_ADM_s0 + 0.99 b_k1_ADM_s0
                                 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.877661 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.942712 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.03768 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.976001 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.24775 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.10669 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.823179 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.771934 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.81961 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.36478 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.26355 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.01854 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.28745 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.863147 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.764798 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.684721 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.01179 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.12324 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.06892 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.51451 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.04896 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.16531 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.11853 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.826084 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.954904 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.15147 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.92308 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.33199 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.20006 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.28863 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.02991 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.624998 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.16267 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.07088 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.929366 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.955864 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.51957 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16521 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.812219 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.842969 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.11345 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.976326 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.31581 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.994064 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.667729 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.650144 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.869879 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.49555 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.7563 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.998397 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.31653 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.923353 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.652128 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.599674 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.873269 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.40632 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.973334 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.41404 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.25226 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.33458 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.0735 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.718277 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.730503 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.02866 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.932101 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.46014 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.27449 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.39442 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.969231 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.785586 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.32058 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.09131 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.09317 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.76818 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.47318 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.20128 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.788743 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.872239 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.17085 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.965965 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.555 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.07498 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.535144 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.584414 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.733628 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.2356 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.93738 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.98718 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.42667 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.831269 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.538782 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.614087 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.874805 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.44851 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.24291 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.2008 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.08392 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.12307 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.00812 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.667938 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.707584 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.33749 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.962616 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.6018 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.16158 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.3382 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.03526 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.680278 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.04012 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.901051 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.883453 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.709281 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.68369 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.09643 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.853618 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.977198 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07347 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.953102 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.27844 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.07605 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.719345 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.652412 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.814621 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.31158 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.50835 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.943949 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.23211 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.846773 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.662169 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.728895 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.927483 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.22209 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.13777 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.28687 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.0636 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.07862 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.20706 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.822884 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.839039 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.11121 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.881329 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.52803 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.28113 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.27728 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0798 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.713936 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.06175 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.944665 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.868304 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.85919 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.39077 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13294 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.869074 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.710928 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.21747 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.934382 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.33018 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.85697 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.609023 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.961794 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.988632 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.927523 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.33212 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.35098 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.37228 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.23906 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.01475 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.697504 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.964742 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.75461 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.802139 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.896738 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.29109 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.60544 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.631417 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.621279 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.989867 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.10272 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.50188 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.10989 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.998682 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.779639 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.601999 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.1855 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.37639 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.933441 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.79194 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.580085 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.39419 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11846 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.944553 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.943874 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.13502 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.79336 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.09454 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.944415 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.740531 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.704644 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.791125 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.33851 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.29126 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.818411 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.12737 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.812987 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.768811 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.661952 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.923409 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.989666 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01972 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.927954 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.839287 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.13925 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.18951 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.948139 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.985112 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.27956 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.06529 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.33586 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.15949 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.35373 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.07346 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.773894 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.985476 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.818097 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.10577 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.907125 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.20256 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10316 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.83294 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.844753 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.18233 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.959914 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.42608 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.931679 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.583985 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.691474 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.80868 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.12162 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.45202 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.02412 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.29054 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.99476 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.796539 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.709927 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.955195 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.34007 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.947223 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.07566 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11563 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.25501 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.839715 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.65174 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.954994 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.16927 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.20232 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.21514 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.996935 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.942744 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.841392 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.884541 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.13664 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.926071 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.26898 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.7301 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.37836 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14109 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.756743 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.649599 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.15073 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.07381 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.5296 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.917834 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.643961 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.820306 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.00658 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.28289 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.48427 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.49563 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.34707 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.39872 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.864582 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.753229 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.895063 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.03904 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.881296 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.20523 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.58165 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.80291 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.731379 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.560037 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.81755 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.849825 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.28885 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.05221 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.10809 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.753834 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.70498 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.04628 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.23909 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04549 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.47092 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.427908 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.8162 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.06728 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.857139 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.979074 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.09214 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.948217 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.29015 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00261 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.745418 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.722079 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.873727 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.24206 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.43596 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.89416 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.19361 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.912704 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.702561 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.771927 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.95337 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.19314 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.13705 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.12821 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.992947 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.07665 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.18853 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.896667 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.919339 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.21985 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.910168 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.52479 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.16088 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.19607 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.12086 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.724867 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.973594 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.91071 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.920494 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.835615 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.3333 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06746 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.902626 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.922652 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.1458 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.960655 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.18237 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0049 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.69608 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.79099 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.834666 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.16296 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.39429 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.03101 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.1726 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.869229 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.730178 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.743752 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.900159 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.08499 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.933767 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.05506 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.939344 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11902 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01864 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.795004 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.900147 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.09801 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.05609 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.22457 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.0756 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.06662 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.04376 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.913334 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.16203 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.99583 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.10727 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.814667 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.31577 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.264000090756303 z_k1_s0000
                                 - 0.618999971062928 z_k1_s0001
                                 - 0.55100163030095 z_k1_s0002
                                 - 0.938002237290587 z_k1_s0003
                                 - 0.640999434250627 z_k1_s0004
                                 - 0.800000579564736 z_k1_s0005
                                 - 0.583000849483057 z_k1_s0006
                                 - 0.628999727057337 z_k1_s0007
                                 - 0.267000305212112 z_k1_s0010
                                 - 0.110000096046904 z_k1_s0011
                                 - 0.134999630275204 z_k1_s0012
                                 - 0.0740000391733438 z_k1_s0013
                                 - 0.105000165881219 z_k1_s0014
                                 - 0.129999816709648 z_k1_s0015
                                 - 0.102000161141755 z_k1_s0016
                                 - 0.243000500496664 z_k1_s0017
                                 - 0.296001007118218 z_k1_s0020
                                 - 0.23000005039545 z_k1_s0021
                                 - 0.491000755388088 z_k1_s0022
                                 - 0.123000094255001 z_k1_s0023
                                 - 0.176000164759207 z_k1_s0024
                                 - 0.232000841660258 z_k1_s0025
                                 - 0.106000141008412 z_k1_s0026
                                 - 0.281000877635827 z_k1_s0027
                                 - 0.43399970437749 z_k1_s0030
                                 - 0.465000829261275 z_k1_s0031
                                 - 0.900998698676965 z_k1_s0032
                                 - 0.364999980577318 z_k1_s0033
                                 - 0.626999343714089 z_k1_s0034
                                 - 0.426999938345531 z_k1_s0035
                                 - 0.970001631276235 z_k1_s0036
                                 - 0.394999960142582 z_k1_s0037
                                 - 0.259999803913686 z_k1_s0040
                                 - 0.122000092605869 z_k1_s0041
                                 - 0.08800015286335 z_k1_s0042
                                 - 0.40900097808111 z_k1_s0043
                                 - 0.235000774692329 z_k1_s0044
                                 - 0.10400007156565 z_k1_s0045
                                 - 0.195000227935703 z_k1_s0046
                                 - 0.0870001016943906 z_k1_s0047
                                 - 0.169000043399149 z_k1_s0050
                                 - 0.347999520381385 z_k1_s0051
                                 - 0.0750001068754074 z_k1_s0052
                                 - 0.111000123335585 z_k1_s0053
                                 - 0.15700009597912 z_k1_s0054
                                 - 0.264000236841362 z_k1_s0055
                                 - 0.259000694249909 z_k1_s0056
                                 - 0.359000256083484 z_k1_s0057
                                 - 0.426001587384654 z_k1_s0060
                                 - 0.566001869905121 z_k1_s0061
                                 - 0.787000254024027 z_k1_s0062
                                 - 0.257000445855333 z_k1_s0063
                                 - 0.243000417603287 z_k1_s0064
                                 - 0.418999026255311 z_k1_s0065
                                 - 0.498000277250426 z_k1_s0066
                                 - 0.200000209543271 z_k1_s0067
                                 - 0.192999803368324 z_k1_s0070
                                 - 0.256999814765164 z_k1_s0071
                                 - 0.134000116262243 z_k1_s0072
                                 - 0.213000730930925 z_k1_s0073
                                 - 0.236000966026937 z_k1_s0074
                                 - 0.49000147325869 z_k1_s0075
                                 - 0.14600011839915 z_k1_s0076
                                 - 0.179000262875809 z_k1_s0077
                                 - 0.314000138279407 z_k1_s0100
                                 - 0.20199994006449 z_k1_s0101
                                 - 0.230999909487119 z_k1_s0102
                                 - 0.430998974164023 z_k1_s0103
                                 - 0.734000511710423 z_k1_s0104
                                 - 0.395999929377902 z_k1_s0105
                                 - 0.503000306323364 z_k1_s0106
                                 - 0.147999976594771 z_k1_s0107
                                 - 0.297000615274868 z_k1_s0110
                                 - 0.755999181973135 z_k1_s0111
                                 - 0.157999944671598 z_k1_s0112
                                 - 0.236999917007397 z_k1_s0113
                                 - 0.456000061634416 z_k1_s0114
                                 - 0.331001156659722 z_k1_s0115
                                 - 0.364998784045487 z_k1_s0116
                                 - 0.314000885967295 z_k1_s0117
                                 - 0.629997880828748 z_k1_s0120
                                 - 0.805994879252251 z_k1_s0121
                                 - 0.828998194284149 z_k1_s0122
                                 - 0.715998986550031 z_k1_s0123
                                 - 1.14099771547367 z_k1_s0124
                                 - 0.752997371971813 z_k1_s0125
                                 - 0.800994737547777 z_k1_s0126
                                 - 0.981998535654056 z_k1_s0127
                                 - 0.218999797210352 z_k1_s0130
                                 - 0.22199978946255 z_k1_s0131
                                 - 0.823000655985487 z_k1_s0132
                                 - 0.22199978946255 z_k1_s0133
                                 - 0.258999935772918 z_k1_s0134
                                 - 0.984998060804876 z_k1_s0135
                                 - 0.36400075299951 z_k1_s0136
                                 - 0.534001523290358 z_k1_s0137
                                 - 0.885000220218848 z_k1_s0140
                                 - 0.894997296789638 z_k1_s0141
                                 - 0.295999472797047 z_k1_s0142
                                 - 0.85899949794431 z_k1_s0143
                                 - 0.359999721228427 z_k1_s0144
                                 - 0.627998594037307 z_k1_s0145
                                 - 0.28599962182943 z_k1_s0146
                                 - 0.414999363949796 z_k1_s0147
                                 - 1.14499309530882 z_k1_s0150
                                 - 0.424998492633621 z_k1_s0151
                                 - 0.945997213513542 z_k1_s0152
                                 - 0.40899828532657 z_k1_s0153
                                 - 0.411998355934017 z_k1_s0154
                                 - 0.580997086240845 z_k1_s0155
                                 - 0.594997077210263 z_k1_s0156
                                 - 0.56499637113579 z_k1_s0157
                                 - 1.43800106436002 z_k1_s0160
                                 - 0.768000041383024 z_k1_s0161
                                 - 0.668000114944579 z_k1_s0162
                                 - 2.800001729875 z_k1_s0163
                                 - 1.02700015402648 z_k1_s0164
                                 - 1.28699503628833 z_k1_s0165
                                 - 0.706999820983675 z_k1_s0166
                                 - 0.778999351163515 z_k1_s0167
                                 - 0.28799911723174 z_k1_s0170
                                 - 0.279999460174549 z_k1_s0171
                                 - 0.797996496382119 z_k1_s0172
                                 - 1.11799915463099 z_k1_s0173
                                 - 0.610999395182391 z_k1_s0174
                                 - 0.306999326240146 z_k1_s0175
                                 - 0.3659990487316 z_k1_s0176
                                 - 0.325999125849521 z_k1_s0177
                                 - 0.430000094401419 z_k1_s1000
                                 - 0.191000701748316 z_k1_s1001
                                 - 0.936002844753832 z_k1_s1002
                                 - 0.470001310164391 z_k1_s1003
                                 - 0.213000851415929 z_k1_s1004
                                 - 0.555003461159021 z_k1_s1005
                                 - 0.461001815120754 z_k1_s1006
                                 - 0.204000664369598 z_k1_s1007
                                 - 0.313999890229918 z_k1_s1010
                                 - 0.376000300147997 z_k1_s1011
                                 - 0.282999685270879 z_k1_s1012
                                 - 0.282999685270879 z_k1_s1013
                                 - 0.400998904146972 z_k1_s1014
                                 - 0.290000746390824 z_k1_s1015
                                 - 0.0969999075168762 z_k1_s1016
                                 - 0.376000300147997 z_k1_s1017
                                 - 0.880999397397439 z_k1_s1020
                                 - 0.519998917346088 z_k1_s1021
                                 - 0.722997691890956 z_k1_s1022
                                 - 0.587999005887897 z_k1_s1023
                                 - 1.63500047915498 z_k1_s1024
                                 - 0.76600129735283 z_k1_s1025
                                 - 0.484999081741877 z_k1_s1026
                                 - 1.20100061394583 z_k1_s1027
                                 - 1.32200117812747 z_k1_s1030
                                 - 0.385000112255317 z_k1_s1031
                                 - 0.644000336019777 z_k1_s1032
                                 - 1.22599837882065 z_k1_s1033
                                 - 0.449000037126867 z_k1_s1034
                                 - 0.329000032392737 z_k1_s1035
                                 - 0.797998517242228 z_k1_s1036
                                 - 0.671000395324946 z_k1_s1037
                                 - 0.169000005240575 z_k1_s1040
                                 - 0.227999959709188 z_k1_s1041
                                 - 0.43899997931724 z_k1_s1042
                                 - 0.179999982717777 z_k1_s1043
                                 - 0.51600120165734 z_k1_s1044
                                 - 0.400001188261445 z_k1_s1045
                                 - 0.347000920416829 z_k1_s1046
                                 - 0.48099888968497 z_k1_s1047
                                 - 0.227000803634379 z_k1_s1050
                                 - 0.43299991357641 z_k1_s1051
                                 - 0.62400320263917 z_k1_s1052
                                 - 0.532000904020529 z_k1_s1053
                                 - 0.267000895744262 z_k1_s1054
                                 - 0.563001416836939 z_k1_s1055
                                 - 0.636002106628952 z_k1_s1056
                                 - 0.731002124739355 z_k1_s1057
                                 - 0.354999265261133 z_k1_s1060
                                 - 0.273999826381621 z_k1_s1061
                                 - 0.409000476246623 z_k1_s1062
                                 - 0.426998898576184 z_k1_s1063
                                 - 0.247999892751221 z_k1_s1064
                                 - 0.235999854798766 z_k1_s1065
                                 - 0.56900098227936 z_k1_s1066
                                 - 0.453999504068929 z_k1_s1067
                                 - 0.391999173682993 z_k1_s1070
                                 - 0.550000111728448 z_k1_s1071
                                 - 0.296999843693403 z_k1_s1072
                                 - 0.369000134043257 z_k1_s1073
                                 - 0.174000069812266 z_k1_s1074
                                 - 0.475998609757993 z_k1_s1075
                                 - 0.200999907893513 z_k1_s1076
                                 - 0.249000011193432 z_k1_s1077
                                 - 0.197000424570301 z_k1_s1100
                                 - 0.0930002869832029 z_k1_s1101
                                 - 0.0840002343976545 z_k1_s1102
                                 - 0.110000300844529 z_k1_s1103
                                 - 0.0900002694546868 z_k1_s1104
                                 - 0.210001226996129 z_k1_s1105
                                 - 0.351001153433597 z_k1_s1106
                                 - 0.147000987097689 z_k1_s1107
                                 - 0.17600030938465 z_k1_s1110
                                 - 0.179000119675148 z_k1_s1111
                                 - 0.0610000206764318 z_k1_s1112
                                 - 0.0750000281661067 z_k1_s1113
                                 - 0.0670000597577485 z_k1_s1114
                                 - 0.110000297990486 z_k1_s1115
                                 - 0.103000461645776 z_k1_s1116
                                 - 0.0660000113941637 z_k1_s1117
                                 - 0.1330001872803 z_k1_s1120
                                 - 0.100999448988112 z_k1_s1121
                                 - 0.0909999097709886 z_k1_s1122
                                 - 0.0909999097709886 z_k1_s1123
                                 - 0.131999818459076 z_k1_s1124
                                 - 0.103000186630559 z_k1_s1125
                                 - 0.218000189121231 z_k1_s1126
                                 - 0.0529999089479838 z_k1_s1127
                                 - 0.077999843671337 z_k1_s1130
                                 - 0.101999852647178 z_k1_s1131
                                 - 0.156000429343261 z_k1_s1132
                                 - 0.0440000411890616 z_k1_s1133
                                 - 0.0849998772059818 z_k1_s1134
                                 - 0.0470000237610271 z_k1_s1135
                                 - 0.117000136507299 z_k1_s1136
                                 - 0.113000011344561 z_k1_s1137
                                 - 0.154000365089877 z_k1_s1140
                                 - 0.413000359030564 z_k1_s1141
                                 - 0.434999923985176 z_k1_s1142
                                 - 0.278001810624549 z_k1_s1143
                                 - 0.146000279687658 z_k1_s1144
                                 - 0.418001478159322 z_k1_s1145
                                 - 0.330999544557951 z_k1_s1146
                                 - 0.261000928793274 z_k1_s1147
                                 - 0.0750001525729017 z_k1_s1150
                                 - 0.0470000419123177 z_k1_s1151
                                 - 0.076000047207139 z_k1_s1152
                                 - 0.0330000134320427 z_k1_s1153
                                 - 0.0769999418413762 z_k1_s1154
                                 - 0.0619999113267959 z_k1_s1155
                                 - 0.0869999621844297 z_k1_s1156
                                 - 0.0800001627444285 z_k1_s1157
                                 - 0.0709998132837163 z_k1_s1160
                                 - 0.179999220804694 z_k1_s1161
                                 - 0.0769997873105394 z_k1_s1162
                                 - 0.073999748597281 z_k1_s1163
                                 - 0.0939997309531539 z_k1_s1164
                                 - 0.328999213435579 z_k1_s1165
                                 - 0.0939997309531539 z_k1_s1166
                                 - 0.115000105345657 z_k1_s1167
                                 - 0.0750001566519096 z_k1_s1170
                                 - 0.0540000876293469 z_k1_s1171
                                 - 0.0620000959522079 z_k1_s1172
                                 - 0.0820003054595709 z_k1_s1173
                                 - 0.140000460150418 z_k1_s1174
                                 - 0.128999866880836 z_k1_s1175
                                 - 0.0460000793064859 z_k1_s1176
                                 - 0.0410000269296451 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.44769 w_k1_ADM_s000
                                 + 1.51236 w_k1_AXP_s000
                                 + 1.87812 w_k1_DIS_s000
                                 + 1.41166 w_k1_FDX_s000
                                 + 1.42565 w_k1_IBM_s000
                                 + 1.01224 w_k1_INTC_s000
                                 + 1.42253 w_k1_KO_s000 + 1.31204 w_k1_MAT_s000
                                 + 1.22262 w_k1_MSFT_s000
                                 + 1.23674 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.849435 w_k1_ADM_s000
                                 + 0.660784 w_k1_AXP_s000
                                 + 0.597189 w_k1_DIS_s000
                                 + 0.742111 w_k1_FDX_s000
                                 + 0.791071 w_k1_IBM_s000
                                 + 0.915804 w_k1_INTC_s000
                                 + 0.871892 w_k1_KO_s000
                                 + 0.556502 w_k1_MAT_s000
                                 + 0.877571 w_k1_MSFT_s000
                                 + 0.868484 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.986479 w_k1_ADM_s000
                                 + 0.7289 w_k1_AXP_s000
                                 + 0.777358 w_k1_DIS_s000
                                 + 0.862469 w_k1_FDX_s000
                                 + 0.790361 w_k1_IBM_s000
                                 + 0.937547 w_k1_INTC_s000
                                 + 0.948752 w_k1_KO_s000
                                 + 0.612011 w_k1_MAT_s000
                                 + 0.887477 w_k1_MSFT_s000
                                 + 0.881825 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.856285 w_k1_ADM_s000
                                 + 1.03776 w_k1_AXP_s000
                                 + 0.915584 w_k1_DIS_s000
                                 + 0.923084 w_k1_FDX_s000
                                 + 1.17593 w_k1_IBM_s000
                                 + 1.15793 w_k1_INTC_s000
                                 + 1.06651 w_k1_KO_s000 + 1.2303 w_k1_MAT_s000
                                 + 0.966971 w_k1_MSFT_s000
                                 + 0.977802 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.18585 w_k1_ADM_s000
                                 + 1.15239 w_k1_AXP_s000
                                 + 1.19439 w_k1_DIS_s000
                                 + 1.04693 w_k1_FDX_s000
                                 + 1.54613 w_k1_IBM_s000
                                 + 1.33046 w_k1_INTC_s000
                                 + 1.38803 w_k1_KO_s000 + 1.22908 w_k1_MAT_s000
                                 + 1.13486 w_k1_MSFT_s000 + 1.0975 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.937388 w_k1_ADM_s000
                                 + 1.08892 w_k1_AXP_s000
                                 + 0.911051 w_k1_DIS_s000
                                 + 1.04139 w_k1_FDX_s000
                                 + 0.812228 w_k1_IBM_s000
                                 + 1.05887 w_k1_INTC_s000
                                 + 0.902771 w_k1_KO_s000
                                 + 0.984538 w_k1_MAT_s000
                                 + 0.968253 w_k1_MSFT_s000
                                 + 0.957993 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.13901 w_k1_ADM_s000
                                 + 0.948695 w_k1_AXP_s000
                                 + 1.08394 w_k1_DIS_s000
                                 + 1.11758 w_k1_FDX_s000
                                 + 0.828111 w_k1_IBM_s000
                                 + 0.946401 w_k1_INTC_s000
                                 + 0.985188 w_k1_KO_s000
                                 + 1.03742 w_k1_MAT_s000
                                 + 1.12977 w_k1_MSFT_s000
                                 + 1.01293 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.07236 w_k1_ADM_s000
                                 + 1.48154 w_k1_AXP_s000
                                 + 1.43485 w_k1_DIS_s000
                                 + 1.32975 w_k1_FDX_s000
                                 + 1.10718 w_k1_IBM_s000
                                 + 1.46775 w_k1_INTC_s000
                                 + 1.16412 w_k1_KO_s000 + 1.13144 w_k1_MAT_s000
                                 + 1.2201 w_k1_MSFT_s000 + 1.13971 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.779206 w_k1_ADM_s001
                                 + 0.719635 w_k1_AXP_s001
                                 + 0.731316 w_k1_DIS_s001
                                 + 0.819659 w_k1_FDX_s001
                                 + 1.00469 w_k1_IBM_s001
                                 + 0.971376 w_k1_INTC_s001
                                 + 1.07328 w_k1_KO_s001 + 1.04508 w_k1_MAT_s001
                                 + 0.822338 w_k1_MSFT_s001
                                 + 0.996753 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.41815 w_k1_ADM_s001
                                 + 1.41656 w_k1_AXP_s001
                                 + 1.46967 w_k1_DIS_s001
                                 + 1.42327 w_k1_FDX_s001
                                 + 0.737544 w_k1_IBM_s001
                                 + 1.13438 w_k1_INTC_s001 + 1.1884 w_k1_KO_s001
                                 + 1.07298 w_k1_MAT_s001
                                 + 1.36464 w_k1_MSFT_s001 + 1.2642 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.26187 w_k1_ADM_s001
                                 + 1.09305 w_k1_AXP_s001
                                 + 1.15976 w_k1_DIS_s001
                                 + 1.22943 w_k1_FDX_s001
                                 + 0.919292 w_k1_IBM_s001
                                 + 0.938104 w_k1_INTC_s001
                                 + 1.11378 w_k1_KO_s001 + 1.2028 w_k1_MAT_s001
                                 + 1.02827 w_k1_MSFT_s001
                                 + 1.33651 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.50017 w_k1_ADM_s001
                                 + 1.28246 w_k1_AXP_s001
                                 + 2.14079 w_k1_DIS_s001
                                 + 1.45832 w_k1_FDX_s001
                                 + 1.36134 w_k1_IBM_s001
                                 + 1.17311 w_k1_INTC_s001
                                 + 1.50397 w_k1_KO_s001 + 1.09265 w_k1_MAT_s001
                                 + 1.3846 w_k1_MSFT_s001 + 1.3412 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.787439 w_k1_ADM_s001
                                 + 0.982145 w_k1_AXP_s001
                                 + 0.935426 w_k1_DIS_s001
                                 + 0.913854 w_k1_FDX_s001
                                 + 1.35141 w_k1_IBM_s001
                                 + 0.94098 w_k1_INTC_s001
                                 + 1.11061 w_k1_KO_s001 + 1.26297 w_k1_MAT_s001
                                 + 0.81168 w_k1_MSFT_s001
                                 + 1.18678 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.631543 w_k1_ADM_s001
                                 + 0.590185 w_k1_AXP_s001
                                 + 0.540317 w_k1_DIS_s001
                                 + 0.674036 w_k1_FDX_s001
                                 + 0.655602 w_k1_IBM_s001
                                 + 0.817101 w_k1_INTC_s001
                                 + 0.707692 w_k1_KO_s001
                                 + 0.572878 w_k1_MAT_s001
                                 + 0.850284 w_k1_MSFT_s001
                                 + 0.911698 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09776 w_k1_ADM_s001
                                 + 1.20066 w_k1_AXP_s001 + 1.5567 w_k1_DIS_s001
                                 + 1.21642 w_k1_FDX_s001
                                 + 1.29958 w_k1_IBM_s001
                                 + 1.17183 w_k1_INTC_s001
                                 + 1.38146 w_k1_KO_s001 + 1.20241 w_k1_MAT_s001
                                 + 1.13316 w_k1_MSFT_s001
                                 + 1.14474 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.871978 w_k1_ADM_s001
                                 + 0.854969 w_k1_AXP_s001
                                 + 0.954934 w_k1_DIS_s001
                                 + 1.04107 w_k1_FDX_s001
                                 + 0.729737 w_k1_IBM_s001
                                 + 1.05768 w_k1_INTC_s001
                                 + 0.947153 w_k1_KO_s001
                                 + 0.647272 w_k1_MAT_s001
                                 + 1.18663 w_k1_MSFT_s001
                                 + 1.04253 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.758599 w_k1_ADM_s002
                                 + 0.616115 w_k1_AXP_s002
                                 + 0.631098 w_k1_DIS_s002
                                 + 0.754764 w_k1_FDX_s002
                                 + 0.772427 w_k1_IBM_s002
                                 + 0.760187 w_k1_INTC_s002
                                 + 0.881438 w_k1_KO_s002
                                 + 0.760012 w_k1_MAT_s002
                                 + 0.729317 w_k1_MSFT_s002
                                 + 0.861345 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.37503 w_k1_ADM_s002
                                 + 1.34677 w_k1_AXP_s002
                                 + 1.50264 w_k1_DIS_s002
                                 + 1.31372 w_k1_FDX_s002
                                 + 1.14854 w_k1_IBM_s002
                                 + 1.15636 w_k1_INTC_s002
                                 + 1.42825 w_k1_KO_s002 + 1.33219 w_k1_MAT_s002
                                 + 1.21048 w_k1_MSFT_s002
                                 + 1.11848 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.944117 w_k1_ADM_s002
                                 + 1.07231 w_k1_AXP_s002
                                 + 0.939041 w_k1_DIS_s002
                                 + 1.04314 w_k1_FDX_s002
                                 + 1.35504 w_k1_IBM_s002
                                 + 1.02302 w_k1_INTC_s002 + 1.2032 w_k1_KO_s002
                                 + 1.31376 w_k1_MAT_s002
                                 + 0.876183 w_k1_MSFT_s002
                                 + 1.06609 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.967006 w_k1_ADM_s002
                                 + 1.10623 w_k1_AXP_s002
                                 + 1.22443 w_k1_DIS_s002 + 1.1369 w_k1_FDX_s002
                                 + 0.820731 w_k1_IBM_s002
                                 + 1.03859 w_k1_INTC_s002
                                 + 1.00504 w_k1_KO_s002
                                 + 0.746107 w_k1_MAT_s002
                                 + 1.1675 w_k1_MSFT_s002
                                 + 0.997146 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.975246 w_k1_ADM_s002
                                 + 1.33189 w_k1_AXP_s002
                                 + 0.95676 w_k1_DIS_s002
                                 + 1.14954 w_k1_FDX_s002
                                 + 0.857312 w_k1_IBM_s002
                                 + 1.057 w_k1_INTC_s002 + 0.965068 w_k1_KO_s002
                                 + 1.54841 w_k1_MAT_s002
                                 + 0.952974 w_k1_MSFT_s002
                                 + 1.04249 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.10685 w_k1_ADM_s002
                                 + 0.905861 w_k1_AXP_s002
                                 + 0.966992 w_k1_DIS_s002
                                 + 0.979523 w_k1_FDX_s002
                                 + 0.799314 w_k1_IBM_s002
                                 + 0.927327 w_k1_INTC_s002
                                 + 1.02331 w_k1_KO_s002
                                 + 0.833018 w_k1_MAT_s002
                                 + 0.973587 w_k1_MSFT_s002
                                 + 0.903645 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.880246 w_k1_ADM_s002
                                 + 0.744546 w_k1_AXP_s002
                                 + 0.718598 w_k1_DIS_s002
                                 + 0.967723 w_k1_FDX_s002
                                 + 0.524181 w_k1_IBM_s002
                                 + 1.06147 w_k1_INTC_s002
                                 + 0.812009 w_k1_KO_s002
                                 + 0.601066 w_k1_MAT_s002
                                 + 1.01489 w_k1_MSFT_s002
                                 + 0.929815 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.67993 w_k1_ADM_s002
                                 + 0.774638 w_k1_AXP_s002
                                 + 0.798328 w_k1_DIS_s002
                                 + 0.841751 w_k1_FDX_s002
                                 + 1.10699 w_k1_IBM_s002
                                 + 1.03403 w_k1_INTC_s002
                                 + 1.00727 w_k1_KO_s002 + 1.25275 w_k1_MAT_s002
                                 + 0.802741 w_k1_MSFT_s002
                                 + 0.880507 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.838323 w_k1_ADM_s003
                                 + 0.749939 w_k1_AXP_s003
                                 + 0.783984 w_k1_DIS_s003
                                 + 0.796967 w_k1_FDX_s003
                                 + 1.15288 w_k1_IBM_s003
                                 + 0.811733 w_k1_INTC_s003
                                 + 1.12233 w_k1_KO_s003 + 1.41309 w_k1_MAT_s003
                                 + 0.819418 w_k1_MSFT_s003
                                 + 0.984035 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.18817 w_k1_ADM_s003
                                 + 1.19729 w_k1_AXP_s003
                                 + 1.28309 w_k1_DIS_s003 + 1.1278 w_k1_FDX_s003
                                 + 1.52001 w_k1_IBM_s003
                                 + 1.32216 w_k1_INTC_s003
                                 + 1.37289 w_k1_KO_s003 + 2.03593 w_k1_MAT_s003
                                 + 1.18383 w_k1_MSFT_s003
                                 + 1.31244 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.0866 w_k1_ADM_s003
                                 + 1.14874 w_k1_AXP_s003
                                 + 0.991827 w_k1_DIS_s003
                                 + 1.02123 w_k1_FDX_s003
                                 + 0.855774 w_k1_IBM_s003
                                 + 1.28299 w_k1_INTC_s003
                                 + 0.990271 w_k1_KO_s003
                                 + 1.01643 w_k1_MAT_s003
                                 + 0.967254 w_k1_MSFT_s003
                                 + 0.93432 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.27642 w_k1_ADM_s003
                                 + 1.21454 w_k1_AXP_s003 + 1.6737 w_k1_DIS_s003
                                 + 1.35362 w_k1_FDX_s003
                                 + 1.03977 w_k1_IBM_s003
                                 + 1.20387 w_k1_INTC_s003
                                 + 1.31517 w_k1_KO_s003 + 1.12018 w_k1_MAT_s003
                                 + 1.25777 w_k1_MSFT_s003
                                 + 1.22133 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.16189 w_k1_ADM_s003
                                 + 1.07946 w_k1_AXP_s003
                                 + 1.08618 w_k1_DIS_s003
                                 + 1.11087 w_k1_FDX_s003
                                 + 0.975135 w_k1_IBM_s003
                                 + 1.10348 w_k1_INTC_s003
                                 + 1.12456 w_k1_KO_s003 + 1.17896 w_k1_MAT_s003
                                 + 1.03426 w_k1_MSFT_s003 + 1.1724 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.978003 w_k1_ADM_s003
                                 + 0.994237 w_k1_AXP_s003
                                 + 0.950212 w_k1_DIS_s003
                                 + 0.887579 w_k1_FDX_s003
                                 + 1.45547 w_k1_IBM_s003
                                 + 1.06491 w_k1_INTC_s003
                                 + 1.23639 w_k1_KO_s003 + 1.20217 w_k1_MAT_s003
                                 + 0.864937 w_k1_MSFT_s003
                                 + 1.06325 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.973257 w_k1_ADM_s003
                                 + 0.791875 w_k1_AXP_s003
                                 + 0.763901 w_k1_DIS_s003
                                 + 0.816682 w_k1_FDX_s003
                                 + 0.741132 w_k1_IBM_s003
                                 + 0.793802 w_k1_INTC_s003
                                 + 0.840558 w_k1_KO_s003
                                 + 0.707883 w_k1_MAT_s003
                                 + 0.869256 w_k1_MSFT_s003
                                 + 0.949737 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.5138 w_k1_ADM_s003
                                 + 1.47945 w_k1_AXP_s003
                                 + 1.37178 w_k1_DIS_s003
                                 + 1.29041 w_k1_FDX_s003 + 1.0718 w_k1_IBM_s003
                                 + 1.44406 w_k1_INTC_s003
                                 + 1.11453 w_k1_KO_s003 + 1.32859 w_k1_MAT_s003
                                 + 0.993687 w_k1_MSFT_s003
                                 + 1.28888 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.912826 w_k1_ADM_s004
                                 + 0.749048 w_k1_AXP_s004
                                 + 0.921158 w_k1_DIS_s004
                                 + 1.03586 w_k1_FDX_s004
                                 + 0.628177 w_k1_IBM_s004
                                 + 0.816525 w_k1_INTC_s004
                                 + 0.806675 w_k1_KO_s004
                                 + 0.581761 w_k1_MAT_s004
                                 + 1.04841 w_k1_MSFT_s004
                                 + 0.913385 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.27742 w_k1_ADM_s004
                                 + 1.74484 w_k1_AXP_s004
                                 + 1.49791 w_k1_DIS_s004
                                 + 1.22586 w_k1_FDX_s004
                                 + 1.47358 w_k1_IBM_s004
                                 + 1.1014 w_k1_INTC_s004 + 1.58023 w_k1_KO_s004
                                 + 1.79485 w_k1_MAT_s004
                                 + 1.2372 w_k1_MSFT_s004 + 1.24074 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.843783 w_k1_ADM_s004
                                 + 0.760949 w_k1_AXP_s004
                                 + 1.12612 w_k1_DIS_s004
                                 + 0.889733 w_k1_FDX_s004
                                 + 0.904103 w_k1_IBM_s004
                                 + 0.976226 w_k1_INTC_s004
                                 + 1.00434 w_k1_KO_s004
                                 + 0.914518 w_k1_MAT_s004
                                 + 0.954822 w_k1_MSFT_s004
                                 + 1.02826 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.02201 w_k1_ADM_s004
                                 + 1.13996 w_k1_AXP_s004
                                 + 1.22236 w_k1_DIS_s004
                                 + 1.13511 w_k1_FDX_s004
                                 + 1.14317 w_k1_IBM_s004
                                 + 0.854914 w_k1_INTC_s004
                                 + 0.953423 w_k1_KO_s004
                                 + 0.955533 w_k1_MAT_s004
                                 + 0.933734 w_k1_MSFT_s004
                                 + 0.946532 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.995656 w_k1_ADM_s004
                                 + 1.13611 w_k1_AXP_s004
                                 + 0.93506 w_k1_DIS_s004
                                 + 0.98333 w_k1_FDX_s004
                                 + 0.800344 w_k1_IBM_s004
                                 + 0.942032 w_k1_INTC_s004
                                 + 0.879785 w_k1_KO_s004
                                 + 0.961306 w_k1_MAT_s004
                                 + 0.960155 w_k1_MSFT_s004
                                 + 0.963143 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.939012 w_k1_ADM_s004
                                 + 1.40256 w_k1_AXP_s004
                                 + 1.79321 w_k1_DIS_s004 + 1.4431 w_k1_FDX_s004
                                 + 1.10511 w_k1_IBM_s004
                                 + 1.12542 w_k1_INTC_s004 + 1.1669 w_k1_KO_s004
                                 + 1.02671 w_k1_MAT_s004
                                 + 1.31651 w_k1_MSFT_s004
                                 + 0.94208 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.780101 w_k1_ADM_s004
                                 + 0.805985 w_k1_AXP_s004
                                 + 0.714171 w_k1_DIS_s004
                                 + 0.753418 w_k1_FDX_s004
                                 + 0.912683 w_k1_IBM_s004
                                 + 0.665279 w_k1_INTC_s004
                                 + 0.748683 w_k1_KO_s004
                                 + 0.883626 w_k1_MAT_s004
                                 + 0.69609 w_k1_MSFT_s004
                                 + 0.889026 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.907877 w_k1_ADM_s004
                                 + 1.21855 w_k1_AXP_s004
                                 + 0.913675 w_k1_DIS_s004
                                 + 0.925625 w_k1_FDX_s004
                                 + 1.68193 w_k1_IBM_s004
                                 + 0.802024 w_k1_INTC_s004
                                 + 1.05664 w_k1_KO_s004 + 1.56018 w_k1_MAT_s004
                                 + 0.889805 w_k1_MSFT_s004
                                 + 1.01165 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.35956 w_k1_ADM_s005
                                 + 1.18171 w_k1_AXP_s005
                                 + 1.02312 w_k1_DIS_s005
                                 + 0.987727 w_k1_FDX_s005
                                 + 1.29716 w_k1_IBM_s005
                                 + 1.06219 w_k1_INTC_s005
                                 + 1.10868 w_k1_KO_s005 + 1.9775 w_k1_MAT_s005
                                 + 1.02934 w_k1_MSFT_s005
                                 + 0.998309 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.48352 w_k1_ADM_s005
                                 + 1.11876 w_k1_AXP_s005
                                 + 1.29712 w_k1_DIS_s005
                                 + 1.10936 w_k1_FDX_s005
                                 + 1.37589 w_k1_IBM_s005
                                 + 1.03607 w_k1_INTC_s005
                                 + 1.34275 w_k1_KO_s005 + 1.76009 w_k1_MAT_s005
                                 + 1.18552 w_k1_MSFT_s005
                                 + 1.21371 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.11172 w_k1_ADM_s005
                                 + 1.19975 w_k1_AXP_s005
                                 + 0.767632 w_k1_DIS_s005
                                 + 1.12401 w_k1_FDX_s005
                                 + 0.742481 w_k1_IBM_s005
                                 + 1.06886 w_k1_INTC_s005
                                 + 0.805434 w_k1_KO_s005
                                 + 0.97247 w_k1_MAT_s005
                                 + 0.968535 w_k1_MSFT_s005
                                 + 0.931385 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.46489 w_k1_ADM_s005
                                 + 1.4617 w_k1_AXP_s005 + 1.78998 w_k1_DIS_s005
                                 + 1.60911 w_k1_FDX_s005
                                 + 1.17379 w_k1_IBM_s005
                                 + 1.73148 w_k1_INTC_s005
                                 + 1.52526 w_k1_KO_s005 + 1.22053 w_k1_MAT_s005
                                 + 1.40141 w_k1_MSFT_s005
                                 + 1.36447 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.983891 w_k1_ADM_s005
                                 + 0.77121 w_k1_AXP_s005
                                 + 0.966502 w_k1_DIS_s005
                                 + 0.940166 w_k1_FDX_s005
                                 + 0.809917 w_k1_IBM_s005
                                 + 0.997489 w_k1_INTC_s005
                                 + 0.961341 w_k1_KO_s005
                                 + 1.14926 w_k1_MAT_s005
                                 + 0.991839 w_k1_MSFT_s005
                                 + 0.942939 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.00004 w_k1_ADM_s005
                                 + 0.737789 w_k1_AXP_s005
                                 + 0.685246 w_k1_DIS_s005
                                 + 0.766068 w_k1_FDX_s005
                                 + 0.718545 w_k1_IBM_s005
                                 + 0.750162 w_k1_INTC_s005
                                 + 0.787828 w_k1_KO_s005
                                 + 0.65659 w_k1_MAT_s005
                                 + 0.795834 w_k1_MSFT_s005
                                 + 0.843641 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.38382 w_k1_ADM_s005
                                 + 1.09398 w_k1_AXP_s005
                                 + 1.04205 w_k1_DIS_s005
                                 + 1.23937 w_k1_FDX_s005
                                 + 0.671201 w_k1_IBM_s005
                                 + 1.12735 w_k1_INTC_s005
                                 + 0.917251 w_k1_KO_s005
                                 + 0.898261 w_k1_MAT_s005
                                 + 1.38746 w_k1_MSFT_s005
                                 + 0.991796 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.972901 w_k1_ADM_s005
                                 + 0.844786 w_k1_AXP_s005
                                 + 0.834557 w_k1_DIS_s005
                                 + 0.838253 w_k1_FDX_s005
                                 + 1.31649 w_k1_IBM_s005
                                 + 0.922389 w_k1_INTC_s005
                                 + 1.0268 w_k1_KO_s005 + 1.36484 w_k1_MAT_s005
                                 + 0.975675 w_k1_MSFT_s005
                                 + 0.852342 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.752252 w_k1_ADM_s006
                                 + 0.783567 w_k1_AXP_s006
                                 + 0.581014 w_k1_DIS_s006
                                 + 0.740096 w_k1_FDX_s006
                                 + 0.765352 w_k1_IBM_s006
                                 + 0.935813 w_k1_INTC_s006
                                 + 0.85278 w_k1_KO_s006
                                 + 0.637628 w_k1_MAT_s006
                                 + 0.917618 w_k1_MSFT_s006
                                 + 0.932885 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.998433 w_k1_ADM_s006
                                 + 0.914604 w_k1_AXP_s006
                                 + 0.931018 w_k1_DIS_s006
                                 + 0.925997 w_k1_FDX_s006
                                 + 0.977031 w_k1_IBM_s006
                                 + 1.1985 w_k1_INTC_s006 + 1.12331 w_k1_KO_s006
                                 + 0.81102 w_k1_MAT_s006
                                 + 1.05484 w_k1_MSFT_s006
                                 + 0.932603 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.35051 w_k1_ADM_s006
                                 + 1.52314 w_k1_AXP_s006
                                 + 1.32447 w_k1_DIS_s006
                                 + 1.21865 w_k1_FDX_s006
                                 + 1.20281 w_k1_IBM_s006
                                 + 1.32858 w_k1_INTC_s006 + 1.196 w_k1_KO_s006
                                 + 1.04134 w_k1_MAT_s006
                                 + 1.12145 w_k1_MSFT_s006
                                 + 1.12475 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.21183 w_k1_ADM_s006
                                 + 0.902922 w_k1_AXP_s006
                                 + 1.26266 w_k1_DIS_s006
                                 + 1.16855 w_k1_FDX_s006
                                 + 0.82636 w_k1_IBM_s006
                                 + 1.31609 w_k1_INTC_s006
                                 + 1.15652 w_k1_KO_s006
                                 + 0.673243 w_k1_MAT_s006
                                 + 1.51179 w_k1_MSFT_s006
                                 + 1.08264 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.839926 w_k1_ADM_s006
                                 + 0.96439 w_k1_AXP_s006
                                 + 0.981308 w_k1_DIS_s006
                                 + 0.848172 w_k1_FDX_s006
                                 + 1.20835 w_k1_IBM_s006
                                 + 0.885017 w_k1_INTC_s006
                                 + 1.24885 w_k1_KO_s006 + 1.16861 w_k1_MAT_s006
                                 + 1.11029 w_k1_MSFT_s006
                                 + 1.06755 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.905403 w_k1_ADM_s006
                                 + 0.956183 w_k1_AXP_s006
                                 + 0.881299 w_k1_DIS_s006
                                 + 1.02112 w_k1_FDX_s006
                                 + 0.649227 w_k1_IBM_s006
                                 + 1.12943 w_k1_INTC_s006
                                 + 0.885966 w_k1_KO_s006
                                 + 0.722064 w_k1_MAT_s006
                                 + 1.00489 w_k1_MSFT_s006
                                 + 0.957932 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.12573 w_k1_ADM_s006
                                 + 1.27387 w_k1_AXP_s006
                                 + 1.07226 w_k1_DIS_s006 + 0.9916 w_k1_FDX_s006
                                 + 1.59305 w_k1_IBM_s006
                                 + 1.32417 w_k1_INTC_s006
                                 + 1.37067 w_k1_KO_s006 + 1.59904 w_k1_MAT_s006
                                 + 1.20317 w_k1_MSFT_s006
                                 + 1.14971 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.728819 w_k1_ADM_s006
                                 + 0.996817 w_k1_AXP_s006
                                 + 0.654068 w_k1_DIS_s006
                                 + 0.814196 w_k1_FDX_s006
                                 + 0.975634 w_k1_IBM_s006
                                 + 1.07508 w_k1_INTC_s006
                                 + 1.08499 w_k1_KO_s006 + 1.17617 w_k1_MAT_s006
                                 + 0.84915 w_k1_MSFT_s006
                                 + 0.927056 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.861241 w_k1_ADM_s007
                                 + 1.01468 w_k1_AXP_s007
                                 + 0.785179 w_k1_DIS_s007
                                 + 0.901553 w_k1_FDX_s007
                                 + 0.905312 w_k1_IBM_s007
                                 + 0.965905 w_k1_INTC_s007
                                 + 0.812575 w_k1_KO_s007
                                 + 1.09432 w_k1_MAT_s007
                                 + 0.934406 w_k1_MSFT_s007
                                 + 0.855741 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.09003 w_k1_ADM_s007
                                 + 0.854594 w_k1_AXP_s007
                                 + 0.985478 w_k1_DIS_s007
                                 + 1.13697 w_k1_FDX_s007 + 0.7107 w_k1_IBM_s007
                                 + 1.00529 w_k1_INTC_s007
                                 + 0.944485 w_k1_KO_s007
                                 + 0.799843 w_k1_MAT_s007
                                 + 1.16777 w_k1_MSFT_s007
                                 + 0.982896 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.39527 w_k1_ADM_s007
                                 + 1.29007 w_k1_AXP_s007
                                 + 1.61018 w_k1_DIS_s007
                                 + 1.41887 w_k1_FDX_s007
                                 + 0.901569 w_k1_IBM_s007
                                 + 1.04513 w_k1_INTC_s007 + 1.1728 w_k1_KO_s007
                                 + 0.702057 w_k1_MAT_s007
                                 + 1.10937 w_k1_MSFT_s007
                                 + 1.10582 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.807526 w_k1_ADM_s007
                                 + 0.939688 w_k1_AXP_s007
                                 + 0.869208 w_k1_DIS_s007
                                 + 0.872043 w_k1_FDX_s007
                                 + 1.39118 w_k1_IBM_s007
                                 + 0.864244 w_k1_INTC_s007
                                 + 1.00195 w_k1_KO_s007 + 1.23488 w_k1_MAT_s007
                                 + 0.828804 w_k1_MSFT_s007
                                 + 0.937795 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.80373 w_k1_ADM_s007
                                 + 0.854336 w_k1_AXP_s007
                                 + 0.794962 w_k1_DIS_s007
                                 + 0.851672 w_k1_FDX_s007
                                 + 0.804272 w_k1_IBM_s007
                                 + 0.894539 w_k1_INTC_s007
                                 + 0.84603 w_k1_KO_s007
                                 + 0.959751 w_k1_MAT_s007
                                 + 0.948276 w_k1_MSFT_s007
                                 + 0.94081 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.14606 w_k1_ADM_s007
                                 + 1.04278 w_k1_AXP_s007
                                 + 1.09557 w_k1_DIS_s007
                                 + 1.09757 w_k1_FDX_s007
                                 + 1.12974 w_k1_IBM_s007
                                 + 1.16675 w_k1_INTC_s007 + 1.1121 w_k1_KO_s007
                                 + 1.45504 w_k1_MAT_s007
                                 + 1.08527 w_k1_MSFT_s007
                                 + 1.05751 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09603 w_k1_ADM_s007
                                 + 1.51784 w_k1_AXP_s007
                                 + 1.44773 w_k1_DIS_s007
                                 + 1.27681 w_k1_FDX_s007
                                 + 1.28886 w_k1_IBM_s007
                                 + 1.06992 w_k1_INTC_s007
                                 + 1.18724 w_k1_KO_s007 + 2.09795 w_k1_MAT_s007
                                 + 1.10667 w_k1_MSFT_s007
                                 + 1.20962 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.85862 w_k1_ADM_s007
                                 + 0.594438 w_k1_AXP_s007
                                 + 0.648459 w_k1_DIS_s007
                                 + 0.796967 w_k1_FDX_s007
                                 + 0.647809 w_k1_IBM_s007
                                 + 0.855414 w_k1_INTC_s007
                                 + 0.765089 w_k1_KO_s007
                                 + 0.459096 w_k1_MAT_s007
                                 + 0.866537 w_k1_MSFT_s007
                                 + 0.889169 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.911356 w_k1_ADM_s010
                                 + 0.980061 w_k1_AXP_s010
                                 + 1.19996 w_k1_DIS_s010
                                 + 1.03272 w_k1_FDX_s010 + 1.4932 w_k1_IBM_s010
                                 + 1.06601 w_k1_INTC_s010
                                 + 1.38297 w_k1_KO_s010 + 1.47362 w_k1_MAT_s010
                                 + 1.18995 w_k1_MSFT_s010
                                 + 1.15163 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.1418 w_k1_ADM_s010
                                 + 1.27471 w_k1_AXP_s010
                                 + 1.30713 w_k1_DIS_s010
                                 + 1.17388 w_k1_FDX_s010
                                 + 1.07845 w_k1_IBM_s010
                                 + 1.21736 w_k1_INTC_s010
                                 + 1.22293 w_k1_KO_s010 + 1.35809 w_k1_MAT_s010
                                 + 1.10414 w_k1_MSFT_s010
                                 + 1.14319 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.11941 w_k1_ADM_s010
                                 + 1.27189 w_k1_AXP_s010
                                 + 1.60729 w_k1_DIS_s010
                                 + 1.20732 w_k1_FDX_s010
                                 + 1.26513 w_k1_IBM_s010
                                 + 1.24641 w_k1_INTC_s010
                                 + 1.28169 w_k1_KO_s010 + 1.274 w_k1_MAT_s010
                                 + 1.27098 w_k1_MSFT_s010
                                 + 1.16558 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10663 w_k1_ADM_s010
                                 + 0.80416 w_k1_AXP_s010
                                 + 0.818851 w_k1_DIS_s010
                                 + 0.836224 w_k1_FDX_s010
                                 + 1.21248 w_k1_IBM_s010
                                 + 0.688562 w_k1_INTC_s010
                                 + 0.85453 w_k1_KO_s010 + 1.06552 w_k1_MAT_s010
                                 + 0.811596 w_k1_MSFT_s010
                                 + 0.961149 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.899591 w_k1_ADM_s010
                                 + 0.683698 w_k1_AXP_s010
                                 + 0.746995 w_k1_DIS_s010
                                 + 0.826046 w_k1_FDX_s010
                                 + 0.771982 w_k1_IBM_s010
                                 + 0.931323 w_k1_INTC_s010
                                 + 0.787408 w_k1_KO_s010
                                 + 0.626374 w_k1_MAT_s010
                                 + 0.79496 w_k1_MSFT_s010
                                 + 0.963646 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.287 w_k1_ADM_s010
                                 + 1.08866 w_k1_AXP_s010
                                 + 1.37521 w_k1_DIS_s010 + 1.1729 w_k1_FDX_s010
                                 + 0.935644 w_k1_IBM_s010
                                 + 1.09404 w_k1_INTC_s010
                                 + 1.19617 w_k1_KO_s010
                                 + 0.875233 w_k1_MAT_s010
                                 + 1.19899 w_k1_MSFT_s010 + 1.0139 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.09448 w_k1_ADM_s010
                                 + 0.903546 w_k1_AXP_s010
                                 + 1.02771 w_k1_DIS_s010
                                 + 1.04536 w_k1_FDX_s010 + 0.8952 w_k1_IBM_s010
                                 + 0.941909 w_k1_INTC_s010
                                 + 0.943651 w_k1_KO_s010
                                 + 0.760156 w_k1_MAT_s010
                                 + 0.91549 w_k1_MSFT_s010
                                 + 0.986847 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.875093 w_k1_ADM_s010
                                 + 1.09175 w_k1_AXP_s010
                                 + 0.858627 w_k1_DIS_s010
                                 + 0.899759 w_k1_FDX_s010
                                 + 0.959026 w_k1_IBM_s010
                                 + 0.900265 w_k1_INTC_s010
                                 + 0.90529 w_k1_KO_s010 + 1.04325 w_k1_MAT_s010
                                 + 0.859651 w_k1_MSFT_s010
                                 + 1.02062 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.952354 w_k1_ADM_s011
                                 + 0.781979 w_k1_AXP_s011
                                 + 0.720413 w_k1_DIS_s011
                                 + 0.798078 w_k1_FDX_s011
                                 + 0.747794 w_k1_IBM_s011
                                 + 0.776542 w_k1_INTC_s011
                                 + 0.774277 w_k1_KO_s011
                                 + 0.768589 w_k1_MAT_s011
                                 + 0.878961 w_k1_MSFT_s011
                                 + 0.915949 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.818789 w_k1_ADM_s011
                                 + 0.960302 w_k1_AXP_s011
                                 + 1.00666 w_k1_DIS_s011
                                 + 0.978494 w_k1_FDX_s011
                                 + 1.0023 w_k1_IBM_s011
                                 + 0.914875 w_k1_INTC_s011
                                 + 0.986783 w_k1_KO_s011
                                 + 1.30473 w_k1_MAT_s011
                                 + 1.00251 w_k1_MSFT_s011
                                 + 1.02023 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.07551 w_k1_ADM_s011
                                 + 0.8209 w_k1_AXP_s011 + 1.3813 w_k1_DIS_s011
                                 + 1.11934 w_k1_FDX_s011
                                 + 0.880111 w_k1_IBM_s011
                                 + 1.10824 w_k1_INTC_s011
                                 + 1.06918 w_k1_KO_s011
                                 + 0.764632 w_k1_MAT_s011
                                 + 1.11067 w_k1_MSFT_s011
                                 + 1.09308 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.781781 w_k1_ADM_s011
                                 + 0.63911 w_k1_AXP_s011
                                 + 0.671397 w_k1_DIS_s011
                                 + 0.610139 w_k1_FDX_s011
                                 + 0.934616 w_k1_IBM_s011
                                 + 0.619589 w_k1_INTC_s011
                                 + 0.750346 w_k1_KO_s011
                                 + 1.07861 w_k1_MAT_s011
                                 + 0.785467 w_k1_MSFT_s011
                                 + 0.880629 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.11073 w_k1_ADM_s011
                                 + 1.7645 w_k1_AXP_s011 + 1.44046 w_k1_DIS_s011
                                 + 1.28372 w_k1_FDX_s011
                                 + 1.10672 w_k1_IBM_s011
                                 + 0.984435 w_k1_INTC_s011
                                 + 1.06949 w_k1_KO_s011 + 1.70208 w_k1_MAT_s011
                                 + 1.03027 w_k1_MSFT_s011
                                 + 1.18852 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.09582 w_k1_ADM_s011
                                 + 1.27094 w_k1_AXP_s011
                                 + 1.61868 w_k1_DIS_s011
                                 + 1.25263 w_k1_FDX_s011 + 1.1499 w_k1_IBM_s011
                                 + 1.25302 w_k1_INTC_s011
                                 + 1.38305 w_k1_KO_s011 + 1.18265 w_k1_MAT_s011
                                 + 1.31687 w_k1_MSFT_s011
                                 + 1.20896 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.13137 w_k1_ADM_s011
                                 + 1.14561 w_k1_AXP_s011
                                 + 1.27076 w_k1_DIS_s011
                                 + 0.948748 w_k1_FDX_s011
                                 + 1.50121 w_k1_IBM_s011
                                 + 0.781797 w_k1_INTC_s011
                                 + 1.19643 w_k1_KO_s011 + 1.6174 w_k1_MAT_s011
                                 + 1.01113 w_k1_MSFT_s011
                                 + 1.18295 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09978 w_k1_ADM_s011
                                 + 1.10685 w_k1_AXP_s011
                                 + 0.915309 w_k1_DIS_s011
                                 + 1.06251 w_k1_FDX_s011
                                 + 0.712733 w_k1_IBM_s011
                                 + 0.951327 w_k1_INTC_s011
                                 + 0.779389 w_k1_KO_s011
                                 + 0.774616 w_k1_MAT_s011
                                 + 1.07881 w_k1_MSFT_s011
                                 + 1.04155 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.917411 w_k1_ADM_s012
                                 + 0.959315 w_k1_AXP_s012
                                 + 0.842224 w_k1_DIS_s012
                                 + 0.844667 w_k1_FDX_s012
                                 + 1.28004 w_k1_IBM_s012
                                 + 1.11358 w_k1_INTC_s012
                                 + 1.11152 w_k1_KO_s012 + 1.11564 w_k1_MAT_s012
                                 + 0.908407 w_k1_MSFT_s012
                                 + 1.02869 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.13515 w_k1_ADM_s012
                                 + 1.44442 w_k1_AXP_s012
                                 + 1.27149 w_k1_DIS_s012
                                 + 0.982945 w_k1_FDX_s012
                                 + 1.5539 w_k1_IBM_s012
                                 + 1.13435 w_k1_INTC_s012
                                 + 1.19564 w_k1_KO_s012 + 1.54813 w_k1_MAT_s012
                                 + 1.01055 w_k1_MSFT_s012
                                 + 1.10938 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.1097 w_k1_ADM_s012
                                 + 0.78047 w_k1_AXP_s012
                                 + 0.79012 w_k1_DIS_s012
                                 + 0.909523 w_k1_FDX_s012
                                 + 0.559892 w_k1_IBM_s012
                                 + 1.00229 w_k1_INTC_s012
                                 + 0.779683 w_k1_KO_s012
                                 + 0.562761 w_k1_MAT_s012
                                 + 0.94209 w_k1_MSFT_s012
                                 + 0.907227 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.894432 w_k1_ADM_s012
                                 + 0.737627 w_k1_AXP_s012
                                 + 0.72606 w_k1_DIS_s012
                                 + 0.770972 w_k1_FDX_s012
                                 + 1.02216 w_k1_IBM_s012
                                 + 0.764281 w_k1_INTC_s012
                                 + 0.858676 w_k1_KO_s012
                                 + 0.993416 w_k1_MAT_s012
                                 + 0.745385 w_k1_MSFT_s012
                                 + 0.857697 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.18985 w_k1_ADM_s012
                                 + 1.13818 w_k1_AXP_s012
                                 + 0.908833 w_k1_DIS_s012
                                 + 1.05081 w_k1_FDX_s012
                                 + 0.871141 w_k1_IBM_s012
                                 + 1.05549 w_k1_INTC_s012
                                 + 0.965599 w_k1_KO_s012
                                 + 0.980287 w_k1_MAT_s012
                                 + 0.925871 w_k1_MSFT_s012 + 1.032 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.11947 w_k1_ADM_s012
                                 + 1.00438 w_k1_AXP_s012
                                 + 0.719131 w_k1_DIS_s012
                                 + 0.934846 w_k1_FDX_s012
                                 + 0.9275 w_k1_IBM_s012
                                 + 0.895176 w_k1_INTC_s012
                                 + 0.742607 w_k1_KO_s012
                                 + 0.92396 w_k1_MAT_s012
                                 + 0.788615 w_k1_MSFT_s012
                                 + 0.877686 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.26304 w_k1_ADM_s012
                                 + 0.991022 w_k1_AXP_s012
                                 + 1.24839 w_k1_DIS_s012
                                 + 1.05766 w_k1_FDX_s012
                                 + 1.05908 w_k1_IBM_s012
                                 + 1.06479 w_k1_INTC_s012
                                 + 1.08691 w_k1_KO_s012 + 1.0136 w_k1_MAT_s012
                                 + 1.07967 w_k1_MSFT_s012
                                 + 1.15911 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.56036 w_k1_ADM_s012
                                 + 1.54877 w_k1_AXP_s012
                                 + 1.50547 w_k1_DIS_s012
                                 + 1.42615 w_k1_FDX_s012 + 1.056 w_k1_IBM_s012
                                 + 1.35028 w_k1_INTC_s012
                                 + 1.20222 w_k1_KO_s012
                                 + 0.982204 w_k1_MAT_s012
                                 + 1.38763 w_k1_MSFT_s012
                                 + 1.28011 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.243 w_k1_ADM_s013
                                 + 1.39607 w_k1_AXP_s013
                                 + 1.68802 w_k1_DIS_s013
                                 + 1.28917 w_k1_FDX_s013
                                 + 1.39062 w_k1_IBM_s013
                                 + 1.23424 w_k1_INTC_s013 + 1.283 w_k1_KO_s013
                                 + 1.35773 w_k1_MAT_s013
                                 + 1.2367 w_k1_MSFT_s013 + 1.28012 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.802588 w_k1_ADM_s013
                                 + 1.155 w_k1_AXP_s013 + 1.10726 w_k1_DIS_s013
                                 + 0.959685 w_k1_FDX_s013
                                 + 1.52369 w_k1_IBM_s013
                                 + 0.812687 w_k1_INTC_s013
                                 + 1.06474 w_k1_KO_s013 + 1.72914 w_k1_MAT_s013
                                 + 1.00872 w_k1_MSFT_s013
                                 + 1.11929 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.771031 w_k1_ADM_s013
                                 + 0.842299 w_k1_AXP_s013
                                 + 0.801681 w_k1_DIS_s013
                                 + 0.784251 w_k1_FDX_s013
                                 + 1.00438 w_k1_IBM_s013
                                 + 0.847119 w_k1_INTC_s013
                                 + 0.838024 w_k1_KO_s013
                                 + 1.06413 w_k1_MAT_s013
                                 + 0.842172 w_k1_MSFT_s013
                                 + 0.940025 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.683191 w_k1_ADM_s013
                                 + 1.31983 w_k1_AXP_s013
                                 + 1.51957 w_k1_DIS_s013
                                 + 1.21389 w_k1_FDX_s013
                                 + 1.14566 w_k1_IBM_s013
                                 + 1.08882 w_k1_INTC_s013
                                 + 1.32972 w_k1_KO_s013 + 1.62842 w_k1_MAT_s013
                                 + 1.32803 w_k1_MSFT_s013
                                 + 1.20354 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.956556 w_k1_ADM_s013
                                 + 1.2182 w_k1_AXP_s013 + 1.06548 w_k1_DIS_s013
                                 + 0.992539 w_k1_FDX_s013
                                 + 1.01331 w_k1_IBM_s013
                                 + 0.8986 w_k1_INTC_s013 + 1.03431 w_k1_KO_s013
                                 + 1.25442 w_k1_MAT_s013
                                 + 0.965939 w_k1_MSFT_s013
                                 + 0.96344 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.846766 w_k1_ADM_s013
                                 + 0.922189 w_k1_AXP_s013
                                 + 0.80278 w_k1_DIS_s013
                                 + 0.868495 w_k1_FDX_s013
                                 + 0.695144 w_k1_IBM_s013
                                 + 0.862908 w_k1_INTC_s013
                                 + 0.844377 w_k1_KO_s013
                                 + 0.838537 w_k1_MAT_s013
                                 + 0.889651 w_k1_MSFT_s013
                                 + 0.940244 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.17125 w_k1_ADM_s013
                                 + 1.206 w_k1_AXP_s013 + 1.18179 w_k1_DIS_s013
                                 + 1.15373 w_k1_FDX_s013
                                 + 0.882112 w_k1_IBM_s013
                                 + 0.981684 w_k1_INTC_s013
                                 + 0.965311 w_k1_KO_s013
                                 + 0.968799 w_k1_MAT_s013
                                 + 1.01632 w_k1_MSFT_s013
                                 + 1.02741 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.926864 w_k1_ADM_s013
                                 + 0.969635 w_k1_AXP_s013
                                 + 1.10452 w_k1_DIS_s013
                                 + 0.984805 w_k1_FDX_s013
                                 + 1.16007 w_k1_IBM_s013
                                 + 0.932244 w_k1_INTC_s013
                                 + 1.08265 w_k1_KO_s013 + 1.14017 w_k1_MAT_s013
                                 + 1.04169 w_k1_MSFT_s013 + 1.1152 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.7267 w_k1_ADM_s014
                                 + 0.857999 w_k1_AXP_s014
                                 + 1.14893 w_k1_DIS_s014
                                 + 1.03109 w_k1_FDX_s014
                                 + 0.803454 w_k1_IBM_s014
                                 + 1.10259 w_k1_INTC_s014
                                 + 1.05644 w_k1_KO_s014
                                 + 0.760095 w_k1_MAT_s014
                                 + 1.10229 w_k1_MSFT_s014
                                 + 1.09225 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.771143 w_k1_ADM_s014
                                 + 0.678796 w_k1_AXP_s014
                                 + 0.555433 w_k1_DIS_s014
                                 + 0.642541 w_k1_FDX_s014
                                 + 0.804529 w_k1_IBM_s014
                                 + 0.685338 w_k1_INTC_s014
                                 + 0.74705 w_k1_KO_s014
                                 + 0.883298 w_k1_MAT_s014
                                 + 0.676001 w_k1_MSFT_s014
                                 + 0.801662 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.69297 w_k1_ADM_s014
                                 + 1.32549 w_k1_AXP_s014
                                 + 1.58757 w_k1_DIS_s014
                                 + 1.37421 w_k1_FDX_s014
                                 + 0.833818 w_k1_IBM_s014
                                 + 1.53036 w_k1_INTC_s014
                                 + 1.28067 w_k1_KO_s014 + 1.04459 w_k1_MAT_s014
                                 + 1.60474 w_k1_MSFT_s014
                                 + 1.35427 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.757936 w_k1_ADM_s014
                                 + 1.05951 w_k1_AXP_s014
                                 + 0.844025 w_k1_DIS_s014
                                 + 0.904402 w_k1_FDX_s014
                                 + 1.00599 w_k1_IBM_s014
                                 + 0.965158 w_k1_INTC_s014
                                 + 0.986707 w_k1_KO_s014
                                 + 1.27528 w_k1_MAT_s014
                                 + 0.89681 w_k1_MSFT_s014
                                 + 1.01163 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.658901 w_k1_ADM_s014
                                 + 0.587624 w_k1_AXP_s014
                                 + 0.610616 w_k1_DIS_s014
                                 + 0.572473 w_k1_FDX_s014
                                 + 1.22966 w_k1_IBM_s014
                                 + 0.9034 w_k1_INTC_s014
                                 + 0.968578 w_k1_KO_s014
                                 + 0.931307 w_k1_MAT_s014
                                 + 0.689633 w_k1_MSFT_s014
                                 + 0.789605 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.16897 w_k1_ADM_s014
                                 + 1.29467 w_k1_AXP_s014
                                 + 1.24594 w_k1_DIS_s014
                                 + 1.15697 w_k1_FDX_s014
                                 + 1.19496 w_k1_IBM_s014
                                 + 1.22017 w_k1_INTC_s014
                                 + 1.26081 w_k1_KO_s014 + 1.11854 w_k1_MAT_s014
                                 + 1.01123 w_k1_MSFT_s014
                                 + 1.02979 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.42286 w_k1_ADM_s014
                                 + 1.73686 w_k1_AXP_s014
                                 + 2.09971 w_k1_DIS_s014
                                 + 1.58493 w_k1_FDX_s014
                                 + 1.05593 w_k1_IBM_s014
                                 + 1.59504 w_k1_INTC_s014
                                 + 1.59789 w_k1_KO_s014 + 1.32077 w_k1_MAT_s014
                                 + 1.43851 w_k1_MSFT_s014
                                 + 1.44662 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.29361 w_k1_ADM_s014
                                 + 1.17692 w_k1_AXP_s014 + 1.3113 w_k1_DIS_s014
                                 + 1.08963 w_k1_FDX_s014
                                 + 1.77354 w_k1_IBM_s014
                                 + 0.942669 w_k1_INTC_s014
                                 + 1.30014 w_k1_KO_s014 + 1.74369 w_k1_MAT_s014
                                 + 0.944163 w_k1_MSFT_s014
                                 + 1.20401 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.998003 w_k1_ADM_s015
                                 + 0.941834 w_k1_AXP_s015
                                 + 1.10004 w_k1_DIS_s015
                                 + 0.986732 w_k1_FDX_s015
                                 + 1.29459 w_k1_IBM_s015
                                 + 1.07662 w_k1_INTC_s015
                                 + 1.16209 w_k1_KO_s015 + 1.33499 w_k1_MAT_s015
                                 + 1.10574 w_k1_MSFT_s015
                                 + 1.09801 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.34033 w_k1_ADM_s015
                                 + 1.36405 w_k1_AXP_s015 + 1.7642 w_k1_DIS_s015
                                 + 1.52164 w_k1_FDX_s015
                                 + 1.14024 w_k1_IBM_s015
                                 + 1.28718 w_k1_INTC_s015
                                 + 1.52251 w_k1_KO_s015 + 1.0157 w_k1_MAT_s015
                                 + 1.38498 w_k1_MSFT_s015
                                 + 1.39856 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.855652 w_k1_ADM_s015
                                 + 0.674783 w_k1_AXP_s015
                                 + 0.757268 w_k1_DIS_s015
                                 + 0.794954 w_k1_FDX_s015
                                 + 1.02012 w_k1_IBM_s015
                                 + 0.744786 w_k1_INTC_s015
                                 + 0.998777 w_k1_KO_s015
                                 + 0.672298 w_k1_MAT_s015
                                 + 0.841267 w_k1_MSFT_s015
                                 + 0.934321 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.93714 w_k1_ADM_s015
                                 + 0.877974 w_k1_AXP_s015
                                 + 0.964222 w_k1_DIS_s015
                                 + 0.933186 w_k1_FDX_s015
                                 + 0.846085 w_k1_IBM_s015
                                 + 0.891906 w_k1_INTC_s015
                                 + 1.00662 w_k1_KO_s015
                                 + 0.874167 w_k1_MAT_s015
                                 + 1.17595 w_k1_MSFT_s015
                                 + 1.00159 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.22975 w_k1_ADM_s015
                                 + 1.42536 w_k1_AXP_s015
                                 + 1.81534 w_k1_DIS_s015
                                 + 1.26024 w_k1_FDX_s015 + 1.7885 w_k1_IBM_s015
                                 + 1.31716 w_k1_INTC_s015
                                 + 1.83693 w_k1_KO_s015 + 1.50742 w_k1_MAT_s015
                                 + 1.41423 w_k1_MSFT_s015
                                 + 1.13132 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.928304 w_k1_ADM_s015
                                 + 0.822177 w_k1_AXP_s015
                                 + 0.763714 w_k1_DIS_s015
                                 + 1.0121 w_k1_FDX_s015
                                 + 0.649075 w_k1_IBM_s015
                                 + 1.01078 w_k1_INTC_s015
                                 + 0.893453 w_k1_KO_s015
                                 + 0.571506 w_k1_MAT_s015
                                 + 0.942616 w_k1_MSFT_s015
                                 + 0.942219 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.81599 w_k1_ADM_s015
                                 + 0.642521 w_k1_AXP_s015
                                 + 0.511312 w_k1_DIS_s015
                                 + 0.728027 w_k1_FDX_s015
                                 + 0.669917 w_k1_IBM_s015
                                 + 0.905651 w_k1_INTC_s015
                                 + 0.81787 w_k1_KO_s015
                                 + 0.653846 w_k1_MAT_s015
                                 + 0.783018 w_k1_MSFT_s015
                                 + 0.896654 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.937463 w_k1_ADM_s015
                                 + 0.944027 w_k1_AXP_s015
                                 + 1.38779 w_k1_DIS_s015 + 1.2393 w_k1_FDX_s015
                                 + 0.965275 w_k1_IBM_s015
                                 + 1.17133 w_k1_INTC_s015
                                 + 1.15844 w_k1_KO_s015
                                 + 0.679963 w_k1_MAT_s015
                                 + 1.33814 w_k1_MSFT_s015
                                 + 1.13813 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.682315 w_k1_ADM_s016
                                 + 0.827456 w_k1_AXP_s016
                                 + 0.918321 w_k1_DIS_s016
                                 + 0.884439 w_k1_FDX_s016
                                 + 1.24786 w_k1_IBM_s016
                                 + 1.14993 w_k1_INTC_s016
                                 + 1.00846 w_k1_KO_s016 + 1.0001 w_k1_MAT_s016
                                 + 0.890229 w_k1_MSFT_s016
                                 + 1.01749 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.825829 w_k1_ADM_s016
                                 + 0.61498 w_k1_AXP_s016
                                 + 0.617429 w_k1_DIS_s016
                                 + 0.71488 w_k1_FDX_s016
                                 + 0.748687 w_k1_IBM_s016
                                 + 0.875049 w_k1_INTC_s016
                                 + 0.724214 w_k1_KO_s016
                                 + 0.68032 w_k1_MAT_s016
                                 + 0.769855 w_k1_MSFT_s016
                                 + 0.835677 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.678319 w_k1_ADM_s016
                                 + 0.703842 w_k1_AXP_s016
                                 + 0.786261 w_k1_DIS_s016
                                 + 0.768681 w_k1_FDX_s016
                                 + 0.953567 w_k1_IBM_s016
                                 + 0.921175 w_k1_INTC_s016
                                 + 0.755572 w_k1_KO_s016
                                 + 0.652684 w_k1_MAT_s016
                                 + 0.884101 w_k1_MSFT_s016
                                 + 0.868902 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.13265 w_k1_ADM_s016
                                 + 1.08632 w_k1_AXP_s016
                                 + 1.26529 w_k1_DIS_s016 + 1.123 w_k1_FDX_s016
                                 + 1.04976 w_k1_IBM_s016
                                 + 1.17791 w_k1_INTC_s016
                                 + 1.00549 w_k1_KO_s016 + 1.00604 w_k1_MAT_s016
                                 + 1.18218 w_k1_MSFT_s016
                                 + 1.04054 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.823519 w_k1_ADM_s016
                                 + 0.900984 w_k1_AXP_s016
                                 + 0.893862 w_k1_DIS_s016
                                 + 0.930396 w_k1_FDX_s016
                                 + 0.836584 w_k1_IBM_s016
                                 + 1.07655 w_k1_INTC_s016
                                 + 0.816408 w_k1_KO_s016
                                 + 0.682343 w_k1_MAT_s016
                                 + 0.956676 w_k1_MSFT_s016
                                 + 0.905074 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.1696 w_k1_ADM_s016
                                 + 1.70064 w_k1_AXP_s016
                                 + 1.89784 w_k1_DIS_s016
                                 + 1.45506 w_k1_FDX_s016
                                 + 1.37672 w_k1_IBM_s016
                                 + 1.45976 w_k1_INTC_s016
                                 + 1.33845 w_k1_KO_s016 + 1.41218 w_k1_MAT_s016
                                 + 1.46246 w_k1_MSFT_s016
                                 + 1.41113 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.594648 w_k1_ADM_s016
                                 + 0.723639 w_k1_AXP_s016
                                 + 0.517531 w_k1_DIS_s016
                                 + 0.635219 w_k1_FDX_s016
                                 + 0.768321 w_k1_IBM_s016
                                 + 0.739635 w_k1_INTC_s016
                                 + 0.557481 w_k1_KO_s016
                                 + 0.569663 w_k1_MAT_s016
                                 + 0.639344 w_k1_MSFT_s016
                                 + 0.697595 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.720828 w_k1_ADM_s016
                                 + 0.830176 w_k1_AXP_s016
                                 + 0.807929 w_k1_DIS_s016
                                 + 0.943779 w_k1_FDX_s016
                                 + 0.648581 w_k1_IBM_s016
                                 + 1.05896 w_k1_INTC_s016
                                 + 0.732573 w_k1_KO_s016
                                 + 0.746571 w_k1_MAT_s016
                                 + 0.983155 w_k1_MSFT_s016
                                 + 0.881486 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.978364 w_k1_ADM_s017
                                 + 1.07427 w_k1_AXP_s017
                                 + 0.996973 w_k1_DIS_s017
                                 + 1.03029 w_k1_FDX_s017
                                 + 1.18302 w_k1_IBM_s017
                                 + 1.11965 w_k1_INTC_s017
                                 + 1.10347 w_k1_KO_s017
                                 + 0.998528 w_k1_MAT_s017
                                 + 1.04952 w_k1_MSFT_s017
                                 + 1.22297 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.03795 w_k1_ADM_s017
                                 + 1.28074 w_k1_AXP_s017
                                 + 0.822037 w_k1_DIS_s017
                                 + 0.855585 w_k1_FDX_s017
                                 + 1.00794 w_k1_IBM_s017
                                 + 0.974733 w_k1_INTC_s017
                                 + 0.917895 w_k1_KO_s017
                                 + 1.09993 w_k1_MAT_s017
                                 + 0.910127 w_k1_MSFT_s017
                                 + 1.0407 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.825794 w_k1_ADM_s017
                                 + 0.808455 w_k1_AXP_s017
                                 + 0.698425 w_k1_DIS_s017
                                 + 0.777495 w_k1_FDX_s017
                                 + 1.04105 w_k1_IBM_s017
                                 + 1.11565 w_k1_INTC_s017
                                 + 0.950954 w_k1_KO_s017
                                 + 0.797914 w_k1_MAT_s017
                                 + 0.872741 w_k1_MSFT_s017
                                 + 0.965728 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.33752 w_k1_ADM_s017
                                 + 1.18434 w_k1_AXP_s017
                                 + 1.10268 w_k1_DIS_s017
                                 + 1.19221 w_k1_FDX_s017
                                 + 0.923748 w_k1_IBM_s017
                                 + 1.13752 w_k1_INTC_s017
                                 + 1.03115 w_k1_KO_s017 + 1.01517 w_k1_MAT_s017
                                 + 1.13731 w_k1_MSFT_s017
                                 + 1.20528 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.874591 w_k1_ADM_s017
                                 + 0.735581 w_k1_AXP_s017
                                 + 0.705133 w_k1_DIS_s017
                                 + 0.829202 w_k1_FDX_s017
                                 + 0.706498 w_k1_IBM_s017
                                 + 1.1574 w_k1_INTC_s017
                                 + 0.798545 w_k1_KO_s017
                                 + 0.551771 w_k1_MAT_s017
                                 + 0.873392 w_k1_MSFT_s017
                                 + 0.906143 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.36307 w_k1_ADM_s017
                                 + 1.10425 w_k1_AXP_s017
                                 + 1.42829 w_k1_DIS_s017
                                 + 1.13099 w_k1_FDX_s017 + 1.3392 w_k1_IBM_s017
                                 + 1.21768 w_k1_INTC_s017
                                 + 1.36999 w_k1_KO_s017 + 1.36211 w_k1_MAT_s017
                                 + 1.1627 w_k1_MSFT_s017 + 1.10951 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.6312 w_k1_ADM_s017
                                 + 1.46238 w_k1_AXP_s017
                                 + 1.84449 w_k1_DIS_s017
                                 + 1.54225 w_k1_FDX_s017
                                 + 1.24875 w_k1_IBM_s017
                                 + 1.54529 w_k1_INTC_s017
                                 + 1.52912 w_k1_KO_s017 + 1.06674 w_k1_MAT_s017
                                 + 1.48397 w_k1_MSFT_s017 + 1.3758 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.14216 w_k1_ADM_s017
                                 + 0.971086 w_k1_AXP_s017
                                 + 0.978356 w_k1_DIS_s017
                                 + 0.947303 w_k1_FDX_s017
                                 + 1.48004 w_k1_IBM_s017
                                 + 0.968127 w_k1_INTC_s017
                                 + 1.2246 w_k1_KO_s017 + 1.44624 w_k1_MAT_s017
                                 + 0.954721 w_k1_MSFT_s017
                                 + 1.18926 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.651975 w_k1_ADM_s100
                                 + 0.657682 w_k1_AXP_s100
                                 + 0.562513 w_k1_DIS_s100
                                 + 0.732264 w_k1_FDX_s100
                                 + 0.668297 w_k1_IBM_s100
                                 + 0.804059 w_k1_INTC_s100
                                 + 0.683937 w_k1_KO_s100
                                 + 0.600427 w_k1_MAT_s100
                                 + 0.656637 w_k1_MSFT_s100
                                 + 0.7499 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.952404 w_k1_ADM_s100
                                 + 1.25968 w_k1_AXP_s100
                                 + 0.962904 w_k1_DIS_s100
                                 + 0.962411 w_k1_FDX_s100
                                 + 1.07654 w_k1_IBM_s100
                                 + 0.980058 w_k1_INTC_s100
                                 + 1.06639 w_k1_KO_s100 + 1.24925 w_k1_MAT_s100
                                 + 0.994994 w_k1_MSFT_s100
                                 + 1.03016 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.808131 w_k1_ADM_s100
                                 + 0.827311 w_k1_AXP_s100
                                 + 0.755828 w_k1_DIS_s100
                                 + 0.851963 w_k1_FDX_s100
                                 + 1.07323 w_k1_IBM_s100
                                 + 0.84404 w_k1_INTC_s100
                                 + 0.863953 w_k1_KO_s100 + 1.0225 w_k1_MAT_s100
                                 + 0.849163 w_k1_MSFT_s100
                                 + 0.872491 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.07772 w_k1_ADM_s100
                                 + 1.07358 w_k1_AXP_s100
                                 + 1.01629 w_k1_DIS_s100
                                 + 1.11387 w_k1_FDX_s100
                                 + 0.802171 w_k1_IBM_s100
                                 + 1.11475 w_k1_INTC_s100
                                 + 0.895859 w_k1_KO_s100
                                 + 0.617157 w_k1_MAT_s100
                                 + 1.04618 w_k1_MSFT_s100
                                 + 0.976902 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.27133 w_k1_ADM_s100
                                 + 1.4465 w_k1_AXP_s100 + 1.56552 w_k1_DIS_s100
                                 + 1.44638 w_k1_FDX_s100 + 1.0562 w_k1_IBM_s100
                                 + 1.48081 w_k1_INTC_s100
                                 + 1.14891 w_k1_KO_s100 + 1.04375 w_k1_MAT_s100
                                 + 1.33188 w_k1_MSFT_s100
                                 + 1.34091 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.03912 w_k1_ADM_s100
                                 + 1.29529 w_k1_AXP_s100
                                 + 1.23582 w_k1_DIS_s100
                                 + 1.03669 w_k1_FDX_s100
                                 + 1.59009 w_k1_IBM_s100
                                 + 1.13229 w_k1_INTC_s100
                                 + 1.42809 w_k1_KO_s100 + 1.73969 w_k1_MAT_s100
                                 + 1.15611 w_k1_MSFT_s100
                                 + 1.20139 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.891272 w_k1_ADM_s100
                                 + 0.814377 w_k1_AXP_s100
                                 + 0.728642 w_k1_DIS_s100
                                 + 0.925121 w_k1_FDX_s100
                                 + 0.657188 w_k1_IBM_s100
                                 + 0.905082 w_k1_INTC_s100
                                 + 0.89104 w_k1_KO_s100
                                 + 0.579852 w_k1_MAT_s100
                                 + 0.863573 w_k1_MSFT_s100
                                 + 0.906005 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.16869 w_k1_ADM_s100
                                 + 1.34049 w_k1_AXP_s100
                                 + 1.23635 w_k1_DIS_s100
                                 + 1.20163 w_k1_FDX_s100
                                 + 1.19455 w_k1_IBM_s100
                                 + 1.10522 w_k1_INTC_s100
                                 + 1.33367 w_k1_KO_s100 + 1.25683 w_k1_MAT_s100
                                 + 1.04463 w_k1_MSFT_s100
                                 + 1.13981 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.87537 w_k1_ADM_s101
                                 + 0.868031 w_k1_AXP_s101
                                 + 0.759717 w_k1_DIS_s101
                                 + 0.903084 w_k1_FDX_s101
                                 + 0.938361 w_k1_IBM_s101
                                 + 0.994823 w_k1_INTC_s101
                                 + 1.05208 w_k1_KO_s101
                                 + 0.872518 w_k1_MAT_s101
                                 + 0.854003 w_k1_MSFT_s101
                                 + 0.931536 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.12051 w_k1_ADM_s101
                                 + 1.14281 w_k1_AXP_s101
                                 + 1.28201 w_k1_DIS_s101
                                 + 1.19229 w_k1_FDX_s101
                                 + 0.785329 w_k1_IBM_s101
                                 + 1.33805 w_k1_INTC_s101
                                 + 1.03661 w_k1_KO_s101
                                 + 0.844264 w_k1_MAT_s101
                                 + 1.07548 w_k1_MSFT_s101
                                 + 1.02697 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.970782 w_k1_ADM_s101
                                 + 0.706944 w_k1_AXP_s101
                                 + 0.920354 w_k1_DIS_s101
                                 + 0.85922 w_k1_FDX_s101 + 1.0096 w_k1_IBM_s101
                                 + 0.826115 w_k1_INTC_s101
                                 + 0.996332 w_k1_KO_s101
                                 + 1.00303 w_k1_MAT_s101
                                 + 0.943241 w_k1_MSFT_s101
                                 + 0.929043 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.00506 w_k1_ADM_s101
                                 + 1.10081 w_k1_AXP_s101
                                 + 0.909453 w_k1_DIS_s101
                                 + 1.02334 w_k1_FDX_s101
                                 + 0.869418 w_k1_IBM_s101
                                 + 1.13971 w_k1_INTC_s101
                                 + 0.965373 w_k1_KO_s101
                                 + 0.703832 w_k1_MAT_s101
                                 + 0.945477 w_k1_MSFT_s101
                                 + 0.941349 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.3478 w_k1_ADM_s101
                                 + 1.46927 w_k1_AXP_s101
                                 + 1.60256 w_k1_DIS_s101
                                 + 1.25003 w_k1_FDX_s101
                                 + 1.26485 w_k1_IBM_s101
                                 + 1.29066 w_k1_INTC_s101
                                 + 1.34394 w_k1_KO_s101 + 1.38879 w_k1_MAT_s101
                                 + 1.21147 w_k1_MSFT_s101
                                 + 1.18785 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.72924 w_k1_ADM_s101
                                 + 0.658309 w_k1_AXP_s101
                                 + 0.682361 w_k1_DIS_s101
                                 + 0.765578 w_k1_FDX_s101
                                 + 0.911443 w_k1_IBM_s101
                                 + 0.847816 w_k1_INTC_s101
                                 + 0.913261 w_k1_KO_s101
                                 + 0.778372 w_k1_MAT_s101
                                 + 0.730053 w_k1_MSFT_s101
                                 + 0.862761 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.01323 w_k1_ADM_s101
                                 + 1.51114 w_k1_AXP_s101 + 0.841 w_k1_DIS_s101
                                 + 1.02398 w_k1_FDX_s101
                                 + 1.15325 w_k1_IBM_s101
                                 + 1.25455 w_k1_INTC_s101
                                 + 0.90869 w_k1_KO_s101 + 1.09762 w_k1_MAT_s101
                                 + 0.775042 w_k1_MSFT_s101
                                 + 0.930164 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.07751 w_k1_ADM_s101
                                 + 0.973611 w_k1_AXP_s101
                                 + 1.22653 w_k1_DIS_s101
                                 + 1.04373 w_k1_FDX_s101 + 1.4132 w_k1_IBM_s101
                                 + 0.934928 w_k1_INTC_s101
                                 + 1.20805 w_k1_KO_s101 + 1.1325 w_k1_MAT_s101
                                 + 1.01416 w_k1_MSFT_s101
                                 + 1.01925 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.916304 w_k1_ADM_s102
                                 + 0.812246 w_k1_AXP_s102
                                 + 0.879807 w_k1_DIS_s102
                                 + 0.983312 w_k1_FDX_s102
                                 + 0.656371 w_k1_IBM_s102
                                 + 0.936374 w_k1_INTC_s102
                                 + 0.887872 w_k1_KO_s102
                                 + 0.878889 w_k1_MAT_s102
                                 + 0.977708 w_k1_MSFT_s102
                                 + 0.934009 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.26016 w_k1_ADM_s102
                                 + 1.39251 w_k1_AXP_s102
                                 + 1.06844 w_k1_DIS_s102
                                 + 1.08272 w_k1_FDX_s102
                                 + 1.46962 w_k1_IBM_s102
                                 + 0.905063 w_k1_INTC_s102
                                 + 1.32091 w_k1_KO_s102 + 1.5799 w_k1_MAT_s102
                                 + 1.00413 w_k1_MSFT_s102 + 1.2006 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.15969 w_k1_ADM_s102
                                 + 1.12802 w_k1_AXP_s102
                                 + 0.964683 w_k1_DIS_s102 + 1.147 w_k1_FDX_s102
                                 + 0.911354 w_k1_IBM_s102
                                 + 1.18771 w_k1_INTC_s102
                                 + 0.996615 w_k1_KO_s102
                                 + 0.763135 w_k1_MAT_s102
                                 + 0.917602 w_k1_MSFT_s102
                                 + 1.02582 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.41184 w_k1_ADM_s102
                                 + 1.58894 w_k1_AXP_s102
                                 + 1.60951 w_k1_DIS_s102
                                 + 1.45326 w_k1_FDX_s102 + 1.4943 w_k1_IBM_s102
                                 + 1.25082 w_k1_INTC_s102 + 1.5362 w_k1_KO_s102
                                 + 1.69481 w_k1_MAT_s102
                                 + 1.26239 w_k1_MSFT_s102
                                 + 1.33095 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.949592 w_k1_ADM_s102
                                 + 0.926823 w_k1_AXP_s102
                                 + 0.897035 w_k1_DIS_s102
                                 + 0.944452 w_k1_FDX_s102
                                 + 1.13809 w_k1_IBM_s102
                                 + 1.08178 w_k1_INTC_s102
                                 + 1.08218 w_k1_KO_s102 + 1.09283 w_k1_MAT_s102
                                 + 0.952488 w_k1_MSFT_s102
                                 + 0.999976 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.2517 w_k1_ADM_s102
                                 + 1.09377 w_k1_AXP_s102 + 1.3591 w_k1_DIS_s102
                                 + 1.40498 w_k1_FDX_s102
                                 + 0.878358 w_k1_IBM_s102
                                 + 1.23485 w_k1_INTC_s102
                                 + 1.15487 w_k1_KO_s102 + 1.01762 w_k1_MAT_s102
                                 + 1.24944 w_k1_MSFT_s102
                                 + 1.15046 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.14855 w_k1_ADM_s102
                                 + 1.15391 w_k1_AXP_s102
                                 + 1.52044 w_k1_DIS_s102
                                 + 1.17682 w_k1_FDX_s102
                                 + 1.33319 w_k1_IBM_s102
                                 + 1.08909 w_k1_INTC_s102 + 1.4989 w_k1_KO_s102
                                 + 1.02411 w_k1_MAT_s102
                                 + 1.24303 w_k1_MSFT_s102
                                 + 1.12328 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.851254 w_k1_ADM_s102
                                 + 0.718917 w_k1_AXP_s102
                                 + 0.573603 w_k1_DIS_s102
                                 + 0.707164 w_k1_FDX_s102
                                 + 0.984199 w_k1_IBM_s102
                                 + 0.74365 w_k1_INTC_s102
                                 + 0.911157 w_k1_KO_s102 + 1.1387 w_k1_MAT_s102
                                 + 0.735279 w_k1_MSFT_s102
                                 + 0.830689 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.09671 w_k1_ADM_s103
                                 + 1.28419 w_k1_AXP_s103
                                 + 1.32402 w_k1_DIS_s103
                                 + 1.19888 w_k1_FDX_s103
                                 + 0.930843 w_k1_IBM_s103
                                 + 1.18548 w_k1_INTC_s103
                                 + 0.884094 w_k1_KO_s103
                                 + 0.949244 w_k1_MAT_s103
                                 + 1.06368 w_k1_MSFT_s103
                                 + 0.973523 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.714492 w_k1_ADM_s103
                                 + 0.836762 w_k1_AXP_s103
                                 + 0.980639 w_k1_DIS_s103
                                 + 0.988205 w_k1_FDX_s103
                                 + 0.700614 w_k1_IBM_s103
                                 + 1.06346 w_k1_INTC_s103
                                 + 0.761016 w_k1_KO_s103
                                 + 0.753655 w_k1_MAT_s103
                                 + 0.935712 w_k1_MSFT_s103
                                 + 1.01345 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05964 w_k1_ADM_s103
                                 + 1.66641 w_k1_AXP_s103
                                 + 1.63368 w_k1_DIS_s103
                                 + 1.40376 w_k1_FDX_s103
                                 + 1.40172 w_k1_IBM_s103
                                 + 1.39646 w_k1_INTC_s103
                                 + 1.34761 w_k1_KO_s103 + 1.5446 w_k1_MAT_s103
                                 + 1.30787 w_k1_MSFT_s103
                                 + 1.27475 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.597898 w_k1_ADM_s103
                                 + 0.736029 w_k1_AXP_s103
                                 + 0.807269 w_k1_DIS_s103
                                 + 0.798981 w_k1_FDX_s103
                                 + 0.780972 w_k1_IBM_s103
                                 + 0.900188 w_k1_INTC_s103
                                 + 0.718491 w_k1_KO_s103
                                 + 0.827397 w_k1_MAT_s103
                                 + 0.802777 w_k1_MSFT_s103
                                 + 0.840911 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.05139 w_k1_ADM_s103
                                 + 1.27407 w_k1_AXP_s103
                                 + 0.895043 w_k1_DIS_s103
                                 + 0.943603 w_k1_FDX_s103
                                 + 0.854809 w_k1_IBM_s103
                                 + 0.853244 w_k1_INTC_s103
                                 + 0.824501 w_k1_KO_s103
                                 + 0.991084 w_k1_MAT_s103
                                 + 0.857301 w_k1_MSFT_s103
                                 + 0.961967 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.669558 w_k1_ADM_s103
                                 + 0.658111 w_k1_AXP_s103
                                 + 0.676065 w_k1_DIS_s103
                                 + 0.635799 w_k1_FDX_s103
                                 + 1.07724 w_k1_IBM_s103
                                 + 0.828232 w_k1_INTC_s103
                                 + 0.907535 w_k1_KO_s103
                                 + 0.915361 w_k1_MAT_s103
                                 + 0.755968 w_k1_MSFT_s103
                                 + 0.877247 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.914925 w_k1_ADM_s103
                                 + 1.34654 w_k1_AXP_s103
                                 + 1.11673 w_k1_DIS_s103
                                 + 1.04671 w_k1_FDX_s103
                                 + 1.21839 w_k1_IBM_s103
                                 + 1.12175 w_k1_INTC_s103
                                 + 1.04403 w_k1_KO_s103 + 1.51659 w_k1_MAT_s103
                                 + 0.993023 w_k1_MSFT_s103
                                 + 1.10692 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.796301 w_k1_ADM_s103
                                 + 1.04356 w_k1_AXP_s103
                                 + 0.942016 w_k1_DIS_s103
                                 + 0.869384 w_k1_FDX_s103
                                 + 1.39057 w_k1_IBM_s103
                                 + 0.884318 w_k1_INTC_s103
                                 + 1.00447 w_k1_KO_s103 + 1.54598 w_k1_MAT_s103
                                 + 0.843559 w_k1_MSFT_s103
                                 + 0.946827 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.625933 w_k1_ADM_s104
                                 + 0.577866 w_k1_AXP_s104
                                 + 0.583721 w_k1_DIS_s104
                                 + 0.82415 w_k1_FDX_s104
                                 + 0.583041 w_k1_IBM_s104
                                 + 0.769091 w_k1_INTC_s104
                                 + 0.637434 w_k1_KO_s104
                                 + 0.69055 w_k1_MAT_s104
                                 + 0.731999 w_k1_MSFT_s104
                                 + 0.811997 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.1788 w_k1_ADM_s104
                                 + 1.39178 w_k1_AXP_s104
                                 + 1.70785 w_k1_DIS_s104
                                 + 1.40727 w_k1_FDX_s104
                                 + 1.19377 w_k1_IBM_s104
                                 + 1.23196 w_k1_INTC_s104
                                 + 1.26003 w_k1_KO_s104 + 1.74811 w_k1_MAT_s104
                                 + 1.32414 w_k1_MSFT_s104
                                 + 1.00974 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.97213 w_k1_ADM_s104
                                 + 1.15566 w_k1_AXP_s104
                                 + 1.01321 w_k1_DIS_s104 + 1.0472 w_k1_FDX_s104
                                 + 1.2517 w_k1_IBM_s104
                                 + 1.02491 w_k1_INTC_s104
                                 + 1.20125 w_k1_KO_s104 + 1.40393 w_k1_MAT_s104
                                 + 1.05246 w_k1_MSFT_s104
                                 + 1.01486 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.974988 w_k1_ADM_s104
                                 + 0.966819 w_k1_AXP_s104
                                 + 0.849673 w_k1_DIS_s104
                                 + 1.06855 w_k1_FDX_s104
                                 + 0.721144 w_k1_IBM_s104
                                 + 1.05917 w_k1_INTC_s104
                                 + 0.811338 w_k1_KO_s104
                                 + 0.750013 w_k1_MAT_s104
                                 + 0.988834 w_k1_MSFT_s104
                                 + 0.926502 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.749013 w_k1_ADM_s104
                                 + 0.945636 w_k1_AXP_s104
                                 + 0.964103 w_k1_DIS_s104
                                 + 0.932613 w_k1_FDX_s104
                                 + 0.905648 w_k1_IBM_s104
                                 + 0.858422 w_k1_INTC_s104
                                 + 0.923621 w_k1_KO_s104
                                 + 1.05395 w_k1_MAT_s104
                                 + 0.874624 w_k1_MSFT_s104
                                 + 0.999011 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.694806 w_k1_ADM_s104
                                 + 0.760726 w_k1_AXP_s104
                                 + 0.598302 w_k1_DIS_s104
                                 + 0.775682 w_k1_FDX_s104
                                 + 0.871909 w_k1_IBM_s104
                                 + 0.748968 w_k1_INTC_s104
                                 + 0.721444 w_k1_KO_s104 + 1.1341 w_k1_MAT_s104
                                 + 0.771471 w_k1_MSFT_s104
                                 + 0.864312 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.0323 w_k1_ADM_s104
                                 + 1.37083 w_k1_AXP_s104
                                 + 1.19173 w_k1_DIS_s104
                                 + 1.10002 w_k1_FDX_s104
                                 + 1.64143 w_k1_IBM_s104
                                 + 1.08222 w_k1_INTC_s104
                                 + 1.31913 w_k1_KO_s104 + 2.29795 w_k1_MAT_s104
                                 + 1.02652 w_k1_MSFT_s104
                                 + 1.15542 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.15172 w_k1_ADM_s104
                                 + 1.03846 w_k1_AXP_s104
                                 + 1.11277 w_k1_DIS_s104
                                 + 1.22687 w_k1_FDX_s104 + 1.0016 w_k1_IBM_s104
                                 + 1.12105 w_k1_INTC_s104
                                 + 1.09888 w_k1_KO_s104 + 1.04335 w_k1_MAT_s104
                                 + 1.08202 w_k1_MSFT_s104
                                 + 0.972115 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.793456 w_k1_ADM_s105
                                 + 0.937644 w_k1_AXP_s105
                                 + 0.951139 w_k1_DIS_s105
                                 + 0.827367 w_k1_FDX_s105
                                 + 1.57191 w_k1_IBM_s105
                                 + 0.953483 w_k1_INTC_s105
                                 + 1.19925 w_k1_KO_s105 + 1.67371 w_k1_MAT_s105
                                 + 1.01434 w_k1_MSFT_s105
                                 + 0.922999 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.718463 w_k1_ADM_s105
                                 + 0.687869 w_k1_AXP_s105
                                 + 0.714609 w_k1_DIS_s105
                                 + 0.758408 w_k1_FDX_s105
                                 + 0.78422 w_k1_IBM_s105
                                 + 0.781342 w_k1_INTC_s105
                                 + 0.79649 w_k1_KO_s105
                                 + 0.544806 w_k1_MAT_s105
                                 + 0.866897 w_k1_MSFT_s105
                                 + 0.822242 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.19695 w_k1_ADM_s105
                                 + 1.35668 w_k1_AXP_s105 + 1.6204 w_k1_DIS_s105
                                 + 1.26446 w_k1_FDX_s105
                                 + 1.09679 w_k1_IBM_s105
                                 + 1.50824 w_k1_INTC_s105
                                 + 1.19485 w_k1_KO_s105 + 1.04453 w_k1_MAT_s105
                                 + 1.40008 w_k1_MSFT_s105
                                 + 1.20954 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.982939 w_k1_ADM_s105
                                 + 0.801079 w_k1_AXP_s105
                                 + 0.853641 w_k1_DIS_s105
                                 + 1.00745 w_k1_FDX_s105
                                 + 0.647703 w_k1_IBM_s105
                                 + 0.837063 w_k1_INTC_s105
                                 + 0.77787 w_k1_KO_s105
                                 + 0.579335 w_k1_MAT_s105
                                 + 0.911957 w_k1_MSFT_s105
                                 + 0.893208 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.30399 w_k1_ADM_s105
                                 + 1.36928 w_k1_AXP_s105
                                 + 1.66664 w_k1_DIS_s105
                                 + 1.26834 w_k1_FDX_s105
                                 + 1.66909 w_k1_IBM_s105
                                 + 1.47031 w_k1_INTC_s105
                                 + 1.64573 w_k1_KO_s105 + 1.93744 w_k1_MAT_s105
                                 + 1.54636 w_k1_MSFT_s105
                                 + 1.21664 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.23101 w_k1_ADM_s105
                                 + 1.00989 w_k1_AXP_s105
                                 + 1.05308 w_k1_DIS_s105
                                 + 1.03061 w_k1_FDX_s105
                                 + 1.17085 w_k1_IBM_s105
                                 + 1.03496 w_k1_INTC_s105 + 1.053 w_k1_KO_s105
                                 + 1.08194 w_k1_MAT_s105
                                 + 1.11328 w_k1_MSFT_s105
                                 + 0.954032 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.840616 w_k1_ADM_s105
                                 + 0.868228 w_k1_AXP_s105
                                 + 0.717439 w_k1_DIS_s105
                                 + 0.745874 w_k1_FDX_s105
                                 + 0.962703 w_k1_IBM_s105
                                 + 0.834008 w_k1_INTC_s105
                                 + 0.842875 w_k1_KO_s105 + 1.0312 w_k1_MAT_s105
                                 + 0.8873 w_k1_MSFT_s105
                                 + 0.860948 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.02433 w_k1_ADM_s105
                                 + 0.971278 w_k1_AXP_s105
                                 + 1.10326 w_k1_DIS_s105
                                 + 1.04688 w_k1_FDX_s105
                                 + 0.780534 w_k1_IBM_s105
                                 + 1.03739 w_k1_INTC_s105
                                 + 0.976746 w_k1_KO_s105
                                 + 0.832229 w_k1_MAT_s105
                                 + 1.04064 w_k1_MSFT_s105
                                 + 0.868473 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.954117 w_k1_ADM_s106
                                 + 0.709302 w_k1_AXP_s106
                                 + 0.67345 w_k1_DIS_s106
                                 + 0.796863 w_k1_FDX_s106
                                 + 0.688176 w_k1_IBM_s106
                                 + 0.816779 w_k1_INTC_s106
                                 + 0.83201 w_k1_KO_s106 + 0.67523 w_k1_MAT_s106
                                 + 0.744797 w_k1_MSFT_s106
                                 + 0.817984 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.39708 w_k1_ADM_s106
                                 + 1.30121 w_k1_AXP_s106 + 1.0281 w_k1_DIS_s106
                                 + 1.16435 w_k1_FDX_s106
                                 + 0.724249 w_k1_IBM_s106
                                 + 1.23578 w_k1_INTC_s106
                                 + 0.957517 w_k1_KO_s106
                                 + 0.933768 w_k1_MAT_s106
                                 + 1.01211 w_k1_MSFT_s106
                                 + 0.992357 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.00097 w_k1_ADM_s106
                                 + 0.983041 w_k1_AXP_s106
                                 + 0.934188 w_k1_DIS_s106
                                 + 0.913174 w_k1_FDX_s106
                                 + 1.55669 w_k1_IBM_s106
                                 + 1.02462 w_k1_INTC_s106 + 1.2349 w_k1_KO_s106
                                 + 1.62299 w_k1_MAT_s106
                                 + 0.915218 w_k1_MSFT_s106
                                 + 1.00507 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.18888 w_k1_ADM_s106
                                 + 1.15731 w_k1_AXP_s106
                                 + 0.943548 w_k1_DIS_s106
                                 + 1.02295 w_k1_FDX_s106
                                 + 1.12591 w_k1_IBM_s106
                                 + 1.08278 w_k1_INTC_s106
                                 + 1.06593 w_k1_KO_s106 + 1.39823 w_k1_MAT_s106
                                 + 0.895057 w_k1_MSFT_s106
                                 + 1.00487 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.992981 w_k1_ADM_s106
                                 + 0.932865 w_k1_AXP_s106
                                 + 0.898668 w_k1_DIS_s106
                                 + 1.00007 w_k1_FDX_s106
                                 + 0.590971 w_k1_IBM_s106
                                 + 1.25206 w_k1_INTC_s106
                                 + 0.785122 w_k1_KO_s106
                                 + 0.686264 w_k1_MAT_s106
                                 + 0.906482 w_k1_MSFT_s106
                                 + 0.861601 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.802714 w_k1_ADM_s106
                                 + 0.71526 w_k1_AXP_s106
                                 + 0.698237 w_k1_DIS_s106
                                 + 0.774778 w_k1_FDX_s106
                                 + 0.954827 w_k1_IBM_s106
                                 + 0.782863 w_k1_INTC_s106
                                 + 1.01053 w_k1_KO_s106
                                 + 0.840639 w_k1_MAT_s106
                                 + 0.706836 w_k1_MSFT_s106
                                 + 0.891623 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.995214 w_k1_ADM_s106
                                 + 0.930972 w_k1_AXP_s106
                                 + 1.07462 w_k1_DIS_s106
                                 + 0.980939 w_k1_FDX_s106
                                 + 1.02807 w_k1_IBM_s106
                                 + 1.13555 w_k1_INTC_s106
                                 + 1.03652 w_k1_KO_s106
                                 + 0.896541 w_k1_MAT_s106
                                 + 0.979295 w_k1_MSFT_s106
                                 + 1.17631 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.09519 w_k1_ADM_s106
                                 + 1.47265 w_k1_AXP_s106
                                 + 1.76995 w_k1_DIS_s106
                                 + 1.33938 w_k1_FDX_s106
                                 + 1.32681 w_k1_IBM_s106
                                 + 1.26352 w_k1_INTC_s106
                                 + 1.64311 w_k1_KO_s106 + 1.56041 w_k1_MAT_s106
                                 + 1.09695 w_k1_MSFT_s106
                                 + 1.34018 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.897802 w_k1_ADM_s107
                                 + 0.961238 w_k1_AXP_s107
                                 + 0.850116 w_k1_DIS_s107
                                 + 0.971885 w_k1_FDX_s107
                                 + 0.817322 w_k1_IBM_s107
                                 + 0.989594 w_k1_INTC_s107
                                 + 0.881723 w_k1_KO_s107
                                 + 0.798836 w_k1_MAT_s107
                                 + 0.881553 w_k1_MSFT_s107
                                 + 0.944381 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.8122 w_k1_ADM_s107
                                 + 0.75483 w_k1_AXP_s107
                                 + 0.895105 w_k1_DIS_s107
                                 + 0.893815 w_k1_FDX_s107
                                 + 0.83818 w_k1_IBM_s107
                                 + 1.11355 w_k1_INTC_s107
                                 + 0.92569 w_k1_KO_s107
                                 + 0.843362 w_k1_MAT_s107
                                 + 0.961441 w_k1_MSFT_s107
                                 + 0.874524 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.816698 w_k1_ADM_s107
                                 + 0.822887 w_k1_AXP_s107
                                 + 0.545259 w_k1_DIS_s107
                                 + 0.75486 w_k1_FDX_s107
                                 + 0.734738 w_k1_IBM_s107
                                 + 0.907777 w_k1_INTC_s107
                                 + 0.847958 w_k1_KO_s107
                                 + 0.730961 w_k1_MAT_s107
                                 + 0.694564 w_k1_MSFT_s107
                                 + 0.829415 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.10756 w_k1_ADM_s107
                                 + 1.20782 w_k1_AXP_s107
                                 + 1.26979 w_k1_DIS_s107 + 1.0527 w_k1_FDX_s107
                                 + 1.2023 w_k1_IBM_s107
                                 + 1.27863 w_k1_INTC_s107
                                 + 1.15678 w_k1_KO_s107 + 1.11066 w_k1_MAT_s107
                                 + 1.10374 w_k1_MSFT_s107
                                 + 1.16937 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.37088 w_k1_ADM_s107
                                 + 1.73397 w_k1_AXP_s107
                                 + 1.49493 w_k1_DIS_s107
                                 + 1.37681 w_k1_FDX_s107
                                 + 1.01925 w_k1_IBM_s107
                                 + 1.44032 w_k1_INTC_s107
                                 + 1.47693 w_k1_KO_s107 + 1.59312 w_k1_MAT_s107
                                 + 1.31777 w_k1_MSFT_s107
                                 + 1.21179 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.96799 w_k1_ADM_s107
                                 + 1.01187 w_k1_AXP_s107
                                 + 0.867259 w_k1_DIS_s107
                                 + 0.864385 w_k1_FDX_s107
                                 + 1.30006 w_k1_IBM_s107
                                 + 0.896823 w_k1_INTC_s107
                                 + 1.28334 w_k1_KO_s107 + 1.31749 w_k1_MAT_s107
                                 + 0.88927 w_k1_MSFT_s107
                                 + 1.02425 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.29087 w_k1_ADM_s107
                                 + 1.58035 w_k1_AXP_s107
                                 + 1.50768 w_k1_DIS_s107 + 1.2168 w_k1_FDX_s107
                                 + 1.6931 w_k1_IBM_s107
                                 + 1.46475 w_k1_INTC_s107
                                 + 1.61553 w_k1_KO_s107 + 1.21661 w_k1_MAT_s107
                                 + 1.2111 w_k1_MSFT_s107 + 1.33114 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.98795 w_k1_ADM_s107
                                 + 1.01069 w_k1_AXP_s107
                                 + 1.28538 w_k1_DIS_s107
                                 + 1.17948 w_k1_FDX_s107
                                 + 0.949039 w_k1_IBM_s107
                                 + 1.28642 w_k1_INTC_s107
                                 + 1.39741 w_k1_KO_s107 + 1.0002 w_k1_MAT_s107
                                 + 1.32875 w_k1_MSFT_s107
                                 + 1.15173 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.965603 w_k1_ADM_s110
                                 + 0.974751 w_k1_AXP_s110
                                 + 1.17031 w_k1_DIS_s110
                                 + 1.12404 w_k1_FDX_s110
                                 + 0.951102 w_k1_IBM_s110
                                 + 1.15454 w_k1_INTC_s110
                                 + 0.96117 w_k1_KO_s110
                                 + 0.794238 w_k1_MAT_s110
                                 + 1.09811 w_k1_MSFT_s110
                                 + 0.972537 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.14937 w_k1_ADM_s110
                                 + 1.39749 w_k1_AXP_s110
                                 + 1.60341 w_k1_DIS_s110
                                 + 1.47653 w_k1_FDX_s110
                                 + 1.43951 w_k1_IBM_s110
                                 + 1.09655 w_k1_INTC_s110
                                 + 1.36613 w_k1_KO_s110 + 2.18669 w_k1_MAT_s110
                                 + 1.32739 w_k1_MSFT_s110
                                 + 1.22117 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.738266 w_k1_ADM_s110
                                 + 0.665924 w_k1_AXP_s110
                                 + 0.533572 w_k1_DIS_s110
                                 + 0.677824 w_k1_FDX_s110
                                 + 0.882001 w_k1_IBM_s110
                                 + 0.638331 w_k1_INTC_s110
                                 + 0.67024 w_k1_KO_s110
                                 + 0.957326 w_k1_MAT_s110
                                 + 0.675595 w_k1_MSFT_s110
                                 + 0.797221 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.874897 w_k1_ADM_s110
                                 + 0.772613 w_k1_AXP_s110
                                 + 1.0293 w_k1_DIS_s110
                                 + 0.885978 w_k1_FDX_s110
                                 + 1.32416 w_k1_IBM_s110
                                 + 1.01091 w_k1_INTC_s110
                                 + 1.15509 w_k1_KO_s110
                                 + 0.881227 w_k1_MAT_s110
                                 + 0.960597 w_k1_MSFT_s110
                                 + 0.939139 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.854871 w_k1_ADM_s110
                                 + 1.26929 w_k1_AXP_s110
                                 + 0.895782 w_k1_DIS_s110
                                 + 0.96822 w_k1_FDX_s110
                                 + 1.65588 w_k1_IBM_s110
                                 + 0.825215 w_k1_INTC_s110
                                 + 1.19795 w_k1_KO_s110 + 1.99706 w_k1_MAT_s110
                                 + 0.84136 w_k1_MSFT_s110
                                 + 1.00823 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.10164 w_k1_ADM_s110
                                 + 1.39529 w_k1_AXP_s110
                                 + 1.26481 w_k1_DIS_s110
                                 + 1.25775 w_k1_FDX_s110
                                 + 1.14553 w_k1_IBM_s110
                                 + 1.20442 w_k1_INTC_s110
                                 + 1.39522 w_k1_KO_s110 + 1.27956 w_k1_MAT_s110
                                 + 1.29835 w_k1_MSFT_s110
                                 + 1.13482 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.839059 w_k1_ADM_s110
                                 + 0.841173 w_k1_AXP_s110
                                 + 0.852542 w_k1_DIS_s110
                                 + 0.945581 w_k1_FDX_s110
                                 + 0.968767 w_k1_IBM_s110
                                 + 1.00937 w_k1_INTC_s110
                                 + 0.929252 w_k1_KO_s110
                                 + 0.964993 w_k1_MAT_s110
                                 + 0.907254 w_k1_MSFT_s110
                                 + 0.971607 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.831497 w_k1_ADM_s110
                                 + 0.710232 w_k1_AXP_s110
                                 + 0.699947 w_k1_DIS_s110
                                 + 0.893217 w_k1_FDX_s110
                                 + 0.743863 w_k1_IBM_s110
                                 + 0.771949 w_k1_INTC_s110
                                 + 0.76139 w_k1_KO_s110
                                 + 0.752633 w_k1_MAT_s110
                                 + 0.82743 w_k1_MSFT_s110
                                 + 0.895254 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.876209 w_k1_ADM_s111
                                 + 0.943195 w_k1_AXP_s111
                                 + 0.920335 w_k1_DIS_s111
                                 + 1.00603 w_k1_FDX_s111
                                 + 0.935288 w_k1_IBM_s111
                                 + 1.12609 w_k1_INTC_s111
                                 + 0.952589 w_k1_KO_s111
                                 + 0.716941 w_k1_MAT_s111
                                 + 1.02552 w_k1_MSFT_s111 + 0.9383 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.785874 w_k1_ADM_s111
                                 + 0.745958 w_k1_AXP_s111
                                 + 0.688091 w_k1_DIS_s111
                                 + 0.786512 w_k1_FDX_s111
                                 + 1.02827 w_k1_IBM_s111
                                 + 0.869132 w_k1_INTC_s111
                                 + 0.883896 w_k1_KO_s111
                                 + 0.774988 w_k1_MAT_s111
                                 + 0.801196 w_k1_MSFT_s111
                                 + 0.831422 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.06069 w_k1_ADM_s111
                                 + 0.934647 w_k1_AXP_s111
                                 + 1.23806 w_k1_DIS_s111 + 1.0617 w_k1_FDX_s111
                                 + 0.871548 w_k1_IBM_s111
                                 + 1.13353 w_k1_INTC_s111 + 1.0811 w_k1_KO_s111
                                 + 0.666144 w_k1_MAT_s111
                                 + 0.955081 w_k1_MSFT_s111
                                 + 0.966301 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.863835 w_k1_ADM_s111
                                 + 1.00303 w_k1_AXP_s111
                                 + 0.952932 w_k1_DIS_s111
                                 + 0.849134 w_k1_FDX_s111
                                 + 1.56363 w_k1_IBM_s111
                                 + 0.987989 w_k1_INTC_s111
                                 + 0.946313 w_k1_KO_s111
                                 + 1.32385 w_k1_MAT_s111
                                 + 0.772746 w_k1_MSFT_s111
                                 + 0.806576 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.30296 w_k1_ADM_s111
                                 + 1.28895 w_k1_AXP_s111
                                 + 1.26758 w_k1_DIS_s111
                                 + 1.27061 w_k1_FDX_s111
                                 + 0.901816 w_k1_IBM_s111
                                 + 1.11098 w_k1_INTC_s111
                                 + 1.05696 w_k1_KO_s111
                                 + 0.959197 w_k1_MAT_s111
                                 + 1.11631 w_k1_MSFT_s111
                                 + 1.15301 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.19142 w_k1_ADM_s111
                                 + 1.13606 w_k1_AXP_s111
                                 + 1.14307 w_k1_DIS_s111
                                 + 1.05916 w_k1_FDX_s111
                                 + 1.24236 w_k1_IBM_s111
                                 + 1.2073 w_k1_INTC_s111 + 1.1453 w_k1_KO_s111
                                 + 0.792489 w_k1_MAT_s111
                                 + 1.02261 w_k1_MSFT_s111
                                 + 0.961824 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.47338 w_k1_ADM_s111
                                 + 1.27592 w_k1_AXP_s111
                                 + 1.50308 w_k1_DIS_s111
                                 + 1.19859 w_k1_FDX_s111
                                 + 1.54337 w_k1_IBM_s111
                                 + 1.1404 w_k1_INTC_s111 + 1.32258 w_k1_KO_s111
                                 + 1.16148 w_k1_MAT_s111
                                 + 1.22566 w_k1_MSFT_s111
                                 + 1.13062 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.21642 w_k1_ADM_s111
                                 + 1.17602 w_k1_AXP_s111
                                 + 0.919474 w_k1_DIS_s111
                                 + 1.19598 w_k1_FDX_s111
                                 + 0.780758 w_k1_IBM_s111
                                 + 1.07566 w_k1_INTC_s111
                                 + 0.687046 w_k1_KO_s111
                                 + 0.437283 w_k1_MAT_s111
                                 + 0.885875 w_k1_MSFT_s111
                                 + 0.804108 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.23537 w_k1_ADM_s112
                                 + 1.04096 w_k1_AXP_s112
                                 + 1.23165 w_k1_DIS_s112
                                 + 0.959019 w_k1_FDX_s112
                                 + 1.42917 w_k1_IBM_s112
                                 + 0.978315 w_k1_INTC_s112
                                 + 1.1971 w_k1_KO_s112 + 1.28995 w_k1_MAT_s112
                                 + 1.0407 w_k1_MSFT_s112 + 1.03693 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.924563 w_k1_ADM_s112
                                 + 0.734404 w_k1_AXP_s112
                                 + 0.663992 w_k1_DIS_s112
                                 + 0.859963 w_k1_FDX_s112
                                 + 0.59716 w_k1_IBM_s112
                                 + 1.05766 w_k1_INTC_s112
                                 + 0.664873 w_k1_KO_s112
                                 + 0.526566 w_k1_MAT_s112
                                 + 0.867211 w_k1_MSFT_s112
                                 + 0.793226 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.723564 w_k1_ADM_s112
                                 + 0.698249 w_k1_AXP_s112
                                 + 0.641834 w_k1_DIS_s112
                                 + 0.741383 w_k1_FDX_s112
                                 + 0.90146 w_k1_IBM_s112
                                 + 0.927086 w_k1_INTC_s112
                                 + 0.812775 w_k1_KO_s112
                                 + 0.94926 w_k1_MAT_s112
                                 + 0.858907 w_k1_MSFT_s112
                                 + 0.717235 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.11797 w_k1_ADM_s112
                                 + 0.898363 w_k1_AXP_s112
                                 + 0.922089 w_k1_DIS_s112
                                 + 0.871189 w_k1_FDX_s112
                                 + 1.13548 w_k1_IBM_s112
                                 + 0.786475 w_k1_INTC_s112
                                 + 1.05064 w_k1_KO_s112 + 1.32778 w_k1_MAT_s112
                                 + 1.07284 w_k1_MSFT_s112
                                 + 0.976075 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.61346 w_k1_ADM_s112
                                 + 1.46202 w_k1_AXP_s112
                                 + 1.53203 w_k1_DIS_s112
                                 + 1.32635 w_k1_FDX_s112
                                 + 1.30947 w_k1_IBM_s112
                                 + 1.24397 w_k1_INTC_s112
                                 + 1.32679 w_k1_KO_s112 + 1.60835 w_k1_MAT_s112
                                 + 1.25419 w_k1_MSFT_s112
                                 + 1.18956 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.17935 w_k1_ADM_s112
                                 + 1.07665 w_k1_AXP_s112
                                 + 1.11823 w_k1_DIS_s112
                                 + 1.09432 w_k1_FDX_s112
                                 + 1.13501 w_k1_IBM_s112
                                 + 1.24933 w_k1_INTC_s112
                                 + 1.02035 w_k1_KO_s112 + 1.17051 w_k1_MAT_s112
                                 + 1.18444 w_k1_MSFT_s112
                                 + 0.942365 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.35173 w_k1_ADM_s112
                                 + 0.84058 w_k1_AXP_s112
                                 + 1.08486 w_k1_DIS_s112
                                 + 1.10985 w_k1_FDX_s112
                                 + 0.83925 w_k1_IBM_s112
                                 + 1.1999 w_k1_INTC_s112
                                 + 0.823078 w_k1_KO_s112
                                 + 0.730284 w_k1_MAT_s112
                                 + 1.15115 w_k1_MSFT_s112
                                 + 0.92446 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.32881 w_k1_ADM_s112
                                 + 0.93595 w_k1_AXP_s112
                                 + 0.679211 w_k1_DIS_s112
                                 + 0.838239 w_k1_FDX_s112
                                 + 0.742472 w_k1_IBM_s112
                                 + 0.832804 w_k1_INTC_s112
                                 + 0.77513 w_k1_KO_s112
                                 + 0.820318 w_k1_MAT_s112
                                 + 0.878511 w_k1_MSFT_s112
                                 + 0.854976 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.751638 w_k1_ADM_s113
                                 + 0.788255 w_k1_AXP_s113
                                 + 0.883606 w_k1_DIS_s113
                                 + 0.878585 w_k1_FDX_s113
                                 + 0.677235 w_k1_IBM_s113
                                 + 0.871575 w_k1_INTC_s113
                                 + 0.716561 w_k1_KO_s113
                                 + 0.892382 w_k1_MAT_s113
                                 + 0.866183 w_k1_MSFT_s113
                                 + 0.785939 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.796911 w_k1_ADM_s113
                                 + 0.812883 w_k1_AXP_s113
                                 + 0.678611 w_k1_DIS_s113
                                 + 0.778349 w_k1_FDX_s113
                                 + 0.765015 w_k1_IBM_s113
                                 + 1.04439 w_k1_INTC_s113
                                 + 0.80944 w_k1_KO_s113 + 1.10368 w_k1_MAT_s113
                                 + 0.834012 w_k1_MSFT_s113
                                 + 0.699093 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.787673 w_k1_ADM_s113
                                 + 0.78309 w_k1_AXP_s113
                                 + 0.931941 w_k1_DIS_s113
                                 + 0.844516 w_k1_FDX_s113
                                 + 1.05831 w_k1_IBM_s113
                                 + 0.905661 w_k1_INTC_s113
                                 + 0.963109 w_k1_KO_s113
                                 + 1.04988 w_k1_MAT_s113
                                 + 1.01245 w_k1_MSFT_s113
                                 + 0.835505 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.0619 w_k1_ADM_s113
                                 + 1.05255 w_k1_AXP_s113
                                 + 1.47235 w_k1_DIS_s113
                                 + 1.18715 w_k1_FDX_s113
                                 + 1.10689 w_k1_IBM_s113
                                 + 1.03886 w_k1_INTC_s113
                                 + 1.19046 w_k1_KO_s113 + 1.12914 w_k1_MAT_s113
                                 + 1.1764 w_k1_MSFT_s113
                                 + 0.955217 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.05796 w_k1_ADM_s113
                                 + 1.13147 w_k1_AXP_s113 + 1.4711 w_k1_DIS_s113
                                 + 1.06621 w_k1_FDX_s113
                                 + 0.835477 w_k1_IBM_s113
                                 + 1.05226 w_k1_INTC_s113
                                 + 1.21844 w_k1_KO_s113 + 1.06351 w_k1_MAT_s113
                                 + 1.17438 w_k1_MSFT_s113
                                 + 0.90718 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.12809 w_k1_ADM_s113
                                 + 1.07409 w_k1_AXP_s113
                                 + 1.25775 w_k1_DIS_s113 + 1.2169 w_k1_FDX_s113
                                 + 0.824621 w_k1_IBM_s113
                                 + 1.48029 w_k1_INTC_s113
                                 + 0.92867 w_k1_KO_s113
                                 + 0.694587 w_k1_MAT_s113
                                 + 1.22196 w_k1_MSFT_s113
                                 + 1.11199 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.05405 w_k1_ADM_s113
                                 + 1.31022 w_k1_AXP_s113 + 1.0921 w_k1_DIS_s113
                                 + 1.07309 w_k1_FDX_s113
                                 + 1.08294 w_k1_IBM_s113
                                 + 1.13799 w_k1_INTC_s113
                                 + 1.01824 w_k1_KO_s113 + 1.3536 w_k1_MAT_s113
                                 + 1.08822 w_k1_MSFT_s113
                                 + 1.01069 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.37612 w_k1_ADM_s113
                                 + 1.30005 w_k1_AXP_s113
                                 + 1.45654 w_k1_DIS_s113
                                 + 1.22074 w_k1_FDX_s113 + 1.3831 w_k1_IBM_s113
                                 + 1.26253 w_k1_INTC_s113
                                 + 1.32288 w_k1_KO_s113 + 1.52494 w_k1_MAT_s113
                                 + 1.32062 w_k1_MSFT_s113 + 1.1933 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09233 w_k1_ADM_s114
                                 + 0.948033 w_k1_AXP_s114
                                 + 1.20144 w_k1_DIS_s114
                                 + 1.23989 w_k1_FDX_s114
                                 + 1.07965 w_k1_IBM_s114
                                 + 1.34688 w_k1_INTC_s114
                                 + 1.09955 w_k1_KO_s114 + 0.97081 w_k1_MAT_s114
                                 + 1.1032 w_k1_MSFT_s114 + 1.13514 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.849904 w_k1_ADM_s114
                                 + 1.03145 w_k1_AXP_s114
                                 + 1.14079 w_k1_DIS_s114
                                 + 1.14914 w_k1_FDX_s114
                                 + 1.01158 w_k1_IBM_s114
                                 + 1.38008 w_k1_INTC_s114
                                 + 1.09324 w_k1_KO_s114 + 1.07401 w_k1_MAT_s114
                                 + 1.11281 w_k1_MSFT_s114
                                 + 1.08609 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.83722 w_k1_ADM_s114
                                 + 0.899148 w_k1_AXP_s114
                                 + 0.752496 w_k1_DIS_s114
                                 + 0.862882 w_k1_FDX_s114
                                 + 0.891335 w_k1_IBM_s114
                                 + 1.08459 w_k1_INTC_s114
                                 + 0.718007 w_k1_KO_s114
                                 + 0.892272 w_k1_MAT_s114
                                 + 0.886268 w_k1_MSFT_s114
                                 + 0.961528 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.703236 w_k1_ADM_s114
                                 + 0.693055 w_k1_AXP_s114
                                 + 0.902799 w_k1_DIS_s114
                                 + 0.924769 w_k1_FDX_s114
                                 + 0.973618 w_k1_IBM_s114
                                 + 1.29865 w_k1_INTC_s114
                                 + 0.913299 w_k1_KO_s114
                                 + 0.702563 w_k1_MAT_s114
                                 + 1.14771 w_k1_MSFT_s114
                                 + 0.956647 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14769 w_k1_ADM_s114
                                 + 1.14922 w_k1_AXP_s114
                                 + 1.64547 w_k1_DIS_s114
                                 + 1.35962 w_k1_FDX_s114
                                 + 1.02142 w_k1_IBM_s114
                                 + 1.5128 w_k1_INTC_s114 + 1.20973 w_k1_KO_s114
                                 + 0.708714 w_k1_MAT_s114
                                 + 1.32562 w_k1_MSFT_s114
                                 + 1.15604 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.681067 w_k1_ADM_s114
                                 + 0.67816 w_k1_AXP_s114
                                 + 0.665046 w_k1_DIS_s114
                                 + 0.792875 w_k1_FDX_s114
                                 + 0.65849 w_k1_IBM_s114
                                 + 0.860819 w_k1_INTC_s114
                                 + 0.663536 w_k1_KO_s114
                                 + 0.585005 w_k1_MAT_s114
                                 + 0.815829 w_k1_MSFT_s114 + 0.827 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.863921 w_k1_ADM_s114
                                 + 0.967814 w_k1_AXP_s114
                                 + 0.948026 w_k1_DIS_s114
                                 + 0.923938 w_k1_FDX_s114
                                 + 1.31148 w_k1_IBM_s114
                                 + 1.16916 w_k1_INTC_s114
                                 + 1.10766 w_k1_KO_s114 + 1.28344 w_k1_MAT_s114
                                 + 0.847964 w_k1_MSFT_s114
                                 + 0.852073 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.06426 w_k1_ADM_s114
                                 + 1.50438 w_k1_AXP_s114
                                 + 1.61505 w_k1_DIS_s114
                                 + 1.29879 w_k1_FDX_s114
                                 + 1.59688 w_k1_IBM_s114
                                 + 1.48031 w_k1_INTC_s114
                                 + 1.51873 w_k1_KO_s114 + 1.1402 w_k1_MAT_s114
                                 + 1.32364 w_k1_MSFT_s114
                                 + 1.19841 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.29836 w_k1_ADM_s115
                                 + 0.847791 w_k1_AXP_s115
                                 + 1.02653 w_k1_DIS_s115
                                 + 1.00698 w_k1_FDX_s115
                                 + 0.679939 w_k1_IBM_s115
                                 + 0.881358 w_k1_INTC_s115
                                 + 0.924026 w_k1_KO_s115
                                 + 0.723997 w_k1_MAT_s115
                                 + 0.995164 w_k1_MSFT_s115
                                 + 0.914719 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.39784 w_k1_ADM_s115
                                 + 1.63365 w_k1_AXP_s115 + 1.8835 w_k1_DIS_s115
                                 + 1.31799 w_k1_FDX_s115
                                 + 1.52104 w_k1_IBM_s115
                                 + 1.34541 w_k1_INTC_s115
                                 + 1.42189 w_k1_KO_s115 + 1.67809 w_k1_MAT_s115
                                 + 1.34007 w_k1_MSFT_s115
                                 + 1.17899 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.869294 w_k1_ADM_s115
                                 + 0.916245 w_k1_AXP_s115
                                 + 0.847069 w_k1_DIS_s115
                                 + 0.806402 w_k1_FDX_s115
                                 + 1.1841 w_k1_IBM_s115
                                 + 0.678994 w_k1_INTC_s115
                                 + 0.930536 w_k1_KO_s115
                                 + 1.34263 w_k1_MAT_s115
                                 + 0.718162 w_k1_MSFT_s115
                                 + 0.941782 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.714502 w_k1_ADM_s115
                                 + 0.930773 w_k1_AXP_s115
                                 + 1.07087 w_k1_DIS_s115
                                 + 0.876394 w_k1_FDX_s115
                                 + 1.67003 w_k1_IBM_s115
                                 + 0.875536 w_k1_INTC_s115
                                 + 0.986733 w_k1_KO_s115
                                 + 1.54806 w_k1_MAT_s115
                                 + 0.951904 w_k1_MSFT_s115
                                 + 0.876783 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.77423 w_k1_ADM_s115
                                 + 1.07432 w_k1_AXP_s115
                                 + 1.11043 w_k1_DIS_s115
                                 + 0.980602 w_k1_FDX_s115
                                 + 0.927154 w_k1_IBM_s115
                                 + 1.09283 w_k1_INTC_s115
                                 + 0.889254 w_k1_KO_s115
                                 + 1.07556 w_k1_MAT_s115
                                 + 0.891339 w_k1_MSFT_s115
                                 + 0.884427 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0307 w_k1_ADM_s115
                                 + 1.11861 w_k1_AXP_s115
                                 + 1.34984 w_k1_DIS_s115
                                 + 1.24468 w_k1_FDX_s115
                                 + 0.871358 w_k1_IBM_s115
                                 + 1.04057 w_k1_INTC_s115
                                 + 0.858585 w_k1_KO_s115
                                 + 1.01889 w_k1_MAT_s115
                                 + 1.06188 w_k1_MSFT_s115
                                 + 0.900635 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.21897 w_k1_ADM_s115
                                 + 1.4862 w_k1_AXP_s115 + 1.54371 w_k1_DIS_s115
                                 + 1.27975 w_k1_FDX_s115
                                 + 1.15889 w_k1_IBM_s115
                                 + 1.05935 w_k1_INTC_s115
                                 + 1.18445 w_k1_KO_s115 + 1.71155 w_k1_MAT_s115
                                 + 1.24499 w_k1_MSFT_s115
                                 + 1.14719 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.902858 w_k1_ADM_s115
                                 + 0.729841 w_k1_AXP_s115
                                 + 0.900786 w_k1_DIS_s115
                                 + 0.861908 w_k1_FDX_s115
                                 + 0.891022 w_k1_IBM_s115
                                 + 0.713155 w_k1_INTC_s115
                                 + 0.829421 w_k1_KO_s115
                                 + 0.824498 w_k1_MAT_s115
                                 + 0.802179 w_k1_MSFT_s115
                                 + 0.815738 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.582493 w_k1_ADM_s116
                                 + 0.558134 w_k1_AXP_s116
                                 + 0.480777 w_k1_DIS_s116
                                 + 0.682678 w_k1_FDX_s116
                                 + 0.642925 w_k1_IBM_s116
                                 + 0.681343 w_k1_INTC_s116
                                 + 0.591676 w_k1_KO_s116
                                 + 0.76354 w_k1_MAT_s116
                                 + 0.690556 w_k1_MSFT_s116
                                 + 0.713992 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0311 w_k1_ADM_s116
                                 + 1.12217 w_k1_AXP_s116
                                 + 0.985893 w_k1_DIS_s116
                                 + 1.19586 w_k1_FDX_s116
                                 + 0.928355 w_k1_IBM_s116
                                 + 1.11393 w_k1_INTC_s116
                                 + 0.870236 w_k1_KO_s116
                                 + 1.23433 w_k1_MAT_s116
                                 + 1.09751 w_k1_MSFT_s116
                                 + 1.03222 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.845314 w_k1_ADM_s116
                                 + 1.18588 w_k1_AXP_s116
                                 + 1.43845 w_k1_DIS_s116
                                 + 1.19569 w_k1_FDX_s116
                                 + 1.50635 w_k1_IBM_s116
                                 + 1.19465 w_k1_INTC_s116
                                 + 1.36007 w_k1_KO_s116 + 1.65554 w_k1_MAT_s116
                                 + 1.23403 w_k1_MSFT_s116
                                 + 1.09939 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.807814 w_k1_ADM_s116
                                 + 0.897014 w_k1_AXP_s116
                                 + 0.949069 w_k1_DIS_s116
                                 + 0.923712 w_k1_FDX_s116
                                 + 1.68721 w_k1_IBM_s116
                                 + 1.0705 w_k1_INTC_s116 + 1.11691 w_k1_KO_s116
                                 + 1.774 w_k1_MAT_s116 + 1.04104 w_k1_MSFT_s116
                                 + 1.06363 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.29255 w_k1_ADM_s116
                                 + 1.41518 w_k1_AXP_s116
                                 + 1.47376 w_k1_DIS_s116
                                 + 1.38887 w_k1_FDX_s116
                                 + 1.06757 w_k1_IBM_s116
                                 + 1.32527 w_k1_INTC_s116
                                 + 1.08384 w_k1_KO_s116 + 1.44972 w_k1_MAT_s116
                                 + 1.40638 w_k1_MSFT_s116
                                 + 1.35391 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.784322 w_k1_ADM_s116
                                 + 0.916303 w_k1_AXP_s116
                                 + 0.793094 w_k1_DIS_s116
                                 + 0.940617 w_k1_FDX_s116
                                 + 0.982619 w_k1_IBM_s116
                                 + 1.13985 w_k1_INTC_s116
                                 + 0.793402 w_k1_KO_s116 + 1.2577 w_k1_MAT_s116
                                 + 1.01065 w_k1_MSFT_s116
                                 + 0.939726 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.932981 w_k1_ADM_s116
                                 + 0.974629 w_k1_AXP_s116
                                 + 1.03843 w_k1_DIS_s116
                                 + 0.97392 w_k1_FDX_s116
                                 + 1.31616 w_k1_IBM_s116
                                 + 1.20845 w_k1_INTC_s116
                                 + 1.16677 w_k1_KO_s116 + 1.37598 w_k1_MAT_s116
                                 + 1.21625 w_k1_MSFT_s116
                                 + 1.11428 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.758959 w_k1_ADM_s116
                                 + 0.68067 w_k1_AXP_s116
                                 + 0.931463 w_k1_DIS_s116
                                 + 0.745187 w_k1_FDX_s116
                                 + 1.02771 w_k1_IBM_s116
                                 + 0.834214 w_k1_INTC_s116
                                 + 0.817313 w_k1_KO_s116
                                 + 1.38755 w_k1_MAT_s116
                                 + 0.934213 w_k1_MSFT_s116
                                 + 0.882871 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.04653 w_k1_ADM_s117
                                 + 1.03496 w_k1_AXP_s117
                                 + 0.774667 w_k1_DIS_s117
                                 + 0.927297 w_k1_FDX_s117
                                 + 0.598669 w_k1_IBM_s117
                                 + 1.02778 w_k1_INTC_s117
                                 + 0.780098 w_k1_KO_s117
                                 + 0.459059 w_k1_MAT_s117
                                 + 0.902661 w_k1_MSFT_s117
                                 + 0.880229 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.903462 w_k1_ADM_s117
                                 + 1.25265 w_k1_AXP_s117
                                 + 1.57229 w_k1_DIS_s117
                                 + 1.22651 w_k1_FDX_s117
                                 + 1.27202 w_k1_IBM_s117
                                 + 1.29102 w_k1_INTC_s117
                                 + 1.38959 w_k1_KO_s117 + 1.21682 w_k1_MAT_s117
                                 + 1.52773 w_k1_MSFT_s117
                                 + 1.31591 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.21623 w_k1_ADM_s117
                                 + 1.73429 w_k1_AXP_s117
                                 + 1.51885 w_k1_DIS_s117 + 1.4854 w_k1_FDX_s117
                                 + 0.898743 w_k1_IBM_s117
                                 + 1.77288 w_k1_INTC_s117
                                 + 1.28863 w_k1_KO_s117 + 1.08446 w_k1_MAT_s117
                                 + 1.31379 w_k1_MSFT_s117
                                 + 1.20938 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.00763 w_k1_ADM_s117
                                 + 1.18197 w_k1_AXP_s117
                                 + 1.01585 w_k1_DIS_s117
                                 + 0.99213 w_k1_FDX_s117
                                 + 0.922614 w_k1_IBM_s117
                                 + 1.18626 w_k1_INTC_s117 + 1.0331 w_k1_KO_s117
                                 + 0.896396 w_k1_MAT_s117
                                 + 1.15363 w_k1_MSFT_s117
                                 + 1.00233 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.67434 w_k1_ADM_s117
                                 + 0.829633 w_k1_AXP_s117
                                 + 0.918362 w_k1_DIS_s117
                                 + 0.887261 w_k1_FDX_s117
                                 + 0.836147 w_k1_IBM_s117
                                 + 1.22648 w_k1_INTC_s117
                                 + 0.906728 w_k1_KO_s117
                                 + 0.541267 w_k1_MAT_s117
                                 + 1.04926 w_k1_MSFT_s117
                                 + 0.874929 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.759524 w_k1_ADM_s117
                                 + 1.22631 w_k1_AXP_s117
                                 + 0.951928 w_k1_DIS_s117
                                 + 0.862138 w_k1_FDX_s117
                                 + 1.25185 w_k1_IBM_s117
                                 + 0.949649 w_k1_INTC_s117
                                 + 1.06137 w_k1_KO_s117 + 1.19836 w_k1_MAT_s117
                                 + 0.812315 w_k1_MSFT_s117
                                 + 0.955662 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.762596 w_k1_ADM_s117
                                 + 1.48283 w_k1_AXP_s117 + 1.2683 w_k1_DIS_s117
                                 + 1.0473 w_k1_FDX_s117 + 1.74129 w_k1_IBM_s117
                                 + 1.47881 w_k1_INTC_s117
                                 + 1.25442 w_k1_KO_s117 + 1.58503 w_k1_MAT_s117
                                 + 1.02539 w_k1_MSFT_s117
                                 + 1.08486 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.607932 w_k1_ADM_s117
                                 + 0.946656 w_k1_AXP_s117
                                 + 1.0798 w_k1_DIS_s117
                                 + 0.938545 w_k1_FDX_s117
                                 + 1.06292 w_k1_IBM_s117
                                 + 0.973204 w_k1_INTC_s117
                                 + 0.98711 w_k1_KO_s117
                                 + 0.915639 w_k1_MAT_s117
                                 + 1.08616 w_k1_MSFT_s117
                                 + 0.934738 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0510333738208289 w_k1_ADM_s000
                                 + 0.0113405398543673 w_k1_ADM_s001
                                 + 0.0183610774173945 w_k1_ADM_s002
                                 + 0.0504937373615168 w_k1_ADM_s003
                                 + 0.0144817857256301 w_k1_ADM_s004
                                 + 0.0211817586411315 w_k1_ADM_s005
                                 + 0.0354971120340385 w_k1_ADM_s006
                                 + 0.0187029268255129 w_k1_ADM_s007
                                 + 0.0310233912726801 w_k1_ADM_s010
                                 + 0.028845572954045 w_k1_ADM_s011
                                 + 0.0779777750293036 w_k1_ADM_s012
                                 + 0.0323971187225787 w_k1_ADM_s013
                                 + 0.0440058587331067 w_k1_ADM_s014
                                 + 0.0496624959003979 w_k1_ADM_s015
                                 + 0.0857290963985149 w_k1_ADM_s016
                                 + 0.046489173095763 w_k1_ADM_s017
                                 + 0.0322200298601798 w_k1_ADM_s100
                                 + 0.0251071461138751 w_k1_ADM_s101
                                 + 0.0722194828815426 w_k1_ADM_s102
                                 + 0.0509714773893894 w_k1_ADM_s103
                                 + 0.0255341183087157 w_k1_ADM_s104
                                 + 0.0408568246203463 w_k1_ADM_s105
                                 + 0.0313775119805059 w_k1_ADM_s106
                                 + 0.0265465612180738 w_k1_ADM_s107
                                 + 0.0118039531969666 w_k1_ADM_s110
                                 + 0.00874771863330879 w_k1_ADM_s111
                                 + 0.0112481812459921 w_k1_ADM_s112
                                 + 0.00731285717381524 w_k1_ADM_s113
                                 + 0.0209490177332722 w_k1_ADM_s114
                                 + 0.00554518801730812 w_k1_ADM_s115
                                 + 0.00906342123195935 w_k1_ADM_s116
                                 + 0.00537700395641724 w_k1_ADM_s117
                                 + 0.0527813095989266 w_k1_AXP_s000
                                 + 0.0110050267493787 w_k1_AXP_s001
                                 + 0.0189586855535529 w_k1_AXP_s002
                                 + 0.0481439705076221 w_k1_AXP_s003
                                 + 0.016168656611422 w_k1_AXP_s004
                                 + 0.0174374372780075 w_k1_AXP_s005
                                 + 0.0375096942424329 w_k1_AXP_s006
                                 + 0.0182908453658527 w_k1_AXP_s007
                                 + 0.0275463609725971 w_k1_AXP_s010
                                 + 0.0323039979854099 w_k1_AXP_s011
                                 + 0.0731335950479954 w_k1_AXP_s012
                                 + 0.0372900298899019 w_k1_AXP_s013
                                 + 0.0467907055880378 w_k1_AXP_s014
                                 + 0.0463614368338037 w_k1_AXP_s015
                                 + 0.0944639684435857 w_k1_AXP_s016
                                 + 0.0427747712468794 w_k1_AXP_s017
                                 + 0.0347823883267807 w_k1_AXP_s100
                                 + 0.0250659658449974 w_k1_AXP_s101
                                 + 0.0696579326264076 w_k1_AXP_s102
                                 + 0.065587352311551 w_k1_AXP_s103
                                 + 0.0286376525533046 w_k1_AXP_s104
                                 + 0.0397981018234421 w_k1_AXP_s105
                                 + 0.0310302536365187 w_k1_AXP_s106
                                 + 0.0283471577838694 w_k1_AXP_s107
                                 + 0.0126982531877103 w_k1_AXP_s110
                                 + 0.00852134295641969 w_k1_AXP_s111
                                 + 0.00894647211495964 w_k1_AXP_s112
                                 + 0.00759730982747331 w_k1_AXP_s113
                                 + 0.0232003539927916 w_k1_AXP_s114
                                 + 0.00580479298479829 w_k1_AXP_s115
                                 + 0.0100368994466548 w_k1_AXP_s116
                                 + 0.00731079466952122 w_k1_AXP_s117
                                 + 0.051815320313698 w_k1_DIS_s000
                                 + 0.0123120523835899 w_k1_DIS_s001
                                 + 0.0183732299854892 w_k1_DIS_s002
                                 + 0.0481103491608696 w_k1_DIS_s003
                                 + 0.0164627834421522 w_k1_DIS_s004
                                 + 0.0178270880830684 w_k1_DIS_s005
                                 + 0.0341385441559625 w_k1_DIS_s006
                                 + 0.0185759909948514 w_k1_DIS_s007
                                 + 0.0310192856893913 w_k1_DIS_s010
                                 + 0.0329623209042234 w_k1_DIS_s011
                                 + 0.0678708332737479 w_k1_DIS_s012
                                 + 0.0369929954074426 w_k1_DIS_s013
                                 + 0.0485582636077596 w_k1_DIS_s014
                                 + 0.0540001727144855 w_k1_DIS_s015
                                 + 0.102185462626328 w_k1_DIS_s016
                                 + 0.0417077796132484 w_k1_DIS_s017
                                 + 0.0321837374871463 w_k1_DIS_s100
                                 + 0.0262168457800497 w_k1_DIS_s101
                                 + 0.0690858551950419 w_k1_DIS_s102
                                 + 0.0631724481885931 w_k1_DIS_s103
                                 + 0.0277135123643189 w_k1_DIS_s104
                                 + 0.04291265550318 w_k1_DIS_s105
                                 + 0.0310842042638863 w_k1_DIS_s106
                                 + 0.0275208231570786 w_k1_DIS_s107
                                 + 0.0128608103831419 w_k1_DIS_s110
                                 + 0.00858308433262845 w_k1_DIS_s111
                                 + 0.00963092745581346 w_k1_DIS_s112
                                 + 0.00824828493485692 w_k1_DIS_s113
                                 + 0.0248803935317951 w_k1_DIS_s114
                                 + 0.00640789237880098 w_k1_DIS_s115
                                 + 0.00996777089895489 w_k1_DIS_s116
                                 + 0.00674456217697773 w_k1_DIS_s117
                                 + 0.051652783556064 w_k1_FDX_s000
                                 + 0.0120993231756208 w_k1_FDX_s001
                                 + 0.0194626883950693 w_k1_FDX_s002
                                 + 0.0466191526242842 w_k1_FDX_s003
                                 + 0.0157004813655741 w_k1_FDX_s004
                                 + 0.0178767395015451 w_k1_FDX_s005
                                 + 0.0338940471633425 w_k1_FDX_s006
                                 + 0.0190995321790308 w_k1_FDX_s007
                                 + 0.0293046762436296 w_k1_FDX_s010
                                 + 0.0297814819673734 w_k1_FDX_s011
                                 + 0.06780960848244 w_k1_FDX_s012
                                 + 0.0346867963538094 w_k1_FDX_s013
                                 + 0.0450938442591511 w_k1_FDX_s014
                                 + 0.0515080967495888 w_k1_FDX_s015
                                 + 0.0949120781268526 w_k1_FDX_s016
                                 + 0.0421675107801268 w_k1_FDX_s017
                                 + 0.0337471740462756 w_k1_FDX_s100
                                 + 0.0247968117237923 w_k1_FDX_s101
                                 + 0.0715356128796773 w_k1_FDX_s102
                                 + 0.0590043684006445 w_k1_FDX_s103
                                 + 0.0287553282815206 w_k1_FDX_s104
                                 + 0.0399972671866233 w_k1_FDX_s105
                                 + 0.0300930246011673 w_k1_FDX_s106
                                 + 0.0267449452543834 w_k1_FDX_s107
                                 + 0.0132762219889843 w_k1_FDX_s110
                                 + 0.00850325007744745 w_k1_FDX_s111
                                 + 0.00935316626274349 w_k1_FDX_s112
                                 + 0.00743217841768869 w_k1_FDX_s113
                                 + 0.0247271608183701 w_k1_FDX_s114
                                 + 0.00560644171193592 w_k1_FDX_s115
                                 + 0.0104140667113518 w_k1_FDX_s116
                                 + 0.00631317862162927 w_k1_FDX_s117
                                 + 0.0522461648620509 w_k1_IBM_s000
                                 + 0.0111123598058112 w_k1_IBM_s001
                                 + 0.0196203607719446 w_k1_IBM_s002
                                 + 0.0473287759140286 w_k1_IBM_s003
                                 + 0.015175360967866 w_k1_IBM_s004
                                 + 0.018473210145451 w_k1_IBM_s005
                                 + 0.0359215058538387 w_k1_IBM_s006
                                 + 0.0182202244125644 w_k1_IBM_s007
                                 + 0.0303090623425374 w_k1_IBM_s010
                                 + 0.0299741589823929 w_k1_IBM_s011
                                 + 0.0683255582498432 w_k1_IBM_s012
                                 + 0.0361147733110877 w_k1_IBM_s013
                                 + 0.0477318685022471 w_k1_IBM_s014
                                 + 0.053359201566477 w_k1_IBM_s015
                                 + 0.0962517902756612 w_k1_IBM_s016
                                 + 0.0426877151393079 w_k1_IBM_s017
                                 + 0.0352868622991749 w_k1_IBM_s100
                                 + 0.0253644207504526 w_k1_IBM_s101
                                 + 0.0724223760926725 w_k1_IBM_s102
                                 + 0.0600405826899023 w_k1_IBM_s103
                                 + 0.029174419266486 w_k1_IBM_s104
                                 + 0.0401306296209588 w_k1_IBM_s105
                                 + 0.0311943905571434 w_k1_IBM_s106
                                 + 0.0281605456403294 w_k1_IBM_s107
                                 + 0.0137996791144947 w_k1_IBM_s110
                                 + 0.00926687725972268 w_k1_IBM_s111
                                 + 0.00947817716043623 w_k1_IBM_s112
                                 + 0.00737422753893807 w_k1_IBM_s113
                                 + 0.0251771146635937 w_k1_IBM_s114
                                 + 0.00565107354817139 w_k1_IBM_s115
                                 + 0.0111913009206823 w_k1_IBM_s116
                                 + 0.00647194917134391 w_k1_IBM_s117
                                 + 0.0561173261479296 w_k1_INTC_s000
                                 + 0.0117916313106397 w_k1_INTC_s001
                                 + 0.0192528136236562 w_k1_INTC_s002
                                 + 0.0504947284298172 w_k1_INTC_s003
                                 + 0.0132016367781969 w_k1_INTC_s004
                                 + 0.0179019226124636 w_k1_INTC_s005
                                 + 0.0402358240206549 w_k1_INTC_s006
                                 + 0.0186106002868819 w_k1_INTC_s007
                                 + 0.0288917508303842 w_k1_INTC_s010
                                 + 0.0269194856886386 w_k1_INTC_s011
                                 + 0.0705119537647558 w_k1_INTC_s012
                                 + 0.0332746543629267 w_k1_INTC_s013
                                 + 0.0481010070568721 w_k1_INTC_s014
                                 + 0.0517971041806351 w_k1_INTC_s015
                                 + 0.105713051311637 w_k1_INTC_s016
                                 + 0.0471959498027337 w_k1_INTC_s017
                                 + 0.034334437321788 w_k1_INTC_s100
                                 + 0.0260845433172933 w_k1_INTC_s101
                                 + 0.0702570849192064 w_k1_INTC_s102
                                 + 0.0612371542071399 w_k1_INTC_s103
                                 + 0.0270873927876476 w_k1_INTC_s104
                                 + 0.0420525046131604 w_k1_INTC_s105
                                 + 0.0322500820455266 w_k1_INTC_s106
                                 + 0.0303403241443453 w_k1_INTC_s107
                                 + 0.0128921124203193 w_k1_INTC_s110
                                 + 0.00892705861544288 w_k1_INTC_s111
                                 + 0.00991475052252754 w_k1_INTC_s112
                                 + 0.00796330253111915 w_k1_INTC_s113
                                 + 0.0296426371610227 w_k1_INTC_s114
                                 + 0.00507711606848114 w_k1_INTC_s115
                                 + 0.0112919973422306 w_k1_INTC_s116
                                 + 0.00756130393037227 w_k1_INTC_s117
                                 + 0.0535694220781727 w_k1_KO_s000
                                 + 0.0125862578901768 w_k1_KO_s001
                                 + 0.020801739621021 w_k1_KO_s002
                                 + 0.0498637455599006 w_k1_KO_s003
                                 + 0.0144688554439104 w_k1_KO_s004
                                 + 0.0184946252491466 w_k1_KO_s005
                                 + 0.0381184188381673 w_k1_KO_s006
                                 + 0.0182501355236945 w_k1_KO_s007
                                 + 0.0300593713664494 w_k1_KO_s010
                                 + 0.0294963319409331 w_k1_KO_s011
                                 + 0.0663721674901042 w_k1_KO_s012
                                 + 0.035313472686555 w_k1_KO_s013
                                 + 0.0496724375176671 w_k1_KO_s014
                                 + 0.0575124597941941 w_k1_KO_s015
                                 + 0.0885230136450588 w_k1_KO_s016
                                 + 0.0435386279280038 w_k1_KO_s017
                                 + 0.0344764590680004 w_k1_KO_s100
                                 + 0.0262141542068803 w_k1_KO_s101
                                 + 0.0756818286125632 w_k1_KO_s102
                                 + 0.053864088320815 w_k1_KO_s103
                                 + 0.0281987007689252 w_k1_KO_s104
                                 + 0.0405885413784491 w_k1_KO_s105
                                 + 0.0328689518287296 w_k1_KO_s106
                                 + 0.03073990125156 w_k1_KO_s107
                                 + 0.0133866928486536 w_k1_KO_s110
                                 + 0.00841165191345882 w_k1_KO_s111
                                 + 0.00896682186352411 w_k1_KO_s112
                                 + 0.00756914549699692 w_k1_KO_s113
                                 + 0.0240407622327331 w_k1_KO_s114
                                 + 0.00530519583099539 w_k1_KO_s115
                                 + 0.00952603296892952 w_k1_KO_s116
                                 + 0.00660189035198628 w_k1_KO_s117
                                 + 0.0507405935997782 w_k1_MAT_s000
                                 + 0.0112731780834118 w_k1_MAT_s001
                                 + 0.0214971469072781 w_k1_MAT_s002
                                 + 0.0534863189006058 w_k1_MAT_s003
                                 + 0.0148224992578836 w_k1_MAT_s004
                                 + 0.0223152437406103 w_k1_MAT_s005
                                 + 0.0334130141308814 w_k1_MAT_s006
                                 + 0.0210182226977431 w_k1_MAT_s007
                                 + 0.0283369361315868 w_k1_MAT_s010
                                 + 0.0359227475919673 w_k1_MAT_s011
                                 + 0.067191042252283 w_k1_MAT_s012
                                 + 0.0403084799648867 w_k1_MAT_s013
                                 + 0.0500697830287628 w_k1_MAT_s014
                                 + 0.0468006110893753 w_k1_MAT_s015
                                 + 0.0871610236002916 w_k1_MAT_s016
                                 + 0.0398444416753866 w_k1_MAT_s017
                                 + 0.0345547939018832 w_k1_MAT_s100
                                 + 0.0238937658989487 w_k1_MAT_s101
                                 + 0.0757468603021432 w_k1_MAT_s102
                                 + 0.0654790955946518 w_k1_MAT_s103
                                 + 0.035633213019058 w_k1_MAT_s104
                                 + 0.0396646978295904 w_k1_MAT_s105
                                 + 0.033435469327978 w_k1_MAT_s106
                                 + 0.0280183594648559 w_k1_MAT_s107
                                 + 0.0143497542237044 w_k1_MAT_s110
                                 + 0.00704762309266971 w_k1_MAT_s111
                                 + 0.00967500387875221 w_k1_MAT_s112
                                 + 0.00849378238119361 w_k1_MAT_s113
                                 + 0.0224694035971394 w_k1_MAT_s114
                                 + 0.0064715003652818 w_k1_MAT_s115
                                 + 0.0137410716421079 w_k1_MAT_s116
                                 + 0.00581697892311228 w_k1_MAT_s117
                                 + 0.0519015746563794 w_k1_MSFT_s000
                                 + 0.0121064903741682 w_k1_MSFT_s001
                                 + 0.0179484414547505 w_k1_MSFT_s002
                                 + 0.0449018417262015 w_k1_MSFT_s003
                                 + 0.0146515264741155 w_k1_MSFT_s004
                                 + 0.0189015566821179 w_k1_MSFT_s005
                                 + 0.0371891831264585 w_k1_MSFT_s006
                                 + 0.0187791066749484 w_k1_MSFT_s007
                                 + 0.0288609506719373 w_k1_MSFT_s010
                                 + 0.0299408730305582 w_k1_MSFT_s011
                                 + 0.0657918075823984 w_k1_MSFT_s012
                                 + 0.0353539110652774 w_k1_MSFT_s013
                                 + 0.0451246656056095 w_k1_MSFT_s014
                                 + 0.054837320988164 w_k1_MSFT_s015
                                 + 0.0985466289679316 w_k1_MSFT_s016
                                 + 0.0427001170116271 w_k1_MSFT_s017
                                 + 0.032954740385244 w_k1_MSFT_s100
                                 + 0.0236106177465621 w_k1_MSFT_s101
                                 + 0.0678954480117571 w_k1_MSFT_s102
                                 + 0.0558500843959809 w_k1_MSFT_s103
                                 + 0.027021818777792 w_k1_MSFT_s104
                                 + 0.04329131116397 w_k1_MSFT_s105
                                 + 0.0274508850815732 w_k1_MSFT_s106
                                 + 0.0271479942138115 w_k1_MSFT_s107
                                 + 0.0129064983150207 w_k1_MSFT_s110
                                 + 0.008121134970271 w_k1_MSFT_s111
                                 + 0.00986852244842011 w_k1_MSFT_s112
                                 + 0.00796141791200158 w_k1_MSFT_s113
                                 + 0.0249478168887661 w_k1_MSFT_s114
                                 + 0.00530572039159692 w_k1_MSFT_s115
                                 + 0.0110510120056547 w_k1_MSFT_s116
                                 + 0.00669636408978643 w_k1_MSFT_s117
                                 + 0.0504446668307833 w_k1_PG_s000
                                 + 0.0129810433104925 w_k1_PG_s001
                                 + 0.0189741806433836 w_k1_PG_s002
                                 + 0.0494441882080738 w_k1_PG_s003
                                 + 0.014521595308458 w_k1_PG_s004
                                 + 0.0174602486117424 w_k1_PG_s005
                                 + 0.0350743799518793 w_k1_PG_s006
                                 + 0.0184167263848872 w_k1_PG_s007
                                 + 0.0303229646315822 w_k1_PG_s010
                                 + 0.0312570132739571 w_k1_PG_s011
                                 + 0.0693233889624782 w_k1_PG_s012
                                 + 0.0371482091022208 w_k1_PG_s013
                                 + 0.0479834009075761 w_k1_PG_s014
                                 + 0.0533518771835084 w_k1_PG_s015
                                 + 0.0952439711457998 w_k1_PG_s016
                                 + 0.0454727015921451 w_k1_PG_s017
                                 + 0.0339760015826656 w_k1_PG_s100
                                 + 0.0240795558747063 w_k1_PG_s101
                                 + 0.0703009769474466 w_k1_PG_s102
                                 + 0.0576825118791436 w_k1_PG_s103
                                 + 0.0270947678218161 w_k1_PG_s104
                                 + 0.03839882214374 w_k1_PG_s105
                                 + 0.0310430431153511 w_k1_PG_s106
                                 + 0.0278175160238187 w_k1_PG_s107
                                 + 0.0127712464924653 w_k1_PG_s110
                                 + 0.00785981516117699 w_k1_PG_s111
                                 + 0.00873049910241928 w_k1_PG_s112
                                 + 0.00687446754714588 w_k1_PG_s113
                                 + 0.0241687302317895 w_k1_PG_s114
                                 + 0.00513530748287279 w_k1_PG_s115
                                 + 0.010425618119498 w_k1_PG_s116
                                 + 0.00628248273006477 w_k1_PG_s117 >= 1.06
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
