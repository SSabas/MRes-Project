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
 w_k1_ADM_s0_return_constraint:  1.05444 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.934805 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.917165 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.15738 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.963771 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.862132 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.800766 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.868346 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.24636 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.937495 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.846304 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.00422 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.19532 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.29267 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.970921 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.783657 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.774358 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.05575 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.34398 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.909414 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.19508 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.11881 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.08926 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.82461 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.18983 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.22715 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.811652 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.47709 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.687382 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.14499 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.785825 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.11878 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.839341 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.06105 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.48341 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.20881 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.991809 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.926409 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16906 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.809722 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.758358 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.13002 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.09222 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.02739 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.761447 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.04428 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.34018 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.735667 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.872756 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.05799 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.28824 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.65375 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.1297 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.805262 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.75645 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.04273 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.63331 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.10208 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.28695 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.50563 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.16094 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.766113 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.16945 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.26356 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.719863 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.74865 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.535231 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.882884 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.644859 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.906113 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.708594 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.15466 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.80773 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.14087 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.32054 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.782754 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.19099 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.804994 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.793334 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.26196 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.08404 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.01822 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.667917 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.07655 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.15876 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.656358 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.807702 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.02519 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.27519 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.80858 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.980304 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.767033 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.643956 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.985822 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.67136 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.95485 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.44809 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.68113 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.914994 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.613956 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.24752 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.23761 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.734145 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.9412 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.479314 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.02524 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.532646 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.09358 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.619404 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.2569 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 2.15067 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.14671 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.24027 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.689651 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1208 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.879407 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.84791 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.13589 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.04953 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.900175 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.75772 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.951374 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.139 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.881425 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.914888 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.02349 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.38572 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.44091 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.928612 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.739797 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.784232 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.982515 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.47871 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.10686 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.30594 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.25696 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.00325 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.81703 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.24388 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.07731 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.810003 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.57487 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.614456 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.13663 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.730544 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.06833 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.76973 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.24938 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.54206 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.17624 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.1462 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.837877 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14563 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.86733 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.824974 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.1818 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.880152 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.37221 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.970317 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.6921 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.13583 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.564901 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.76762 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.901908 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.817442 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.2646 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.69391 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.30461 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.770579 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.05266 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.13645 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.659516 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.799965 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.70191 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.06305 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.608278 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.12531 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.61704 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.993421 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.44451 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.755546 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.788935 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.605765 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.80021 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.816729 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.712836 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.65789 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.02411 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.42972 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.984781 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.0868 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.920854 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.820906 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.07862 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.03331 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.961713 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.75603 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.942746 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.2409 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.900007 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.960229 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.01605 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.27219 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.38262 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.1166 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.856744 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.785319 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.00114 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.34914 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.15976 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.08263 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.27259 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.970436 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.766976 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.14238 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.04335 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.799988 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.29985 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.63611 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.982508 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.778971 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.957868 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.796442 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.17098 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.29351 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.10692 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.24767 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.877648 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19618 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.82434 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.837038 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.12554 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.09004 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.25466 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.742628 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.23822 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.0425 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.647485 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.671206 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.88661 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.998645 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.44267 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.22834 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.03143 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.773939 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.985677 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.34574 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.751841 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.964627 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.58705 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.913843 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.647904 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.03239 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.30012 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.821527 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.51215 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.614505 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.895335 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.582547 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.891378 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.662364 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.962844 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.47908 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.1034 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.12818 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.850955 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.12497 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.763984 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.838378 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.43711 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.074 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.55085 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.965657 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.7137 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.03738 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.659556 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.87503 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.828069 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.865331 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.50435 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.4179 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.993696 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.593928 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.886077 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.976431 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.547811 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.694957 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.72223 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.00051 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.570681 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.895617 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.24786 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.743179 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.7192 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.563567 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.817988 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.449895 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.730444 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.650464 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.810387 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.66046 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.930566 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.29404 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.93964 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12453 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.873603 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.866114 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.09351 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.0628 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.970686 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.712268 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.95366 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.08726 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.817593 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.895354 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.989048 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.3203 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.45545 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.971847 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.783748 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.763529 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05724 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.30667 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.05546 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.24852 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2981 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.969538 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.771097 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.17551 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.05384 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.838718 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.43686 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.630523 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.12411 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.807745 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.04818 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.745114 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.10973 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.4422 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.12683 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.15297 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.85834 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.03707 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.885951 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.959914 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.11276 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.02907 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.995821 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.851294 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.02226 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.01284 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.832722 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.838739 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.940971 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.07991 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.2135 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.06585 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.865079 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.804263 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.0781 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.27756 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.983674 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.12963 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.49602 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.98336 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.791412 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.10504 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.09109 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.841256 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.33056 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.672588 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.09724 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.788316 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.00062 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.859604 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.08151 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.35328 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.17646 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.10183 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.98066 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.802998877852799 z_k1_s0000
                                 - 0.568998760701571 z_k1_s0001
                                 - 0.246999995826339 z_k1_s0002
                                 - 0.293000038522728 z_k1_s0003
                                 - 0.789001132509666 z_k1_s0004
                                 - 0.461998461299168 z_k1_s0005
                                 - 0.380999836746349 z_k1_s0006
                                 - 0.291000086688086 z_k1_s0007
                                 - 0.28999956014407 z_k1_s0010
                                 - 0.31099951491038 z_k1_s0011
                                 - 0.346998097568458 z_k1_s0012
                                 - 0.250999828949281 z_k1_s0013
                                 - 0.474999669856216 z_k1_s0014
                                 - 0.674999701058637 z_k1_s0015
                                 - 0.516999579388836 z_k1_s0016
                                 - 0.367999669332902 z_k1_s0017
                                 - 0.564000293240537 z_k1_s0020
                                 - 0.71899798221215 z_k1_s0021
                                 - 0.229999402968324 z_k1_s0022
                                 - 0.733997442113264 z_k1_s0023
                                 - 0.947997257687394 z_k1_s0024
                                 - 0.652996434655124 z_k1_s0025
                                 - 0.300999625994684 z_k1_s0026
                                 - 0.755996322968887 z_k1_s0027
                                 - 0.132999855816134 z_k1_s0030
                                 - 0.266999226443492 z_k1_s0031
                                 - 0.13699975465884 z_k1_s0032
                                 - 0.128999752573672 z_k1_s0033
                                 - 0.501000156133629 z_k1_s0034
                                 - 0.379000073234874 z_k1_s0035
                                 - 0.288998772278251 z_k1_s0036
                                 - 0.208998751426569 z_k1_s0037
                                 - 0.306000008288093 z_k1_s0040
                                 - 0.0900000214964874 z_k1_s0041
                                 - 0.196000450014346 z_k1_s0042
                                 - 0.0849998943023088 z_k1_s0043
                                 - 0.207000243841812 z_k1_s0044
                                 - 0.21599933879197 z_k1_s0045
                                 - 0.159999858216079 z_k1_s0046
                                 - 0.359999437986313 z_k1_s0047
                                 - 0.342999806376788 z_k1_s0050
                                 - 1.27499953603017 z_k1_s0051
                                 - 0.441999692943887 z_k1_s0052
                                 - 0.816999362964651 z_k1_s0053
                                 - 0.926999236928095 z_k1_s0054
                                 - 0.955000899590407 z_k1_s0055
                                 - 0.303999801217054 z_k1_s0056
                                 - 0.419999718151198 z_k1_s0057
                                 - 0.246000296985845 z_k1_s0060
                                 - 0.119999907607802 z_k1_s0061
                                 - 0.153999970339985 z_k1_s0062
                                 - 0.106999962071943 z_k1_s0063
                                 - 0.153999970339985 z_k1_s0064
                                 - 0.415000373451769 z_k1_s0065
                                 - 0.25299960385151 z_k1_s0066
                                 - 0.119999907607802 z_k1_s0067
                                 - 0.17199923956428 z_k1_s0070
                                 - 0.0909999106005137 z_k1_s0071
                                 - 0.106999885920394 z_k1_s0072
                                 - 0.19700011115681 z_k1_s0073
                                 - 0.154000010016071 z_k1_s0074
                                 - 0.248999827071654 z_k1_s0075
                                 - 0.0599998783245858 z_k1_s0076
                                 - 0.135000046604954 z_k1_s0077
                                 - 0.551999490684627 z_k1_s0100
                                 - 0.275001037575114 z_k1_s0101
                                 - 0.32899914302203 z_k1_s0102
                                 - 0.147000122629127 z_k1_s0103
                                 - 0.189000157666021 z_k1_s0104
                                 - 0.154999967669445 z_k1_s0105
                                 - 0.33499922262732 z_k1_s0106
                                 - 0.629000512285854 z_k1_s0107
                                 - 0.189000066927753 z_k1_s0110
                                 - 0.677000583041003 z_k1_s0111
                                 - 0.219000148579844 z_k1_s0112
                                 - 0.160000104010841 z_k1_s0113
                                 - 0.253999829506896 z_k1_s0114
                                 - 0.282999046623111 z_k1_s0115
                                 - 0.222000256185146 z_k1_s0116
                                 - 0.482001046703343 z_k1_s0117
                                 - 1.08600343601343 z_k1_s0120
                                 - 0.927001598496734 z_k1_s0121
                                 - 0.992997803953357 z_k1_s0122
                                 - 1.0069997975976 z_k1_s0123
                                 - 1.00500357222265 z_k1_s0124
                                 - 0.610000298553751 z_k1_s0125
                                 - 0.784999644689153 z_k1_s0126
                                 - 0.691000162344537 z_k1_s0127
                                 - 1.22099946294005 z_k1_s0130
                                 - 0.714999000301524 z_k1_s0131
                                 - 0.577999598706474 z_k1_s0132
                                 - 0.942997048616011 z_k1_s0133
                                 - 0.690999288278875 z_k1_s0134
                                 - 0.433999758371938 z_k1_s0135
                                 - 1.29599856301083 z_k1_s0136
                                 - 1.96300049156502 z_k1_s0137
                                 - 0.162000143916709 z_k1_s0140
                                 - 0.535998899502111 z_k1_s0141
                                 - 0.446999874603484 z_k1_s0142
                                 - 0.366999950200179 z_k1_s0143
                                 - 0.252000058870427 z_k1_s0144
                                 - 0.548998590217632 z_k1_s0145
                                 - 0.124999881880164 z_k1_s0146
                                 - 0.532001279282077 z_k1_s0147
                                 - 0.525000784181896 z_k1_s0150
                                 - 0.627000853946215 z_k1_s0151
                                 - 0.565001203941231 z_k1_s0152
                                 - 1.02300256359878 z_k1_s0153
                                 - 0.785999687393946 z_k1_s0154
                                 - 0.990004274131407 z_k1_s0155
                                 - 0.400000862186732 z_k1_s0156
                                 - 0.642999687687302 z_k1_s0157
                                 - 0.266999205555019 z_k1_s0160
                                 - 0.305999949737746 z_k1_s0161
                                 - 0.261000065835204 z_k1_s0162
                                 - 0.55200058227243 z_k1_s0163
                                 - 0.2830002554766 z_k1_s0164
                                 - 0.272999929275816 z_k1_s0165
                                 - 0.468001010187815 z_k1_s0166
                                 - 0.230000110613431 z_k1_s0167
                                 - 0.158000061377387 z_k1_s0170
                                 - 0.542000109365361 z_k1_s0171
                                 - 0.265999472429622 z_k1_s0172
                                 - 0.262999593278281 z_k1_s0173
                                 - 0.137000202017258 z_k1_s0174
                                 - 0.376000526984562 z_k1_s0175
                                 - 0.390999922741268 z_k1_s0176
                                 - 0.218000230506928 z_k1_s0177
                                 - 0.345001288047191 z_k1_s1000
                                 - 0.307001048799304 z_k1_s1001
                                 - 0.62500460107963 z_k1_s1002
                                 - 1.03400430593675 z_k1_s1003
                                 - 1.03500294373807 z_k1_s1004
                                 - 0.527000036025313 z_k1_s1005
                                 - 0.269000809551416 z_k1_s1006
                                 - 0.460001563995671 z_k1_s1007
                                 - 0.247998928254669 z_k1_s1010
                                 - 0.520000374405376 z_k1_s1011
                                 - 0.447999786718391 z_k1_s1012
                                 - 0.553999103923979 z_k1_s1013
                                 - 0.134999940912909 z_k1_s1014
                                 - 0.126000099652073 z_k1_s1015
                                 - 0.136000026475241 z_k1_s1016
                                 - 0.155000026759281 z_k1_s1017
                                 - 0.375001502912571 z_k1_s1020
                                 - 0.887001870948987 z_k1_s1021
                                 - 0.932001720545249 z_k1_s1022
                                 - 0.71400051493829 z_k1_s1023
                                 - 0.273001118491643 z_k1_s1024
                                 - 0.233000865337952 z_k1_s1025
                                 - 0.195000798922602 z_k1_s1026
                                 - 0.743002352240949 z_k1_s1027
                                 - 0.325999795834979 z_k1_s1030
                                 - 0.328999579005406 z_k1_s1031
                                 - 0.322999669164818 z_k1_s1032
                                 - 0.291999848738668 z_k1_s1033
                                 - 0.817999959757341 z_k1_s1034
                                 - 0.362000628877441 z_k1_s1035
                                 - 0.609000982061831 z_k1_s1036
                                 - 0.375998243007169 z_k1_s1037
                                 - 0.488001782402788 z_k1_s1040
                                 - 0.325000956157691 z_k1_s1041
                                 - 0.321000959836414 z_k1_s1042
                                 - 0.5920005711525 z_k1_s1043
                                 - 0.664001620538979 z_k1_s1044
                                 - 1.0690047343292 z_k1_s1045
                                 - 0.681003417760082 z_k1_s1046
                                 - 1.43800174540871 z_k1_s1047
                                 - 0.168000108234273 z_k1_s1050
                                 - 0.260000746838879 z_k1_s1051
                                 - 0.449000702702401 z_k1_s1052
                                 - 0.168000108234273 z_k1_s1053
                                 - 0.502000233880964 z_k1_s1054
                                 - 0.384001068992798 z_k1_s1055
                                 - 0.128000061397537 z_k1_s1056
                                 - 0.153000090670497 z_k1_s1057
                                 - 0.427999633925243 z_k1_s1060
                                 - 0.243000180196239 z_k1_s1061
                                 - 0.313999446870122 z_k1_s1062
                                 - 0.777999722602793 z_k1_s1063
                                 - 0.300001657725099 z_k1_s1064
                                 - 0.161000040791673 z_k1_s1065
                                 - 0.378001202972793 z_k1_s1066
                                 - 0.166000160687178 z_k1_s1067
                                 - 0.268000815875016 z_k1_s1070
                                 - 0.817003587669145 z_k1_s1071
                                 - 0.405002137151701 z_k1_s1072
                                 - 0.34000094534563 z_k1_s1073
                                 - 0.371001178504377 z_k1_s1074
                                 - 0.265000841788841 z_k1_s1075
                                 - 0.940004403708343 z_k1_s1076
                                 - 0.35100110079574 z_k1_s1077
                                 - 0.0810000998424691 z_k1_s1100
                                 - 0.10100001457757 z_k1_s1101
                                 - 0.21500030987819 z_k1_s1102
                                 - 0.394000373492919 z_k1_s1103
                                 - 0.186000969527671 z_k1_s1104
                                 - 0.309001053843965 z_k1_s1105
                                 - 0.232000173807981 z_k1_s1106
                                 - 0.299000278825842 z_k1_s1107
                                 - 0.251000132194571 z_k1_s1110
                                 - 0.0639999018903977 z_k1_s1111
                                 - 0.0849999352293606 z_k1_s1112
                                 - 0.0879999233492312 z_k1_s1113
                                 - 0.186000195430927 z_k1_s1114
                                 - 0.240999783462106 z_k1_s1115
                                 - 0.189000067050929 z_k1_s1116
                                 - 0.0609999137705271 z_k1_s1117
                                 - 0.119999759468984 z_k1_s1120
                                 - 0.172999893591525 z_k1_s1121
                                 - 0.155999650799744 z_k1_s1122
                                 - 0.177999320707899 z_k1_s1123
                                 - 0.203000107283284 z_k1_s1124
                                 - 0.0989998532843197 z_k1_s1125
                                 - 0.220999505299637 z_k1_s1126
                                 - 0.0669998687460105 z_k1_s1127
                                 - 0.13699962737219 z_k1_s1130
                                 - 0.0739999225279639 z_k1_s1131
                                 - 0.145000230196439 z_k1_s1132
                                 - 0.113000110897959 z_k1_s1133
                                 - 0.0699999267156416 z_k1_s1134
                                 - 0.0739999225279639 z_k1_s1135
                                 - 0.0459999518417073 z_k1_s1136
                                 - 0.105000272073215 z_k1_s1137
                                 - 0.0719998899812802 z_k1_s1140
                                 - 0.184999606479866 z_k1_s1141
                                 - 0.175999861731797 z_k1_s1142
                                 - 0.06799990076009 z_k1_s1143
                                 - 0.0609998566231141 z_k1_s1144
                                 - 0.0489998889595436 z_k1_s1145
                                 - 0.0439999654329492 z_k1_s1146
                                 - 0.184999606479866 z_k1_s1147
                                 - 0.287999759010447 z_k1_s1150
                                 - 0.212999857008749 z_k1_s1151
                                 - 0.824998546118896 z_k1_s1152
                                 - 0.82599843956031 z_k1_s1153
                                 - 0.243999936790717 z_k1_s1154
                                 - 0.205999851119265 z_k1_s1155
                                 - 0.422000140251838 z_k1_s1156
                                 - 0.735000618395765 z_k1_s1157
                                 - 0.483999937936613 z_k1_s1160
                                 - 0.543999682804851 z_k1_s1161
                                 - 0.121999980231957 z_k1_s1162
                                 - 0.223000049760031 z_k1_s1163
                                 - 0.446999097268118 z_k1_s1164
                                 - 0.264998873168049 z_k1_s1165
                                 - 0.217999821521074 z_k1_s1166
                                 - 0.191999932078171 z_k1_s1167
                                 - 0.457999019497284 z_k1_s1170
                                 - 0.214999274436776 z_k1_s1171
                                 - 0.127999815710937 z_k1_s1172
                                 - 0.0939999513813785 z_k1_s1173
                                 - 0.112999831447943 z_k1_s1174
                                 - 0.253999233520562 z_k1_s1175
                                 - 0.0979998471849486 z_k1_s1176
                                 - 0.139999803121333 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.10272 w_k1_ADM_s000
                                 + 1.05824 w_k1_AXP_s000
                                 + 0.859511 w_k1_DIS_s000
                                 + 0.962463 w_k1_FDX_s000
                                 + 0.840821 w_k1_IBM_s000
                                 + 1.0746 w_k1_INTC_s000
                                 + 0.812654 w_k1_KO_s000
                                 + 0.878676 w_k1_MAT_s000
                                 + 0.969739 w_k1_MSFT_s000
                                 + 1.01415 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.94883 w_k1_ADM_s000
                                 + 1.58171 w_k1_AXP_s000
                                 + 1.39499 w_k1_DIS_s000
                                 + 1.14511 w_k1_FDX_s000
                                 + 1.36871 w_k1_IBM_s000
                                 + 1.58065 w_k1_INTC_s000
                                 + 1.41611 w_k1_KO_s000 + 1.45894 w_k1_MAT_s000
                                 + 1.23925 w_k1_MSFT_s000
                                 + 1.12752 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.965957 w_k1_ADM_s000
                                 + 1.21344 w_k1_AXP_s000
                                 + 1.27357 w_k1_DIS_s000
                                 + 1.23851 w_k1_FDX_s000
                                 + 0.964402 w_k1_IBM_s000
                                 + 1.38013 w_k1_INTC_s000
                                 + 1.06738 w_k1_KO_s000
                                 + 0.928721 w_k1_MAT_s000
                                 + 1.30844 w_k1_MSFT_s000
                                 + 1.14778 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.15146 w_k1_ADM_s000
                                 + 1.13016 w_k1_AXP_s000
                                 + 1.12238 w_k1_DIS_s000
                                 + 0.999751 w_k1_FDX_s000
                                 + 1.12387 w_k1_IBM_s000
                                 + 1.21619 w_k1_INTC_s000
                                 + 1.15361 w_k1_KO_s000 + 0.74774 w_k1_MAT_s000
                                 + 1.26126 w_k1_MSFT_s000
                                 + 0.922338 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.726092 w_k1_ADM_s000
                                 + 0.659698 w_k1_AXP_s000
                                 + 0.613011 w_k1_DIS_s000
                                 + 0.717129 w_k1_FDX_s000
                                 + 0.700454 w_k1_IBM_s000
                                 + 1.0662 w_k1_INTC_s000 + 0.86004 w_k1_KO_s000
                                 + 0.621791 w_k1_MAT_s000
                                 + 0.864454 w_k1_MSFT_s000
                                 + 0.777002 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.975305 w_k1_ADM_s000
                                 + 1.00486 w_k1_AXP_s000
                                 + 0.908944 w_k1_DIS_s000
                                 + 0.942811 w_k1_FDX_s000
                                 + 1.08559 w_k1_IBM_s000
                                 + 1.05769 w_k1_INTC_s000
                                 + 1.06656 w_k1_KO_s000
                                 + 0.797819 w_k1_MAT_s000
                                 + 0.995453 w_k1_MSFT_s000
                                 + 1.01343 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.02076 w_k1_ADM_s000
                                 + 0.876788 w_k1_AXP_s000
                                 + 0.912346 w_k1_DIS_s000
                                 + 0.853807 w_k1_FDX_s000
                                 + 1.55034 w_k1_IBM_s000
                                 + 1.18366 w_k1_INTC_s000
                                 + 1.33011 w_k1_KO_s000 + 1.22834 w_k1_MAT_s000
                                 + 1.09505 w_k1_MSFT_s000
                                 + 1.04747 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.19903 w_k1_ADM_s000
                                 + 1.49331 w_k1_AXP_s000
                                 + 1.40696 w_k1_DIS_s000
                                 + 1.31882 w_k1_FDX_s000
                                 + 1.02111 w_k1_IBM_s000
                                 + 1.24537 w_k1_INTC_s000
                                 + 1.19998 w_k1_KO_s000 + 1.14508 w_k1_MAT_s000
                                 + 1.39643 w_k1_MSFT_s000
                                 + 1.29003 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.721851 w_k1_ADM_s001
                                 + 0.69886 w_k1_AXP_s001
                                 + 0.673515 w_k1_DIS_s001
                                 + 0.666488 w_k1_FDX_s001
                                 + 0.98226 w_k1_IBM_s001
                                 + 0.786599 w_k1_INTC_s001
                                 + 0.788977 w_k1_KO_s001
                                 + 0.814269 w_k1_MAT_s001
                                 + 0.719364 w_k1_MSFT_s001
                                 + 0.833795 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.25031 w_k1_ADM_s001
                                 + 1.12415 w_k1_AXP_s001
                                 + 0.873892 w_k1_DIS_s001
                                 + 1.01048 w_k1_FDX_s001
                                 + 0.779772 w_k1_IBM_s001
                                 + 1.15247 w_k1_INTC_s001
                                 + 0.831458 w_k1_KO_s001
                                 + 0.782949 w_k1_MAT_s001
                                 + 0.933437 w_k1_MSFT_s001
                                 + 0.957238 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.47135 w_k1_ADM_s001
                                 + 1.7103 w_k1_AXP_s001 + 1.60858 w_k1_DIS_s001
                                 + 1.23496 w_k1_FDX_s001
                                 + 1.56687 w_k1_IBM_s001
                                 + 1.23552 w_k1_INTC_s001
                                 + 1.43998 w_k1_KO_s001 + 1.51065 w_k1_MAT_s001
                                 + 1.30585 w_k1_MSFT_s001
                                 + 1.24352 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.978271 w_k1_ADM_s001
                                 + 0.810391 w_k1_AXP_s001
                                 + 1.0677 w_k1_DIS_s001 + 0.92275 w_k1_FDX_s001
                                 + 0.69057 w_k1_IBM_s001
                                 + 1.04844 w_k1_INTC_s001
                                 + 0.899418 w_k1_KO_s001
                                 + 0.477366 w_k1_MAT_s001
                                 + 0.947474 w_k1_MSFT_s001
                                 + 0.871147 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.840841 w_k1_ADM_s001
                                 + 0.784762 w_k1_AXP_s001
                                 + 0.637806 w_k1_DIS_s001
                                 + 0.773192 w_k1_FDX_s001
                                 + 0.753037 w_k1_IBM_s001
                                 + 0.950057 w_k1_INTC_s001
                                 + 0.769484 w_k1_KO_s001
                                 + 0.699706 w_k1_MAT_s001
                                 + 0.792663 w_k1_MSFT_s001
                                 + 0.809362 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.2257 w_k1_ADM_s001
                                 + 1.3018 w_k1_AXP_s001 + 1.13425 w_k1_DIS_s001
                                 + 0.977976 w_k1_FDX_s001
                                 + 1.07151 w_k1_IBM_s001
                                 + 1.21429 w_k1_INTC_s001
                                 + 1.07144 w_k1_KO_s001
                                 + 0.933676 w_k1_MAT_s001
                                 + 1.00942 w_k1_MSFT_s001
                                 + 1.08324 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.808895 w_k1_ADM_s001
                                 + 1.17154 w_k1_AXP_s001
                                 + 0.961698 w_k1_DIS_s001
                                 + 0.851021 w_k1_FDX_s001
                                 + 1.38888 w_k1_IBM_s001
                                 + 1.20205 w_k1_INTC_s001
                                 + 1.02491 w_k1_KO_s001 + 1.36977 w_k1_MAT_s001
                                 + 0.922705 w_k1_MSFT_s001
                                 + 0.991705 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.24583 w_k1_ADM_s001
                                 + 1.39693 w_k1_AXP_s001
                                 + 1.54108 w_k1_DIS_s001 + 1.3281 w_k1_FDX_s001
                                 + 0.98126 w_k1_IBM_s001
                                 + 1.4612 w_k1_INTC_s001 + 1.12406 w_k1_KO_s001
                                 + 0.723129 w_k1_MAT_s001
                                 + 1.4807 w_k1_MSFT_s001 + 1.08189 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.907934 w_k1_ADM_s002
                                 + 1.0668 w_k1_AXP_s002 + 1.01663 w_k1_DIS_s002
                                 + 1.10538 w_k1_FDX_s002
                                 + 0.769502 w_k1_IBM_s002
                                 + 0.964884 w_k1_INTC_s002
                                 + 0.967002 w_k1_KO_s002
                                 + 1.08571 w_k1_MAT_s002
                                 + 0.978487 w_k1_MSFT_s002
                                 + 1.0605 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.26442 w_k1_ADM_s002
                                 + 1.17471 w_k1_AXP_s002
                                 + 1.42415 w_k1_DIS_s002
                                 + 1.26246 w_k1_FDX_s002 + 1.0451 w_k1_IBM_s002
                                 + 1.13252 w_k1_INTC_s002
                                 + 1.24946 w_k1_KO_s002 + 1.29972 w_k1_MAT_s002
                                 + 1.25057 w_k1_MSFT_s002
                                 + 1.22013 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.719109 w_k1_ADM_s002
                                 + 0.932072 w_k1_AXP_s002
                                 + 0.687782 w_k1_DIS_s002
                                 + 0.742191 w_k1_FDX_s002
                                 + 0.939562 w_k1_IBM_s002
                                 + 0.79632 w_k1_INTC_s002
                                 + 0.765998 w_k1_KO_s002 + 1.1772 w_k1_MAT_s002
                                 + 0.686933 w_k1_MSFT_s002
                                 + 0.89989 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20002 w_k1_ADM_s002
                                 + 1.26841 w_k1_AXP_s002
                                 + 1.45924 w_k1_DIS_s002
                                 + 1.24503 w_k1_FDX_s002
                                 + 1.55421 w_k1_IBM_s002
                                 + 1.0394 w_k1_INTC_s002 + 1.39313 w_k1_KO_s002
                                 + 1.59131 w_k1_MAT_s002
                                 + 1.06669 w_k1_MSFT_s002
                                 + 1.19905 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.826072 w_k1_ADM_s002
                                 + 0.826905 w_k1_AXP_s002
                                 + 0.898156 w_k1_DIS_s002
                                 + 0.830944 w_k1_FDX_s002
                                 + 0.958257 w_k1_IBM_s002
                                 + 0.882731 w_k1_INTC_s002
                                 + 0.995076 w_k1_KO_s002
                                 + 0.886207 w_k1_MAT_s002
                                 + 0.887078 w_k1_MSFT_s002
                                 + 1.04098 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.990584 w_k1_ADM_s002
                                 + 1.13039 w_k1_AXP_s002
                                 + 1.02344 w_k1_DIS_s002
                                 + 0.946433 w_k1_FDX_s002
                                 + 1.27757 w_k1_IBM_s002
                                 + 0.835231 w_k1_INTC_s002
                                 + 1.09085 w_k1_KO_s002 + 1.57508 w_k1_MAT_s002
                                 + 0.960613 w_k1_MSFT_s002
                                 + 1.08219 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.09026 w_k1_ADM_s002
                                 + 1.1398 w_k1_AXP_s002 + 1.19588 w_k1_DIS_s002
                                 + 1.1958 w_k1_FDX_s002
                                 + 0.715108 w_k1_IBM_s002
                                 + 1.05745 w_k1_INTC_s002
                                 + 1.00332 w_k1_KO_s002
                                 + 0.670957 w_k1_MAT_s002
                                 + 1.05763 w_k1_MSFT_s002
                                 + 1.00897 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.771041 w_k1_ADM_s002
                                 + 0.878152 w_k1_AXP_s002
                                 + 0.775056 w_k1_DIS_s002
                                 + 0.910758 w_k1_FDX_s002
                                 + 0.719719 w_k1_IBM_s002
                                 + 0.851208 w_k1_INTC_s002
                                 + 0.785185 w_k1_KO_s002
                                 + 0.745885 w_k1_MAT_s002
                                 + 0.839003 w_k1_MSFT_s002
                                 + 0.881534 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.37032 w_k1_ADM_s003
                                 + 1.41399 w_k1_AXP_s003
                                 + 1.81531 w_k1_DIS_s003
                                 + 1.47677 w_k1_FDX_s003
                                 + 0.810218 w_k1_IBM_s003
                                 + 1.15716 w_k1_INTC_s003
                                 + 0.971899 w_k1_KO_s003
                                 + 0.626903 w_k1_MAT_s003
                                 + 1.31168 w_k1_MSFT_s003
                                 + 1.05572 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.835285 w_k1_ADM_s003
                                 + 1.24675 w_k1_AXP_s003
                                 + 1.10946 w_k1_DIS_s003
                                 + 0.969578 w_k1_FDX_s003
                                 + 1.43323 w_k1_IBM_s003
                                 + 1.19538 w_k1_INTC_s003
                                 + 1.10054 w_k1_KO_s003 + 1.28985 w_k1_MAT_s003
                                 + 1.03756 w_k1_MSFT_s003
                                 + 1.06285 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.10537 w_k1_ADM_s003
                                 + 1.4701 w_k1_AXP_s003 + 1.3423 w_k1_DIS_s003
                                 + 1.2593 w_k1_FDX_s003 + 1.14082 w_k1_IBM_s003
                                 + 1.2403 w_k1_INTC_s003 + 1.20548 w_k1_KO_s003
                                 + 0.989715 w_k1_MAT_s003
                                 + 1.17118 w_k1_MSFT_s003
                                 + 1.15233 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.38906 w_k1_ADM_s003
                                 + 1.94392 w_k1_AXP_s003
                                 + 1.80896 w_k1_DIS_s003
                                 + 1.40164 w_k1_FDX_s003 + 1.4605 w_k1_IBM_s003
                                 + 1.31867 w_k1_INTC_s003
                                 + 1.35634 w_k1_KO_s003 + 1.59264 w_k1_MAT_s003
                                 + 1.54053 w_k1_MSFT_s003
                                 + 1.26808 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.758496 w_k1_ADM_s003
                                 + 0.776711 w_k1_AXP_s003
                                 + 0.653828 w_k1_DIS_s003
                                 + 0.788008 w_k1_FDX_s003
                                 + 0.814158 w_k1_IBM_s003
                                 + 0.955171 w_k1_INTC_s003
                                 + 0.802676 w_k1_KO_s003
                                 + 0.685717 w_k1_MAT_s003
                                 + 0.878898 w_k1_MSFT_s003
                                 + 0.813332 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.98418 w_k1_ADM_s003
                                 + 1.14404 w_k1_AXP_s003
                                 + 1.23374 w_k1_DIS_s003
                                 + 1.05093 w_k1_FDX_s003
                                 + 0.949455 w_k1_IBM_s003
                                 + 1.06875 w_k1_INTC_s003
                                 + 1.01698 w_k1_KO_s003
                                 + 0.654556 w_k1_MAT_s003
                                 + 1.14107 w_k1_MSFT_s003
                                 + 1.08536 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.96315 w_k1_ADM_s003
                                 + 1.02428 w_k1_AXP_s003
                                 + 0.954477 w_k1_DIS_s003
                                 + 1.04777 w_k1_FDX_s003
                                 + 0.733526 w_k1_IBM_s003
                                 + 1.21143 w_k1_INTC_s003
                                 + 0.859393 w_k1_KO_s003
                                 + 0.56631 w_k1_MAT_s003
                                 + 1.03811 w_k1_MSFT_s003
                                 + 0.995406 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.921407 w_k1_ADM_s003
                                 + 0.822048 w_k1_AXP_s003
                                 + 1.08585 w_k1_DIS_s003
                                 + 0.880491 w_k1_FDX_s003
                                 + 1.08259 w_k1_IBM_s003
                                 + 1.03412 w_k1_INTC_s003
                                 + 1.15469 w_k1_KO_s003
                                 + 0.909884 w_k1_MAT_s003
                                 + 1.06409 w_k1_MSFT_s003
                                 + 1.00205 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.929549 w_k1_ADM_s004
                                 + 0.87542 w_k1_AXP_s004
                                 + 1.04665 w_k1_DIS_s004
                                 + 1.03348 w_k1_FDX_s004
                                 + 0.985006 w_k1_IBM_s004
                                 + 0.884343 w_k1_INTC_s004
                                 + 1.08323 w_k1_KO_s004
                                 + 0.806088 w_k1_MAT_s004
                                 + 1.04672 w_k1_MSFT_s004
                                 + 1.06325 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.15738 w_k1_ADM_s004
                                 + 1.1668 w_k1_AXP_s004
                                 + 0.709319 w_k1_DIS_s004
                                 + 0.934774 w_k1_FDX_s004
                                 + 0.955263 w_k1_IBM_s004
                                 + 0.980958 w_k1_INTC_s004
                                 + 1.00693 w_k1_KO_s004
                                 + 0.932297 w_k1_MAT_s004
                                 + 0.867897 w_k1_MSFT_s004
                                 + 0.841204 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.682816 w_k1_ADM_s004
                                 + 0.738344 w_k1_AXP_s004
                                 + 0.743246 w_k1_DIS_s004
                                 + 0.800778 w_k1_FDX_s004
                                 + 0.734387 w_k1_IBM_s004
                                 + 0.862713 w_k1_INTC_s004
                                 + 0.853291 w_k1_KO_s004
                                 + 0.562371 w_k1_MAT_s004
                                 + 0.762453 w_k1_MSFT_s004
                                 + 0.855712 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15434 w_k1_ADM_s004
                                 + 1.22826 w_k1_AXP_s004
                                 + 1.24209 w_k1_DIS_s004
                                 + 1.03925 w_k1_FDX_s004
                                 + 1.78587 w_k1_IBM_s004
                                 + 1.15265 w_k1_INTC_s004
                                 + 1.46687 w_k1_KO_s004 + 1.33826 w_k1_MAT_s004
                                 + 1.07804 w_k1_MSFT_s004
                                 + 1.09146 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.582 w_k1_ADM_s004
                                 + 1.53882 w_k1_AXP_s004 + 1.4114 w_k1_DIS_s004
                                 + 1.41965 w_k1_FDX_s004
                                 + 0.985964 w_k1_IBM_s004
                                 + 1.10709 w_k1_INTC_s004
                                 + 1.30379 w_k1_KO_s004 + 1.22246 w_k1_MAT_s004
                                 + 1.26411 w_k1_MSFT_s004
                                 + 1.27421 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.14733 w_k1_ADM_s004
                                 + 1.14792 w_k1_AXP_s004
                                 + 0.847016 w_k1_DIS_s004
                                 + 1.02212 w_k1_FDX_s004
                                 + 0.840429 w_k1_IBM_s004
                                 + 1.04019 w_k1_INTC_s004
                                 + 0.916894 w_k1_KO_s004
                                 + 0.736668 w_k1_MAT_s004
                                 + 0.916243 w_k1_MSFT_s004
                                 + 0.945906 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.848512 w_k1_ADM_s004
                                 + 0.77123 w_k1_AXP_s004
                                 + 0.721847 w_k1_DIS_s004
                                 + 0.762633 w_k1_FDX_s004
                                 + 1.03268 w_k1_IBM_s004
                                 + 0.913768 w_k1_INTC_s004
                                 + 1.0336 w_k1_KO_s004 + 0.704207 w_k1_MAT_s004
                                 + 0.83562 w_k1_MSFT_s004
                                 + 0.893302 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.15406 w_k1_ADM_s004
                                 + 1.23885 w_k1_AXP_s004
                                 + 1.26137 w_k1_DIS_s004 + 1.0749 w_k1_FDX_s004
                                 + 1.30062 w_k1_IBM_s004
                                 + 1.22595 w_k1_INTC_s004
                                 + 1.41137 w_k1_KO_s004
                                 + 0.996673 w_k1_MAT_s004
                                 + 1.10551 w_k1_MSFT_s004
                                 + 1.10821 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.13731 w_k1_ADM_s005
                                 + 1.08076 w_k1_AXP_s005
                                 + 1.30102 w_k1_DIS_s005
                                 + 1.08518 w_k1_FDX_s005
                                 + 1.79811 w_k1_IBM_s005
                                 + 1.0061 w_k1_INTC_s005 + 1.37711 w_k1_KO_s005
                                 + 1.33889 w_k1_MAT_s005
                                 + 1.08433 w_k1_MSFT_s005 + 1.1067 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.818841 w_k1_ADM_s005
                                 + 0.90688 w_k1_AXP_s005
                                 + 0.824434 w_k1_DIS_s005
                                 + 0.881598 w_k1_FDX_s005
                                 + 1.30384 w_k1_IBM_s005
                                 + 1.1356 w_k1_INTC_s005 + 1.11934 w_k1_KO_s005
                                 + 0.877897 w_k1_MAT_s005
                                 + 0.92727 w_k1_MSFT_s005 + 1.0667 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.08888 w_k1_ADM_s005
                                 + 1.02988 w_k1_AXP_s005
                                 + 1.42847 w_k1_DIS_s005
                                 + 1.19118 w_k1_FDX_s005 + 1.0601 w_k1_IBM_s005
                                 + 1.19935 w_k1_INTC_s005
                                 + 1.20199 w_k1_KO_s005 + 1.02059 w_k1_MAT_s005
                                 + 1.15073 w_k1_MSFT_s005
                                 + 1.13574 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.971275 w_k1_ADM_s005
                                 + 0.921189 w_k1_AXP_s005
                                 + 1.11685 w_k1_DIS_s005
                                 + 1.18235 w_k1_FDX_s005
                                 + 0.736032 w_k1_IBM_s005
                                 + 1.20168 w_k1_INTC_s005
                                 + 1.05016 w_k1_KO_s005 + 1.00761 w_k1_MAT_s005
                                 + 1.19478 w_k1_MSFT_s005 + 1.0798 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.04453 w_k1_ADM_s005
                                 + 0.959937 w_k1_AXP_s005
                                 + 1.10138 w_k1_DIS_s005
                                 + 0.988638 w_k1_FDX_s005
                                 + 1.03924 w_k1_IBM_s005
                                 + 1.25778 w_k1_INTC_s005
                                 + 1.07711 w_k1_KO_s005
                                 + 0.699024 w_k1_MAT_s005
                                 + 1.0639 w_k1_MSFT_s005
                                 + 0.980417 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.787766 w_k1_ADM_s005
                                 + 0.741657 w_k1_AXP_s005
                                 + 0.627646 w_k1_DIS_s005
                                 + 0.793726 w_k1_FDX_s005
                                 + 0.807087 w_k1_IBM_s005
                                 + 0.909685 w_k1_INTC_s005
                                 + 0.814877 w_k1_KO_s005
                                 + 0.971331 w_k1_MAT_s005
                                 + 0.756666 w_k1_MSFT_s005
                                 + 0.878404 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.34169 w_k1_ADM_s005
                                 + 1.31942 w_k1_AXP_s005
                                 + 1.69029 w_k1_DIS_s005
                                 + 1.41479 w_k1_FDX_s005
                                 + 1.32594 w_k1_IBM_s005
                                 + 1.30672 w_k1_INTC_s005
                                 + 1.38051 w_k1_KO_s005 + 1.89242 w_k1_MAT_s005
                                 + 1.2858 w_k1_MSFT_s005 + 1.29936 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.975501 w_k1_ADM_s005
                                 + 0.64072 w_k1_AXP_s005
                                 + 0.84914 w_k1_DIS_s005
                                 + 0.931046 w_k1_FDX_s005
                                 + 0.574349 w_k1_IBM_s005
                                 + 1.11082 w_k1_INTC_s005
                                 + 0.726221 w_k1_KO_s005
                                 + 0.326953 w_k1_MAT_s005
                                 + 1.0217 w_k1_MSFT_s005
                                 + 0.910605 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.21347 w_k1_ADM_s006
                                 + 1.50467 w_k1_AXP_s006
                                 + 1.27958 w_k1_DIS_s006 + 1.2202 w_k1_FDX_s006
                                 + 1.25689 w_k1_IBM_s006
                                 + 1.20412 w_k1_INTC_s006 + 1.2212 w_k1_KO_s006
                                 + 1.60632 w_k1_MAT_s006
                                 + 1.18818 w_k1_MSFT_s006
                                 + 1.20393 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.931332 w_k1_ADM_s006
                                 + 0.714947 w_k1_AXP_s006
                                 + 0.581413 w_k1_DIS_s006
                                 + 0.802899 w_k1_FDX_s006
                                 + 0.684954 w_k1_IBM_s006
                                 + 0.854019 w_k1_INTC_s006
                                 + 0.712435 w_k1_KO_s006
                                 + 0.468407 w_k1_MAT_s006
                                 + 0.760075 w_k1_MSFT_s006
                                 + 0.795436 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.17233 w_k1_ADM_s006
                                 + 1.04824 w_k1_AXP_s006
                                 + 0.959499 w_k1_DIS_s006
                                 + 1.07701 w_k1_FDX_s006
                                 + 0.704397 w_k1_IBM_s006
                                 + 0.978011 w_k1_INTC_s006
                                 + 0.785598 w_k1_KO_s006
                                 + 0.857974 w_k1_MAT_s006
                                 + 0.986707 w_k1_MSFT_s006
                                 + 0.906939 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.934588 w_k1_ADM_s006
                                 + 0.601455 w_k1_AXP_s006
                                 + 1.07638 w_k1_DIS_s006
                                 + 0.937015 w_k1_FDX_s006
                                 + 0.856124 w_k1_IBM_s006
                                 + 1.09967 w_k1_INTC_s006
                                 + 0.976015 w_k1_KO_s006
                                 + 0.590717 w_k1_MAT_s006
                                 + 1.03505 w_k1_MSFT_s006
                                 + 0.947359 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.42713 w_k1_ADM_s006
                                 + 1.22694 w_k1_AXP_s006
                                 + 1.73506 w_k1_DIS_s006
                                 + 1.38754 w_k1_FDX_s006
                                 + 1.08711 w_k1_IBM_s006
                                 + 1.39472 w_k1_INTC_s006
                                 + 1.47093 w_k1_KO_s006 + 1.45867 w_k1_MAT_s006
                                 + 1.41903 w_k1_MSFT_s006 + 1.1681 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.922369 w_k1_ADM_s006
                                 + 0.874318 w_k1_AXP_s006
                                 + 0.815766 w_k1_DIS_s006
                                 + 0.887597 w_k1_FDX_s006
                                 + 1.04663 w_k1_IBM_s006
                                 + 0.964438 w_k1_INTC_s006
                                 + 1.03569 w_k1_KO_s006
                                 + 0.865685 w_k1_MAT_s006
                                 + 0.798778 w_k1_MSFT_s006
                                 + 0.992331 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.95224 w_k1_ADM_s006
                                 + 0.861229 w_k1_AXP_s006
                                 + 1.25576 w_k1_DIS_s006
                                 + 1.05135 w_k1_FDX_s006
                                 + 0.916198 w_k1_IBM_s006
                                 + 1.1866 w_k1_INTC_s006
                                 + 0.964127 w_k1_KO_s006
                                 + 0.685014 w_k1_MAT_s006
                                 + 1.04006 w_k1_MSFT_s006
                                 + 1.06489 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.01328 w_k1_ADM_s006
                                 + 0.942233 w_k1_AXP_s006
                                 + 1.3697 w_k1_DIS_s006 + 1.11098 w_k1_FDX_s006
                                 + 1.31687 w_k1_IBM_s006
                                 + 0.968623 w_k1_INTC_s006
                                 + 1.43106 w_k1_KO_s006 + 1.40437 w_k1_MAT_s006
                                 + 1.0954 w_k1_MSFT_s006 + 1.21175 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.01455 w_k1_ADM_s007
                                 + 1.10998 w_k1_AXP_s007
                                 + 1.29457 w_k1_DIS_s007
                                 + 1.07574 w_k1_FDX_s007
                                 + 1.48968 w_k1_IBM_s007
                                 + 0.992798 w_k1_INTC_s007
                                 + 1.11604 w_k1_KO_s007 + 1.31584 w_k1_MAT_s007
                                 + 0.952476 w_k1_MSFT_s007
                                 + 1.20398 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.19661 w_k1_ADM_s007
                                 + 0.967204 w_k1_AXP_s007
                                 + 1.24932 w_k1_DIS_s007
                                 + 1.11168 w_k1_FDX_s007
                                 + 0.904861 w_k1_IBM_s007
                                 + 1.04041 w_k1_INTC_s007
                                 + 1.04459 w_k1_KO_s007
                                 + 0.771049 w_k1_MAT_s007
                                 + 1.07808 w_k1_MSFT_s007
                                 + 1.04699 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.64479 w_k1_ADM_s007
                                 + 1.54885 w_k1_AXP_s007 + 1.8365 w_k1_DIS_s007
                                 + 1.40675 w_k1_FDX_s007
                                 + 1.28282 w_k1_IBM_s007
                                 + 1.17168 w_k1_INTC_s007
                                 + 1.30032 w_k1_KO_s007 + 1.11104 w_k1_MAT_s007
                                 + 1.22638 w_k1_MSFT_s007
                                 + 1.22786 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.43217 w_k1_ADM_s007
                                 + 1.37263 w_k1_AXP_s007
                                 + 1.21586 w_k1_DIS_s007
                                 + 1.41504 w_k1_FDX_s007
                                 + 1.05575 w_k1_IBM_s007
                                 + 1.19419 w_k1_INTC_s007
                                 + 0.979637 w_k1_KO_s007
                                 + 0.999375 w_k1_MAT_s007
                                 + 1.05292 w_k1_MSFT_s007
                                 + 1.16917 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.00864 w_k1_ADM_s007
                                 + 1.087 w_k1_AXP_s007 + 0.910041 w_k1_DIS_s007
                                 + 1.04821 w_k1_FDX_s007
                                 + 0.94571 w_k1_IBM_s007
                                 + 0.946858 w_k1_INTC_s007
                                 + 0.956747 w_k1_KO_s007 + 1.0499 w_k1_MAT_s007
                                 + 0.952973 w_k1_MSFT_s007
                                 + 1.11524 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.774277 w_k1_ADM_s007
                                 + 0.8244 w_k1_AXP_s007
                                 + 0.800291 w_k1_DIS_s007
                                 + 0.851766 w_k1_FDX_s007
                                 + 0.966819 w_k1_IBM_s007
                                 + 0.892784 w_k1_INTC_s007
                                 + 0.886957 w_k1_KO_s007
                                 + 1.01447 w_k1_MAT_s007
                                 + 0.941835 w_k1_MSFT_s007
                                 + 0.895337 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.831639 w_k1_ADM_s007
                                 + 0.547867 w_k1_AXP_s007
                                 + 0.591327 w_k1_DIS_s007
                                 + 0.715633 w_k1_FDX_s007
                                 + 0.928581 w_k1_IBM_s007
                                 + 0.701739 w_k1_INTC_s007
                                 + 0.65895 w_k1_KO_s007
                                 + 0.834866 w_k1_MAT_s007
                                 + 0.666633 w_k1_MSFT_s007
                                 + 0.769819 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.895646 w_k1_ADM_s007
                                 + 0.901338 w_k1_AXP_s007
                                 + 1.02469 w_k1_DIS_s007
                                 + 1.06937 w_k1_FDX_s007
                                 + 0.546722 w_k1_IBM_s007
                                 + 1.07459 w_k1_INTC_s007
                                 + 0.791194 w_k1_KO_s007
                                 + 0.49066 w_k1_MAT_s007
                                 + 0.989949 w_k1_MSFT_s007
                                 + 0.966885 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.06928 w_k1_ADM_s010
                                 + 0.874398 w_k1_AXP_s010
                                 + 0.78832 w_k1_DIS_s010
                                 + 0.881535 w_k1_FDX_s010
                                 + 0.708344 w_k1_IBM_s010
                                 + 0.878468 w_k1_INTC_s010
                                 + 0.83474 w_k1_KO_s010 + 0.83561 w_k1_MAT_s010
                                 + 0.975028 w_k1_MSFT_s010
                                 + 1.00523 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.2896 w_k1_ADM_s010
                                 + 1.45591 w_k1_AXP_s010
                                 + 1.38084 w_k1_DIS_s010
                                 + 1.13776 w_k1_FDX_s010
                                 + 1.28536 w_k1_IBM_s010
                                 + 1.12625 w_k1_INTC_s010
                                 + 1.23024 w_k1_KO_s010 + 1.46205 w_k1_MAT_s010
                                 + 1.01349 w_k1_MSFT_s010
                                 + 1.16903 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.26606 w_k1_ADM_s010
                                 + 1.23608 w_k1_AXP_s010
                                 + 0.955002 w_k1_DIS_s010
                                 + 0.944565 w_k1_FDX_s010
                                 + 1.15142 w_k1_IBM_s010
                                 + 0.844783 w_k1_INTC_s010
                                 + 1.14583 w_k1_KO_s010 + 1.36187 w_k1_MAT_s010
                                 + 0.909348 w_k1_MSFT_s010
                                 + 1.08106 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.772143 w_k1_ADM_s010
                                 + 0.818745 w_k1_AXP_s010
                                 + 0.863578 w_k1_DIS_s010
                                 + 0.813681 w_k1_FDX_s010
                                 + 1.39015 w_k1_IBM_s010
                                 + 0.776517 w_k1_INTC_s010
                                 + 1.03827 w_k1_KO_s010 + 1.07218 w_k1_MAT_s010
                                 + 0.958379 w_k1_MSFT_s010
                                 + 1.07078 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.50158 w_k1_ADM_s010
                                 + 1.3489 w_k1_AXP_s010 + 1.54362 w_k1_DIS_s010
                                 + 1.12804 w_k1_FDX_s010
                                 + 1.78128 w_k1_IBM_s010
                                 + 1.25681 w_k1_INTC_s010
                                 + 1.55595 w_k1_KO_s010 + 1.40917 w_k1_MAT_s010
                                 + 1.26593 w_k1_MSFT_s010
                                 + 1.29968 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.970809 w_k1_ADM_s010
                                 + 1.69079 w_k1_AXP_s010
                                 + 1.26662 w_k1_DIS_s010
                                 + 1.26274 w_k1_FDX_s010
                                 + 1.03109 w_k1_IBM_s010
                                 + 1.20988 w_k1_INTC_s010
                                 + 1.17627 w_k1_KO_s010 + 1.45197 w_k1_MAT_s010
                                 + 1.14995 w_k1_MSFT_s010
                                 + 1.18755 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.27876 w_k1_ADM_s010
                                 + 1.1006 w_k1_AXP_s010 + 1.57933 w_k1_DIS_s010
                                 + 1.25358 w_k1_FDX_s010
                                 + 0.900979 w_k1_IBM_s010
                                 + 1.01555 w_k1_INTC_s010
                                 + 1.12707 w_k1_KO_s010 + 1.00314 w_k1_MAT_s010
                                 + 1.26603 w_k1_MSFT_s010
                                 + 1.15947 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.827551 w_k1_ADM_s010
                                 + 0.769634 w_k1_AXP_s010
                                 + 0.866741 w_k1_DIS_s010
                                 + 0.866224 w_k1_FDX_s010
                                 + 0.95004 w_k1_IBM_s010
                                 + 0.765979 w_k1_INTC_s010
                                 + 0.974987 w_k1_KO_s010
                                 + 0.985644 w_k1_MAT_s010
                                 + 0.825521 w_k1_MSFT_s010
                                 + 1.01303 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.711996 w_k1_ADM_s011
                                 + 0.990478 w_k1_AXP_s011
                                 + 0.931843 w_k1_DIS_s011
                                 + 0.794635 w_k1_FDX_s011
                                 + 1.43091 w_k1_IBM_s011
                                 + 0.898184 w_k1_INTC_s011
                                 + 1.02765 w_k1_KO_s011 + 1.70629 w_k1_MAT_s011
                                 + 0.956794 w_k1_MSFT_s011
                                 + 1.05548 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.99569 w_k1_ADM_s011
                                 + 1.11223 w_k1_AXP_s011
                                 + 1.07485 w_k1_DIS_s011
                                 + 1.10008 w_k1_FDX_s011
                                 + 0.962353 w_k1_IBM_s011
                                 + 1.05645 w_k1_INTC_s011
                                 + 1.02401 w_k1_KO_s011 + 1.1006 w_k1_MAT_s011
                                 + 1.05916 w_k1_MSFT_s011
                                 + 1.07357 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.905389 w_k1_ADM_s011
                                 + 0.900461 w_k1_AXP_s011
                                 + 0.640399 w_k1_DIS_s011
                                 + 0.746576 w_k1_FDX_s011
                                 + 0.937922 w_k1_IBM_s011
                                 + 0.855573 w_k1_INTC_s011
                                 + 0.778812 w_k1_KO_s011
                                 + 0.810741 w_k1_MAT_s011
                                 + 0.738005 w_k1_MSFT_s011
                                 + 0.922429 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.03402 w_k1_ADM_s011
                                 + 0.86006 w_k1_AXP_s011
                                 + 0.573941 w_k1_DIS_s011
                                 + 0.787301 w_k1_FDX_s011
                                 + 0.772045 w_k1_IBM_s011
                                 + 0.748504 w_k1_INTC_s011
                                 + 0.764242 w_k1_KO_s011
                                 + 1.02082 w_k1_MAT_s011
                                 + 0.875919 w_k1_MSFT_s011
                                 + 0.940216 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.47747 w_k1_ADM_s011
                                 + 1.14134 w_k1_AXP_s011
                                 + 1.35408 w_k1_DIS_s011
                                 + 1.35737 w_k1_FDX_s011
                                 + 1.07024 w_k1_IBM_s011
                                 + 1.28709 w_k1_INTC_s011
                                 + 1.13657 w_k1_KO_s011 + 1.10118 w_k1_MAT_s011
                                 + 1.31657 w_k1_MSFT_s011
                                 + 1.23706 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.23975 w_k1_ADM_s011
                                 + 1.36622 w_k1_AXP_s011
                                 + 1.35053 w_k1_DIS_s011
                                 + 1.11531 w_k1_FDX_s011
                                 + 1.56239 w_k1_IBM_s011
                                 + 1.21755 w_k1_INTC_s011
                                 + 1.34318 w_k1_KO_s011 + 1.55732 w_k1_MAT_s011
                                 + 1.3279 w_k1_MSFT_s011 + 1.36979 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.940147 w_k1_ADM_s011
                                 + 0.831484 w_k1_AXP_s011
                                 + 0.692916 w_k1_DIS_s011
                                 + 0.766398 w_k1_FDX_s011
                                 + 1.16684 w_k1_IBM_s011
                                 + 0.879564 w_k1_INTC_s011
                                 + 0.9708 w_k1_KO_s011 + 1.10719 w_k1_MAT_s011
                                 + 0.875214 w_k1_MSFT_s011
                                 + 1.07579 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.03671 w_k1_ADM_s011
                                 + 0.821901 w_k1_AXP_s011
                                 + 0.763656 w_k1_DIS_s011
                                 + 0.909122 w_k1_FDX_s011
                                 + 0.685452 w_k1_IBM_s011
                                 + 0.875668 w_k1_INTC_s011
                                 + 0.671692 w_k1_KO_s011
                                 + 0.651904 w_k1_MAT_s011
                                 + 0.88387 w_k1_MSFT_s011
                                 + 1.00543 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.932929 w_k1_ADM_s012
                                 + 0.939543 w_k1_AXP_s012
                                 + 1.17301 w_k1_DIS_s012
                                 + 1.01614 w_k1_FDX_s012
                                 + 0.806615 w_k1_IBM_s012
                                 + 1.1925 w_k1_INTC_s012
                                 + 0.939256 w_k1_KO_s012
                                 + 0.642994 w_k1_MAT_s012
                                 + 1.22436 w_k1_MSFT_s012
                                 + 0.966823 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.34423 w_k1_ADM_s012
                                 + 1.08227 w_k1_AXP_s012
                                 + 1.05802 w_k1_DIS_s012
                                 + 1.21784 w_k1_FDX_s012
                                 + 0.782347 w_k1_IBM_s012
                                 + 1.15459 w_k1_INTC_s012
                                 + 0.929687 w_k1_KO_s012
                                 + 0.784137 w_k1_MAT_s012
                                 + 1.11589 w_k1_MSFT_s012
                                 + 1.15112 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.866815 w_k1_ADM_s012
                                 + 0.790666 w_k1_AXP_s012
                                 + 0.761492 w_k1_DIS_s012
                                 + 0.794431 w_k1_FDX_s012
                                 + 0.922003 w_k1_IBM_s012
                                 + 0.737914 w_k1_INTC_s012
                                 + 0.831493 w_k1_KO_s012
                                 + 1.00117 w_k1_MAT_s012
                                 + 0.874621 w_k1_MSFT_s012
                                 + 0.868397 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.16257 w_k1_ADM_s012
                                 + 1.06517 w_k1_AXP_s012
                                 + 1.10211 w_k1_DIS_s012
                                 + 0.966554 w_k1_FDX_s012
                                 + 1.33511 w_k1_IBM_s012
                                 + 0.973591 w_k1_INTC_s012
                                 + 1.06398 w_k1_KO_s012 + 1.45031 w_k1_MAT_s012
                                 + 1.00502 w_k1_MSFT_s012 + 1.1084 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.31057 w_k1_ADM_s012
                                 + 1.43232 w_k1_AXP_s012
                                 + 1.03159 w_k1_DIS_s012
                                 + 1.05178 w_k1_FDX_s012 + 1.1124 w_k1_IBM_s012
                                 + 1.10522 w_k1_INTC_s012
                                 + 1.06206 w_k1_KO_s012 + 1.70705 w_k1_MAT_s012
                                 + 1.01484 w_k1_MSFT_s012 + 1.1285 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.970404 w_k1_ADM_s012
                                 + 0.684343 w_k1_AXP_s012
                                 + 0.584344 w_k1_DIS_s012
                                 + 0.768511 w_k1_FDX_s012
                                 + 0.54973 w_k1_IBM_s012
                                 + 0.810373 w_k1_INTC_s012
                                 + 0.635459 w_k1_KO_s012
                                 + 0.714517 w_k1_MAT_s012
                                 + 0.788363 w_k1_MSFT_s012
                                 + 0.85075 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.22889 w_k1_ADM_s012
                                 + 1.63572 w_k1_AXP_s012
                                 + 1.68117 w_k1_DIS_s012
                                 + 1.38842 w_k1_FDX_s012
                                 + 1.26709 w_k1_IBM_s012
                                 + 1.56868 w_k1_INTC_s012
                                 + 1.29457 w_k1_KO_s012 + 1.67604 w_k1_MAT_s012
                                 + 1.56313 w_k1_MSFT_s012
                                 + 1.51391 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.751231 w_k1_ADM_s012
                                 + 0.904522 w_k1_AXP_s012
                                 + 0.862853 w_k1_DIS_s012
                                 + 0.76537 w_k1_FDX_s012
                                 + 1.33986 w_k1_IBM_s012
                                 + 1.05731 w_k1_INTC_s012
                                 + 1.07498 w_k1_KO_s012 + 1.41733 w_k1_MAT_s012
                                 + 0.932584 w_k1_MSFT_s012
                                 + 0.962529 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.21969 w_k1_ADM_s013
                                 + 1.37426 w_k1_AXP_s013
                                 + 1.16367 w_k1_DIS_s013
                                 + 1.12465 w_k1_FDX_s013
                                 + 1.17821 w_k1_IBM_s013
                                 + 1.09802 w_k1_INTC_s013
                                 + 1.23559 w_k1_KO_s013 + 1.60105 w_k1_MAT_s013
                                 + 0.993919 w_k1_MSFT_s013
                                 + 1.0662 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.916879 w_k1_ADM_s013
                                 + 1.13447 w_k1_AXP_s013
                                 + 1.33577 w_k1_DIS_s013
                                 + 1.00329 w_k1_FDX_s013
                                 + 1.48795 w_k1_IBM_s013
                                 + 1.24173 w_k1_INTC_s013
                                 + 1.31016 w_k1_KO_s013 + 1.49703 w_k1_MAT_s013
                                 + 1.08719 w_k1_MSFT_s013
                                 + 1.02382 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.788646 w_k1_ADM_s013
                                 + 0.666362 w_k1_AXP_s013
                                 + 0.838435 w_k1_DIS_s013
                                 + 0.834101 w_k1_FDX_s013
                                 + 0.741469 w_k1_IBM_s013
                                 + 0.848657 w_k1_INTC_s013
                                 + 0.934115 w_k1_KO_s013
                                 + 0.745317 w_k1_MAT_s013
                                 + 0.84296 w_k1_MSFT_s013
                                 + 0.854073 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.9719 w_k1_ADM_s013
                                 + 1.48871 w_k1_AXP_s013
                                 + 1.72418 w_k1_DIS_s013
                                 + 1.35279 w_k1_FDX_s013
                                 + 1.28673 w_k1_IBM_s013
                                 + 1.2443 w_k1_INTC_s013 + 1.55163 w_k1_KO_s013
                                 + 1.41903 w_k1_MAT_s013
                                 + 1.33557 w_k1_MSFT_s013
                                 + 1.23953 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.797206 w_k1_ADM_s013
                                 + 0.928701 w_k1_AXP_s013
                                 + 1.00191 w_k1_DIS_s013
                                 + 1.07331 w_k1_FDX_s013
                                 + 0.682979 w_k1_IBM_s013
                                 + 1.07847 w_k1_INTC_s013
                                 + 0.78712 w_k1_KO_s013
                                 + 0.741217 w_k1_MAT_s013
                                 + 0.933853 w_k1_MSFT_s013
                                 + 0.856006 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.773463 w_k1_ADM_s013
                                 + 0.680396 w_k1_AXP_s013
                                 + 0.619669 w_k1_DIS_s013
                                 + 0.737934 w_k1_FDX_s013
                                 + 0.706639 w_k1_IBM_s013
                                 + 0.781367 w_k1_INTC_s013
                                 + 0.668049 w_k1_KO_s013
                                 + 0.735868 w_k1_MAT_s013
                                 + 0.746972 w_k1_MSFT_s013
                                 + 0.754988 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.09812 w_k1_ADM_s013
                                 + 1.08031 w_k1_AXP_s013
                                 + 0.996676 w_k1_DIS_s013
                                 + 1.07462 w_k1_FDX_s013
                                 + 0.851402 w_k1_IBM_s013
                                 + 1.07875 w_k1_INTC_s013
                                 + 0.856112 w_k1_KO_s013
                                 + 0.935616 w_k1_MAT_s013
                                 + 0.977818 w_k1_MSFT_s013
                                 + 0.939548 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.891465 w_k1_ADM_s013
                                 + 0.869808 w_k1_AXP_s013
                                 + 0.87208 w_k1_DIS_s013
                                 + 0.900223 w_k1_FDX_s013
                                 + 1.04632 w_k1_IBM_s013
                                 + 0.871142 w_k1_INTC_s013
                                 + 0.976363 w_k1_KO_s013
                                 + 1.14378 w_k1_MAT_s013
                                 + 0.810666 w_k1_MSFT_s013
                                 + 0.901342 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.988994 w_k1_ADM_s014
                                 + 0.798597 w_k1_AXP_s014
                                 + 0.634882 w_k1_DIS_s014
                                 + 0.865189 w_k1_FDX_s014
                                 + 0.618603 w_k1_IBM_s014
                                 + 0.947041 w_k1_INTC_s014
                                 + 0.6737 w_k1_KO_s014 + 0.4957 w_k1_MAT_s014
                                 + 0.895006 w_k1_MSFT_s014
                                 + 0.846849 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.21587 w_k1_ADM_s014
                                 + 1.2721 w_k1_AXP_s014 + 1.73177 w_k1_DIS_s014
                                 + 1.25084 w_k1_FDX_s014
                                 + 1.12176 w_k1_IBM_s014
                                 + 1.1075 w_k1_INTC_s014 + 1.2698 w_k1_KO_s014
                                 + 1.30165 w_k1_MAT_s014
                                 + 1.34727 w_k1_MSFT_s014
                                 + 1.13978 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.27172 w_k1_ADM_s014
                                 + 1.36976 w_k1_AXP_s014
                                 + 0.996666 w_k1_DIS_s014
                                 + 1.11805 w_k1_FDX_s014
                                 + 0.910071 w_k1_IBM_s014
                                 + 1.01205 w_k1_INTC_s014
                                 + 0.840529 w_k1_KO_s014
                                 + 1.34787 w_k1_MAT_s014
                                 + 1.09399 w_k1_MSFT_s014
                                 + 0.997865 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.945588 w_k1_ADM_s014
                                 + 0.833406 w_k1_AXP_s014
                                 + 1.03101 w_k1_DIS_s014
                                 + 0.946785 w_k1_FDX_s014
                                 + 0.88917 w_k1_IBM_s014
                                 + 0.948709 w_k1_INTC_s014
                                 + 1.00806 w_k1_KO_s014
                                 + 0.921857 w_k1_MAT_s014
                                 + 1.14226 w_k1_MSFT_s014
                                 + 0.995002 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.775303 w_k1_ADM_s014
                                 + 0.99238 w_k1_AXP_s014
                                 + 0.786619 w_k1_DIS_s014
                                 + 0.868808 w_k1_FDX_s014
                                 + 0.962454 w_k1_IBM_s014
                                 + 0.911935 w_k1_INTC_s014
                                 + 0.769844 w_k1_KO_s014
                                 + 1.38147 w_k1_MAT_s014
                                 + 0.858683 w_k1_MSFT_s014
                                 + 0.868035 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.20137 w_k1_ADM_s014
                                 + 1.16612 w_k1_AXP_s014
                                 + 1.12195 w_k1_DIS_s014
                                 + 0.996916 w_k1_FDX_s014
                                 + 1.40871 w_k1_IBM_s014
                                 + 1.29747 w_k1_INTC_s014
                                 + 1.16208 w_k1_KO_s014 + 1.33697 w_k1_MAT_s014
                                 + 1.18051 w_k1_MSFT_s014
                                 + 1.12388 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.886476 w_k1_ADM_s014
                                 + 1.11637 w_k1_AXP_s014
                                 + 0.841586 w_k1_DIS_s014
                                 + 0.943793 w_k1_FDX_s014
                                 + 0.787261 w_k1_IBM_s014
                                 + 1.07378 w_k1_INTC_s014
                                 + 0.700453 w_k1_KO_s014
                                 + 1.05608 w_k1_MAT_s014
                                 + 0.938491 w_k1_MSFT_s014
                                 + 0.929915 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.678584 w_k1_ADM_s014
                                 + 0.671998 w_k1_AXP_s014
                                 + 0.671301 w_k1_DIS_s014
                                 + 0.721244 w_k1_FDX_s014
                                 + 0.879797 w_k1_IBM_s014
                                 + 0.964108 w_k1_INTC_s014
                                 + 0.789141 w_k1_KO_s014
                                 + 0.920609 w_k1_MAT_s014
                                 + 0.872679 w_k1_MSFT_s014
                                 + 0.853623 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.63315 w_k1_ADM_s015
                                 + 0.965299 w_k1_AXP_s015
                                 + 1.53688 w_k1_DIS_s015 + 1.2501 w_k1_FDX_s015
                                 + 1.02597 w_k1_IBM_s015
                                 + 0.889863 w_k1_INTC_s015
                                 + 1.1788 w_k1_KO_s015 + 0.864341 w_k1_MAT_s015
                                 + 1.1003 w_k1_MSFT_s015 + 1.11545 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.18713 w_k1_ADM_s015
                                 + 1.04845 w_k1_AXP_s015
                                 + 1.09384 w_k1_DIS_s015
                                 + 1.14377 w_k1_FDX_s015
                                 + 0.714041 w_k1_IBM_s015
                                 + 1.02459 w_k1_INTC_s015
                                 + 0.941536 w_k1_KO_s015
                                 + 0.870425 w_k1_MAT_s015
                                 + 1.12865 w_k1_MSFT_s015
                                 + 1.03112 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.62175 w_k1_ADM_s015
                                 + 1.63625 w_k1_AXP_s015
                                 + 1.36901 w_k1_DIS_s015
                                 + 1.31957 w_k1_FDX_s015
                                 + 1.16383 w_k1_IBM_s015
                                 + 1.01365 w_k1_INTC_s015 + 1.308 w_k1_KO_s015
                                 + 1.55444 w_k1_MAT_s015
                                 + 1.0779 w_k1_MSFT_s015 + 1.09361 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.26059 w_k1_ADM_s015
                                 + 1.12682 w_k1_AXP_s015
                                 + 1.31722 w_k1_DIS_s015
                                 + 1.10816 w_k1_FDX_s015
                                 + 1.37013 w_k1_IBM_s015
                                 + 1.11515 w_k1_INTC_s015
                                 + 1.40255 w_k1_KO_s015 + 1.66187 w_k1_MAT_s015
                                 + 1.19321 w_k1_MSFT_s015
                                 + 1.07694 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.997461 w_k1_ADM_s015
                                 + 0.58479 w_k1_AXP_s015
                                 + 0.718072 w_k1_DIS_s015
                                 + 0.751615 w_k1_FDX_s015
                                 + 0.920743 w_k1_IBM_s015
                                 + 0.686514 w_k1_INTC_s015
                                 + 0.881933 w_k1_KO_s015
                                 + 0.840007 w_k1_MAT_s015
                                 + 0.823939 w_k1_MSFT_s015
                                 + 0.826546 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.12367 w_k1_ADM_s015
                                 + 0.965315 w_k1_AXP_s015
                                 + 0.797448 w_k1_DIS_s015
                                 + 0.965987 w_k1_FDX_s015
                                 + 0.912315 w_k1_IBM_s015
                                 + 0.905659 w_k1_INTC_s015
                                 + 0.898004 w_k1_KO_s015
                                 + 1.27674 w_k1_MAT_s015
                                 + 0.821937 w_k1_MSFT_s015
                                 + 0.984112 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.08181 w_k1_ADM_s015
                                 + 0.662594 w_k1_AXP_s015
                                 + 0.585923 w_k1_DIS_s015
                                 + 0.800058 w_k1_FDX_s015
                                 + 0.689813 w_k1_IBM_s015
                                 + 0.672333 w_k1_INTC_s015
                                 + 0.727618 w_k1_KO_s015
                                 + 0.56358 w_k1_MAT_s015
                                 + 0.733085 w_k1_MSFT_s015
                                 + 0.809927 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.969578 w_k1_ADM_s015
                                 + 0.698398 w_k1_AXP_s015
                                 + 0.919891 w_k1_DIS_s015
                                 + 0.96809 w_k1_FDX_s015
                                 + 0.851368 w_k1_IBM_s015
                                 + 0.895776 w_k1_INTC_s015
                                 + 0.923553 w_k1_KO_s015
                                 + 0.766275 w_k1_MAT_s015
                                 + 1.0359 w_k1_MSFT_s015
                                 + 0.953862 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.34155 w_k1_ADM_s016
                                 + 1.08456 w_k1_AXP_s016
                                 + 1.03939 w_k1_DIS_s016
                                 + 1.25416 w_k1_FDX_s016
                                 + 0.685513 w_k1_IBM_s016
                                 + 1.22443 w_k1_INTC_s016
                                 + 1.00297 w_k1_KO_s016
                                 + 0.715724 w_k1_MAT_s016
                                 + 1.16706 w_k1_MSFT_s016
                                 + 1.04221 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.06435 w_k1_ADM_s016
                                 + 1.19171 w_k1_AXP_s016
                                 + 0.898485 w_k1_DIS_s016
                                 + 1.01631 w_k1_FDX_s016
                                 + 0.983091 w_k1_IBM_s016
                                 + 1.06493 w_k1_INTC_s016
                                 + 0.916303 w_k1_KO_s016
                                 + 0.900915 w_k1_MAT_s016
                                 + 0.912943 w_k1_MSFT_s016
                                 + 0.967855 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.84376 w_k1_ADM_s016
                                 + 0.822218 w_k1_AXP_s016
                                 + 1.06564 w_k1_DIS_s016
                                 + 0.963064 w_k1_FDX_s016
                                 + 0.980083 w_k1_IBM_s016
                                 + 0.976326 w_k1_INTC_s016
                                 + 1.1235 w_k1_KO_s016 + 1.09193 w_k1_MAT_s016
                                 + 1.07961 w_k1_MSFT_s016
                                 + 0.938483 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.03856 w_k1_ADM_s016
                                 + 1.10334 w_k1_AXP_s016 + 1.0235 w_k1_DIS_s016
                                 + 0.989286 w_k1_FDX_s016
                                 + 1.37896 w_k1_IBM_s016
                                 + 1.12537 w_k1_INTC_s016
                                 + 1.19999 w_k1_KO_s016 + 2.02867 w_k1_MAT_s016
                                 + 1.02593 w_k1_MSFT_s016
                                 + 1.20428 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.14539 w_k1_ADM_s016
                                 + 0.819354 w_k1_AXP_s016
                                 + 0.774446 w_k1_DIS_s016
                                 + 0.909077 w_k1_FDX_s016
                                 + 0.79855 w_k1_IBM_s016
                                 + 1.03277 w_k1_INTC_s016
                                 + 0.91741 w_k1_KO_s016
                                 + 0.740842 w_k1_MAT_s016
                                 + 0.894495 w_k1_MSFT_s016
                                 + 1.02185 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.727928 w_k1_ADM_s016
                                 + 0.65796 w_k1_AXP_s016
                                 + 0.642718 w_k1_DIS_s016
                                 + 0.720964 w_k1_FDX_s016
                                 + 0.782243 w_k1_IBM_s016
                                 + 0.779276 w_k1_INTC_s016
                                 + 0.786627 w_k1_KO_s016
                                 + 0.776485 w_k1_MAT_s016
                                 + 0.721881 w_k1_MSFT_s016
                                 + 0.859457 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.35206 w_k1_ADM_s016
                                 + 1.46926 w_k1_AXP_s016
                                 + 1.59885 w_k1_DIS_s016
                                 + 1.33386 w_k1_FDX_s016
                                 + 1.23185 w_k1_IBM_s016
                                 + 1.12212 w_k1_INTC_s016
                                 + 1.49024 w_k1_KO_s016 + 1.58576 w_k1_MAT_s016
                                 + 1.2885 w_k1_MSFT_s016 + 1.26729 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.86702 w_k1_ADM_s016
                                 + 0.892916 w_k1_AXP_s016
                                 + 0.73241 w_k1_DIS_s016
                                 + 0.847045 w_k1_FDX_s016
                                 + 0.98577 w_k1_IBM_s016
                                 + 1.11403 w_k1_INTC_s016
                                 + 1.10057 w_k1_KO_s016 + 1.16769 w_k1_MAT_s016
                                 + 0.868714 w_k1_MSFT_s016
                                 + 1.07584 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.22568 w_k1_ADM_s017
                                 + 0.909569 w_k1_AXP_s017
                                 + 1.33756 w_k1_DIS_s017
                                 + 1.14665 w_k1_FDX_s017
                                 + 1.15252 w_k1_IBM_s017
                                 + 1.02069 w_k1_INTC_s017
                                 + 1.18788 w_k1_KO_s017
                                 + 0.622841 w_k1_MAT_s017
                                 + 1.17737 w_k1_MSFT_s017
                                 + 0.917608 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.985737 w_k1_ADM_s017
                                 + 0.774096 w_k1_AXP_s017
                                 + 0.737492 w_k1_DIS_s017
                                 + 0.819134 w_k1_FDX_s017
                                 + 0.790887 w_k1_IBM_s017
                                 + 0.936919 w_k1_INTC_s017
                                 + 0.870984 w_k1_KO_s017
                                 + 0.667787 w_k1_MAT_s017
                                 + 0.840692 w_k1_MSFT_s017
                                 + 0.780496 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.51356 w_k1_ADM_s017
                                 + 1.80521 w_k1_AXP_s017
                                 + 1.73942 w_k1_DIS_s017
                                 + 1.40354 w_k1_FDX_s017
                                 + 1.60208 w_k1_IBM_s017
                                 + 1.03067 w_k1_INTC_s017
                                 + 1.37686 w_k1_KO_s017 + 1.61783 w_k1_MAT_s017
                                 + 1.1922 w_k1_MSFT_s017 + 1.08016 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.885501 w_k1_ADM_s017
                                 + 1.07641 w_k1_AXP_s017
                                 + 0.77072 w_k1_DIS_s017
                                 + 0.943926 w_k1_FDX_s017
                                 + 0.877839 w_k1_IBM_s017
                                 + 0.9684 w_k1_INTC_s017
                                 + 0.798917 w_k1_KO_s017
                                 + 0.892648 w_k1_MAT_s017
                                 + 0.857847 w_k1_MSFT_s017
                                 + 0.769092 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.47134 w_k1_ADM_s017
                                 + 1.54768 w_k1_AXP_s017
                                 + 1.42442 w_k1_DIS_s017
                                 + 1.42087 w_k1_FDX_s017
                                 + 1.02562 w_k1_IBM_s017
                                 + 1.09988 w_k1_INTC_s017
                                 + 1.22594 w_k1_KO_s017 + 1.10165 w_k1_MAT_s017
                                 + 1.21013 w_k1_MSFT_s017
                                 + 1.18588 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.881982 w_k1_ADM_s017
                                 + 0.906047 w_k1_AXP_s017
                                 + 1.01329 w_k1_DIS_s017
                                 + 0.905657 w_k1_FDX_s017
                                 + 1.42015 w_k1_IBM_s017
                                 + 1.00524 w_k1_INTC_s017
                                 + 1.15704 w_k1_KO_s017 + 1.55717 w_k1_MAT_s017
                                 + 0.917314 w_k1_MSFT_s017
                                 + 0.904359 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.55299 w_k1_ADM_s017
                                 + 1.16606 w_k1_AXP_s017
                                 + 0.947673 w_k1_DIS_s017
                                 + 1.02857 w_k1_FDX_s017
                                 + 1.01854 w_k1_IBM_s017
                                 + 0.920781 w_k1_INTC_s017
                                 + 1.02274 w_k1_KO_s017 + 1.02146 w_k1_MAT_s017
                                 + 0.870001 w_k1_MSFT_s017
                                 + 0.936747 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0411 w_k1_ADM_s017
                                 + 1.09167 w_k1_AXP_s017
                                 + 1.10954 w_k1_DIS_s017
                                 + 1.13552 w_k1_FDX_s017
                                 + 0.792535 w_k1_IBM_s017
                                 + 1.20362 w_k1_INTC_s017
                                 + 0.869924 w_k1_KO_s017
                                 + 0.704868 w_k1_MAT_s017
                                 + 1.06648 w_k1_MSFT_s017
                                 + 0.904496 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.770955 w_k1_ADM_s100
                                 + 0.879333 w_k1_AXP_s100
                                 + 0.914802 w_k1_DIS_s100
                                 + 0.842687 w_k1_FDX_s100
                                 + 1.4642 w_k1_IBM_s100
                                 + 1.04193 w_k1_INTC_s100
                                 + 1.17748 w_k1_KO_s100 + 1.1838 w_k1_MAT_s100
                                 + 0.874002 w_k1_MSFT_s100
                                 + 0.987603 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.05112 w_k1_ADM_s100
                                 + 0.979456 w_k1_AXP_s100
                                 + 1.36924 w_k1_DIS_s100
                                 + 1.10379 w_k1_FDX_s100
                                 + 1.06193 w_k1_IBM_s100
                                 + 1.25899 w_k1_INTC_s100
                                 + 1.17471 w_k1_KO_s100
                                 + 0.956595 w_k1_MAT_s100
                                 + 1.07213 w_k1_MSFT_s100
                                 + 1.04495 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.719686 w_k1_ADM_s100
                                 + 0.655355 w_k1_AXP_s100
                                 + 0.680198 w_k1_DIS_s100
                                 + 0.73747 w_k1_FDX_s100
                                 + 0.925388 w_k1_IBM_s100
                                 + 0.688496 w_k1_INTC_s100
                                 + 0.826793 w_k1_KO_s100
                                 + 1.05543 w_k1_MAT_s100
                                 + 0.790656 w_k1_MSFT_s100
                                 + 0.863293 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.998663 w_k1_ADM_s100
                                 + 1.13682 w_k1_AXP_s100
                                 + 1.12853 w_k1_DIS_s100
                                 + 1.12915 w_k1_FDX_s100
                                 + 1.23381 w_k1_IBM_s100
                                 + 1.1121 w_k1_INTC_s100 + 1.15647 w_k1_KO_s100
                                 + 1.24005 w_k1_MAT_s100
                                 + 1.01797 w_k1_MSFT_s100
                                 + 1.08587 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.880701 w_k1_ADM_s100
                                 + 0.927201 w_k1_AXP_s100
                                 + 0.911629 w_k1_DIS_s100
                                 + 0.984836 w_k1_FDX_s100
                                 + 0.830147 w_k1_IBM_s100
                                 + 0.926856 w_k1_INTC_s100
                                 + 0.868561 w_k1_KO_s100
                                 + 1.07161 w_k1_MAT_s100
                                 + 0.879094 w_k1_MSFT_s100
                                 + 1.01775 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.15164 w_k1_ADM_s100
                                 + 1.11629 w_k1_AXP_s100
                                 + 1.14217 w_k1_DIS_s100
                                 + 1.19191 w_k1_FDX_s100
                                 + 0.79331 w_k1_IBM_s100
                                 + 1.05137 w_k1_INTC_s100
                                 + 0.936575 w_k1_KO_s100
                                 + 0.934471 w_k1_MAT_s100
                                 + 1.01756 w_k1_MSFT_s100
                                 + 1.08691 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.24611 w_k1_ADM_s100
                                 + 1.29775 w_k1_AXP_s100
                                 + 1.52211 w_k1_DIS_s100
                                 + 1.19258 w_k1_FDX_s100
                                 + 1.43684 w_k1_IBM_s100
                                 + 1.21015 w_k1_INTC_s100
                                 + 1.38451 w_k1_KO_s100 + 1.85161 w_k1_MAT_s100
                                 + 1.32048 w_k1_MSFT_s100
                                 + 1.24558 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.779855 w_k1_ADM_s100
                                 + 0.81498 w_k1_AXP_s100
                                 + 0.719538 w_k1_DIS_s100
                                 + 0.871689 w_k1_FDX_s100
                                 + 0.678764 w_k1_IBM_s100
                                 + 0.899337 w_k1_INTC_s100
                                 + 0.815298 w_k1_KO_s100
                                 + 0.793034 w_k1_MAT_s100
                                 + 0.834338 w_k1_MSFT_s100
                                 + 0.92525 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.17249 w_k1_ADM_s101
                                 + 0.693225 w_k1_AXP_s101
                                 + 0.719607 w_k1_DIS_s101
                                 + 0.936703 w_k1_FDX_s101
                                 + 0.657262 w_k1_IBM_s101
                                 + 0.939065 w_k1_INTC_s101
                                 + 0.685926 w_k1_KO_s101
                                 + 0.450483 w_k1_MAT_s101
                                 + 0.91069 w_k1_MSFT_s101
                                 + 0.827313 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.50222 w_k1_ADM_s101
                                 + 1.43001 w_k1_AXP_s101
                                 + 1.48408 w_k1_DIS_s101 + 1.3965 w_k1_FDX_s101
                                 + 1.25713 w_k1_IBM_s101
                                 + 1.13172 w_k1_INTC_s101
                                 + 1.12354 w_k1_KO_s101 + 1.42189 w_k1_MAT_s101
                                 + 1.24862 w_k1_MSFT_s101
                                 + 1.15186 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.07987 w_k1_ADM_s101
                                 + 0.894883 w_k1_AXP_s101
                                 + 0.950772 w_k1_DIS_s101
                                 + 1.04538 w_k1_FDX_s101
                                 + 0.828819 w_k1_IBM_s101
                                 + 1.14631 w_k1_INTC_s101
                                 + 0.868017 w_k1_KO_s101
                                 + 0.79188 w_k1_MAT_s101
                                 + 1.10369 w_k1_MSFT_s101
                                 + 0.977665 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.23409 w_k1_ADM_s101
                                 + 0.94227 w_k1_AXP_s101
                                 + 1.14933 w_k1_DIS_s101
                                 + 1.05977 w_k1_FDX_s101 + 1.178 w_k1_IBM_s101
                                 + 1.13471 w_k1_INTC_s101
                                 + 1.06846 w_k1_KO_s101 + 1.18966 w_k1_MAT_s101
                                 + 1.14159 w_k1_MSFT_s101 + 1.1026 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.785952 w_k1_ADM_s101
                                 + 0.738713 w_k1_AXP_s101
                                 + 0.786323 w_k1_DIS_s101
                                 + 0.824143 w_k1_FDX_s101
                                 + 0.916865 w_k1_IBM_s101
                                 + 0.939464 w_k1_INTC_s101
                                 + 0.932263 w_k1_KO_s101
                                 + 0.775527 w_k1_MAT_s101
                                 + 0.965102 w_k1_MSFT_s101
                                 + 0.96605 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.820793 w_k1_ADM_s101
                                 + 0.583409 w_k1_AXP_s101
                                 + 0.58613 w_k1_DIS_s101
                                 + 0.698195 w_k1_FDX_s101
                                 + 0.671768 w_k1_IBM_s101
                                 + 0.947785 w_k1_INTC_s101
                                 + 0.662633 w_k1_KO_s101
                                 + 0.356274 w_k1_MAT_s101
                                 + 0.78674 w_k1_MSFT_s101
                                 + 0.800819 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04613 w_k1_ADM_s101
                                 + 0.808095 w_k1_AXP_s101
                                 + 0.705473 w_k1_DIS_s101
                                 + 0.821793 w_k1_FDX_s101
                                 + 1.11097 w_k1_IBM_s101
                                 + 0.808068 w_k1_INTC_s101
                                 + 0.83393 w_k1_KO_s101 + 0.92157 w_k1_MAT_s101
                                 + 0.783193 w_k1_MSFT_s101
                                 + 0.846041 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.824856 w_k1_ADM_s101
                                 + 1.02762 w_k1_AXP_s101
                                 + 0.86716 w_k1_DIS_s101
                                 + 0.843254 w_k1_FDX_s101
                                 + 1.23184 w_k1_IBM_s101
                                 + 0.865121 w_k1_INTC_s101
                                 + 1.00839 w_k1_KO_s101 + 1.56515 w_k1_MAT_s101
                                 + 0.939898 w_k1_MSFT_s101
                                 + 0.971706 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.53167 w_k1_ADM_s102
                                 + 1.60067 w_k1_AXP_s102
                                 + 1.55527 w_k1_DIS_s102
                                 + 1.34941 w_k1_FDX_s102
                                 + 1.23188 w_k1_IBM_s102
                                 + 1.31396 w_k1_INTC_s102
                                 + 1.42892 w_k1_KO_s102 + 2.0593 w_k1_MAT_s102
                                 + 1.29684 w_k1_MSFT_s102
                                 + 1.20101 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.32038 w_k1_ADM_s102
                                 + 1.12771 w_k1_AXP_s102
                                 + 1.10632 w_k1_DIS_s102
                                 + 1.11677 w_k1_FDX_s102
                                 + 0.835228 w_k1_IBM_s102
                                 + 1.13998 w_k1_INTC_s102
                                 + 1.10856 w_k1_KO_s102 + 1.0419 w_k1_MAT_s102
                                 + 1.04814 w_k1_MSFT_s102
                                 + 1.00665 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.985161 w_k1_ADM_s102
                                 + 0.726801 w_k1_AXP_s102
                                 + 0.732182 w_k1_DIS_s102
                                 + 0.824568 w_k1_FDX_s102
                                 + 0.734736 w_k1_IBM_s102
                                 + 0.999019 w_k1_INTC_s102
                                 + 0.875752 w_k1_KO_s102
                                 + 0.889456 w_k1_MAT_s102
                                 + 0.840852 w_k1_MSFT_s102
                                 + 0.901211 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.03878 w_k1_ADM_s102
                                 + 1.11383 w_k1_AXP_s102
                                 + 1.13448 w_k1_DIS_s102
                                 + 1.01639 w_k1_FDX_s102
                                 + 1.06254 w_k1_IBM_s102
                                 + 1.06915 w_k1_INTC_s102
                                 + 1.11503 w_k1_KO_s102 + 1.3568 w_k1_MAT_s102
                                 + 1.03288 w_k1_MSFT_s102
                                 + 1.12409 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.866745 w_k1_ADM_s102
                                 + 0.751327 w_k1_AXP_s102
                                 + 0.818938 w_k1_DIS_s102
                                 + 0.745969 w_k1_FDX_s102
                                 + 1.30388 w_k1_IBM_s102
                                 + 0.921647 w_k1_INTC_s102
                                 + 0.934856 w_k1_KO_s102
                                 + 0.990154 w_k1_MAT_s102
                                 + 0.953771 w_k1_MSFT_s102
                                 + 0.92321 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.38117 w_k1_ADM_s102
                                 + 1.22272 w_k1_AXP_s102
                                 + 1.68001 w_k1_DIS_s102
                                 + 1.32957 w_k1_FDX_s102
                                 + 0.790872 w_k1_IBM_s102
                                 + 1.11549 w_k1_INTC_s102
                                 + 1.15494 w_k1_KO_s102 + 1.10062 w_k1_MAT_s102
                                 + 1.38169 w_k1_MSFT_s102
                                 + 1.30944 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.09759 w_k1_ADM_s102
                                 + 0.957385 w_k1_AXP_s102
                                 + 1.60065 w_k1_DIS_s102
                                 + 1.03708 w_k1_FDX_s102
                                 + 1.57136 w_k1_IBM_s102
                                 + 1.0403 w_k1_INTC_s102 + 1.7333 w_k1_KO_s102
                                 + 1.82175 w_k1_MAT_s102
                                 + 1.26104 w_k1_MSFT_s102
                                 + 1.26417 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.938602 w_k1_ADM_s102
                                 + 0.901827 w_k1_AXP_s102
                                 + 1.00437 w_k1_DIS_s102
                                 + 0.87455 w_k1_FDX_s102
                                 + 1.24849 w_k1_IBM_s102
                                 + 1.03568 w_k1_INTC_s102
                                 + 1.10902 w_k1_KO_s102 + 1.45748 w_k1_MAT_s102
                                 + 1.01073 w_k1_MSFT_s102
                                 + 1.04203 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.984839 w_k1_ADM_s103
                                 + 1.23928 w_k1_AXP_s103
                                 + 1.00139 w_k1_DIS_s103
                                 + 0.997838 w_k1_FDX_s103
                                 + 1.57892 w_k1_IBM_s103
                                 + 1.14607 w_k1_INTC_s103
                                 + 1.35022 w_k1_KO_s103 + 1.79787 w_k1_MAT_s103
                                 + 0.97446 w_k1_MSFT_s103
                                 + 1.18901 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.7707 w_k1_ADM_s103
                                 + 0.658923 w_k1_AXP_s103
                                 + 0.932809 w_k1_DIS_s103
                                 + 0.849587 w_k1_FDX_s103
                                 + 1.20858 w_k1_IBM_s103
                                 + 0.965939 w_k1_INTC_s103
                                 + 1.09379 w_k1_KO_s103 + 1.32576 w_k1_MAT_s103
                                 + 1.01749 w_k1_MSFT_s103
                                 + 1.01023 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05069 w_k1_ADM_s103
                                 + 1.19565 w_k1_AXP_s103
                                 + 0.996837 w_k1_DIS_s103
                                 + 1.03489 w_k1_FDX_s103
                                 + 1.12935 w_k1_IBM_s103
                                 + 0.969863 w_k1_INTC_s103
                                 + 0.880954 w_k1_KO_s103
                                 + 1.10398 w_k1_MAT_s103
                                 + 0.953124 w_k1_MSFT_s103
                                 + 0.9604 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.831013 w_k1_ADM_s103
                                 + 0.742789 w_k1_AXP_s103
                                 + 0.549997 w_k1_DIS_s103
                                 + 0.728009 w_k1_FDX_s103
                                 + 0.74579 w_k1_IBM_s103
                                 + 0.839563 w_k1_INTC_s103
                                 + 0.644729 w_k1_KO_s103
                                 + 0.589579 w_k1_MAT_s103
                                 + 0.727867 w_k1_MSFT_s103
                                 + 0.781676 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.08679 w_k1_ADM_s103
                                 + 1.02841 w_k1_AXP_s103 + 0.951 w_k1_DIS_s103
                                 + 1.05822 w_k1_FDX_s103
                                 + 0.903808 w_k1_IBM_s103
                                 + 1.14467 w_k1_INTC_s103
                                 + 0.878507 w_k1_KO_s103
                                 + 1.02019 w_k1_MAT_s103
                                 + 1.01469 w_k1_MSFT_s103
                                 + 1.06412 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.832643 w_k1_ADM_s103
                                 + 0.805493 w_k1_AXP_s103
                                 + 0.742164 w_k1_DIS_s103
                                 + 0.825275 w_k1_FDX_s103
                                 + 0.944683 w_k1_IBM_s103
                                 + 0.751995 w_k1_INTC_s103
                                 + 0.762214 w_k1_KO_s103
                                 + 1.27868 w_k1_MAT_s103
                                 + 0.858571 w_k1_MSFT_s103
                                 + 0.947961 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.21323 w_k1_ADM_s103
                                 + 1.49697 w_k1_AXP_s103
                                 + 1.58176 w_k1_DIS_s103
                                 + 1.40384 w_k1_FDX_s103
                                 + 1.26187 w_k1_IBM_s103
                                 + 1.39315 w_k1_INTC_s103
                                 + 1.24937 w_k1_KO_s103 + 1.786 w_k1_MAT_s103
                                 + 1.22344 w_k1_MSFT_s103
                                 + 1.18713 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.04311 w_k1_ADM_s103
                                 + 1.24925 w_k1_AXP_s103
                                 + 1.03059 w_k1_DIS_s103
                                 + 1.15993 w_k1_FDX_s103
                                 + 0.666216 w_k1_IBM_s103
                                 + 0.975148 w_k1_INTC_s103
                                 + 0.893109 w_k1_KO_s103
                                 + 1.07524 w_k1_MAT_s103
                                 + 1.13626 w_k1_MSFT_s103
                                 + 1.01617 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.36026 w_k1_ADM_s104
                                 + 1.16075 w_k1_AXP_s104
                                 + 1.24064 w_k1_DIS_s104 + 1.2281 w_k1_FDX_s104
                                 + 1.15378 w_k1_IBM_s104
                                 + 1.08569 w_k1_INTC_s104
                                 + 1.27123 w_k1_KO_s104 + 1.1367 w_k1_MAT_s104
                                 + 1.22389 w_k1_MSFT_s104
                                 + 1.21015 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.882218 w_k1_ADM_s104
                                 + 0.695778 w_k1_AXP_s104
                                 + 0.528952 w_k1_DIS_s104
                                 + 0.718505 w_k1_FDX_s104
                                 + 0.717747 w_k1_IBM_s104
                                 + 0.954313 w_k1_INTC_s104
                                 + 0.74566 w_k1_KO_s104
                                 + 0.786943 w_k1_MAT_s104
                                 + 0.759826 w_k1_MSFT_s104
                                 + 0.83684 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.17196 w_k1_ADM_s104
                                 + 1.35824 w_k1_AXP_s104
                                 + 1.70328 w_k1_DIS_s104
                                 + 1.40144 w_k1_FDX_s104
                                 + 1.21373 w_k1_IBM_s104
                                 + 1.68277 w_k1_INTC_s104
                                 + 1.44713 w_k1_KO_s104 + 1.4337 w_k1_MAT_s104
                                 + 1.48225 w_k1_MSFT_s104 + 1.2845 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.957065 w_k1_ADM_s104
                                 + 0.8522 w_k1_AXP_s104
                                 + 0.799351 w_k1_DIS_s104
                                 + 0.785745 w_k1_FDX_s104 + 1.091 w_k1_IBM_s104
                                 + 0.754747 w_k1_INTC_s104
                                 + 0.988463 w_k1_KO_s104
                                 + 1.18567 w_k1_MAT_s104
                                 + 0.915331 w_k1_MSFT_s104
                                 + 0.990311 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.2571 w_k1_ADM_s104
                                 + 1.08199 w_k1_AXP_s104
                                 + 0.754781 w_k1_DIS_s104
                                 + 0.90955 w_k1_FDX_s104
                                 + 1.15191 w_k1_IBM_s104
                                 + 1.07297 w_k1_INTC_s104
                                 + 0.906053 w_k1_KO_s104 + 1.2042 w_k1_MAT_s104
                                 + 0.8899 w_k1_MSFT_s104
                                 + 0.948392 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.26052 w_k1_ADM_s104
                                 + 1.17355 w_k1_AXP_s104
                                 + 1.17701 w_k1_DIS_s104
                                 + 1.13902 w_k1_FDX_s104
                                 + 0.887667 w_k1_IBM_s104
                                 + 1.21664 w_k1_INTC_s104
                                 + 1.07548 w_k1_KO_s104 + 1.16145 w_k1_MAT_s104
                                 + 1.08666 w_k1_MSFT_s104
                                 + 1.05597 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.35412 w_k1_ADM_s104
                                 + 1.28129 w_k1_AXP_s104 + 1.1195 w_k1_DIS_s104
                                 + 1.03565 w_k1_FDX_s104
                                 + 1.55568 w_k1_IBM_s104
                                 + 1.14365 w_k1_INTC_s104
                                 + 1.37981 w_k1_KO_s104 + 1.78762 w_k1_MAT_s104
                                 + 1.13728 w_k1_MSFT_s104
                                 + 1.13557 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04224 w_k1_ADM_s104
                                 + 1.00768 w_k1_AXP_s104
                                 + 0.805314 w_k1_DIS_s104
                                 + 0.978795 w_k1_FDX_s104
                                 + 0.766183 w_k1_IBM_s104
                                 + 0.883685 w_k1_INTC_s104
                                 + 0.803524 w_k1_KO_s104
                                 + 0.917643 w_k1_MAT_s104
                                 + 0.90565 w_k1_MSFT_s104
                                 + 1.00513 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.974417 w_k1_ADM_s105
                                 + 1.11461 w_k1_AXP_s105
                                 + 1.69885 w_k1_DIS_s105
                                 + 1.41076 w_k1_FDX_s105
                                 + 1.09405 w_k1_IBM_s105
                                 + 1.49596 w_k1_INTC_s105
                                 + 1.26067 w_k1_KO_s105 + 1.31246 w_k1_MAT_s105
                                 + 1.47414 w_k1_MSFT_s105
                                 + 1.31106 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.2299 w_k1_ADM_s105
                                 + 1.07089 w_k1_AXP_s105 + 1.2081 w_k1_DIS_s105
                                 + 1.22963 w_k1_FDX_s105
                                 + 0.853008 w_k1_IBM_s105
                                 + 1.25916 w_k1_INTC_s105
                                 + 0.993269 w_k1_KO_s105
                                 + 0.670303 w_k1_MAT_s105
                                 + 1.32581 w_k1_MSFT_s105
                                 + 1.17234 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.977441 w_k1_ADM_s105
                                 + 0.789376 w_k1_AXP_s105
                                 + 0.651073 w_k1_DIS_s105
                                 + 0.836838 w_k1_FDX_s105
                                 + 0.771004 w_k1_IBM_s105
                                 + 0.946329 w_k1_INTC_s105
                                 + 0.736522 w_k1_KO_s105
                                 + 0.651981 w_k1_MAT_s105
                                 + 0.778344 w_k1_MSFT_s105
                                 + 0.801113 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.0817 w_k1_ADM_s105
                                 + 0.978403 w_k1_AXP_s105
                                 + 1.01561 w_k1_DIS_s105
                                 + 1.02731 w_k1_FDX_s105
                                 + 0.738682 w_k1_IBM_s105
                                 + 1.09291 w_k1_INTC_s105
                                 + 0.817626 w_k1_KO_s105
                                 + 0.73392 w_k1_MAT_s105
                                 + 0.993543 w_k1_MSFT_s105
                                 + 0.955857 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.00975 w_k1_ADM_s105
                                 + 0.742057 w_k1_AXP_s105
                                 + 1.04338 w_k1_DIS_s105
                                 + 0.893864 w_k1_FDX_s105
                                 + 1.27642 w_k1_IBM_s105
                                 + 0.964682 w_k1_INTC_s105
                                 + 1.02925 w_k1_KO_s105
                                 + 0.891035 w_k1_MAT_s105
                                 + 0.949916 w_k1_MSFT_s105
                                 + 0.965492 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.03392 w_k1_ADM_s105
                                 + 1.29586 w_k1_AXP_s105
                                 + 1.01571 w_k1_DIS_s105
                                 + 0.986575 w_k1_FDX_s105
                                 + 1.36835 w_k1_IBM_s105
                                 + 0.996596 w_k1_INTC_s105
                                 + 1.00619 w_k1_KO_s105 + 1.42072 w_k1_MAT_s105
                                 + 1.01993 w_k1_MSFT_s105
                                 + 1.11797 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.763215 w_k1_ADM_s105
                                 + 0.462071 w_k1_AXP_s105
                                 + 0.816739 w_k1_DIS_s105
                                 + 0.811173 w_k1_FDX_s105
                                 + 0.661836 w_k1_IBM_s105
                                 + 0.782823 w_k1_INTC_s105
                                 + 0.782392 w_k1_KO_s105
                                 + 0.368437 w_k1_MAT_s105
                                 + 0.894129 w_k1_MSFT_s105
                                 + 0.879522 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.2252 w_k1_ADM_s105
                                 + 1.82783 w_k1_AXP_s105
                                 + 1.38033 w_k1_DIS_s105
                                 + 1.33706 w_k1_FDX_s105
                                 + 1.30657 w_k1_IBM_s105
                                 + 1.51049 w_k1_INTC_s105
                                 + 1.13131 w_k1_KO_s105 + 1.20928 w_k1_MAT_s105
                                 + 1.29839 w_k1_MSFT_s105
                                 + 1.24782 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.98932 w_k1_ADM_s106
                                 + 0.952175 w_k1_AXP_s106
                                 + 0.853121 w_k1_DIS_s106
                                 + 0.928473 w_k1_FDX_s106
                                 + 0.740164 w_k1_IBM_s106
                                 + 1.01388 w_k1_INTC_s106
                                 + 0.837335 w_k1_KO_s106
                                 + 0.765135 w_k1_MAT_s106
                                 + 0.916248 w_k1_MSFT_s106
                                 + 0.967256 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.746658 w_k1_ADM_s106
                                 + 0.690779 w_k1_AXP_s106
                                 + 0.571017 w_k1_DIS_s106
                                 + 0.738203 w_k1_FDX_s106
                                 + 0.55222 w_k1_IBM_s106
                                 + 0.911985 w_k1_INTC_s106
                                 + 0.627005 w_k1_KO_s106
                                 + 0.820963 w_k1_MAT_s106
                                 + 0.833305 w_k1_MSFT_s106
                                 + 0.862995 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.17908 w_k1_ADM_s106
                                 + 0.952463 w_k1_AXP_s106
                                 + 1.20859 w_k1_DIS_s106
                                 + 1.20202 w_k1_FDX_s106
                                 + 0.768335 w_k1_IBM_s106
                                 + 1.28595 w_k1_INTC_s106
                                 + 1.02243 w_k1_KO_s106
                                 + 0.870875 w_k1_MAT_s106
                                 + 1.3993 w_k1_MSFT_s106 + 1.17422 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.996929 w_k1_ADM_s106
                                 + 1.02797 w_k1_AXP_s106
                                 + 1.25231 w_k1_DIS_s106
                                 + 0.996374 w_k1_FDX_s106
                                 + 1.34733 w_k1_IBM_s106
                                 + 1.15283 w_k1_INTC_s106
                                 + 1.34494 w_k1_KO_s106 + 1.56113 w_k1_MAT_s106
                                 + 1.11303 w_k1_MSFT_s106
                                 + 1.24375 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.78291 w_k1_ADM_s106
                                 + 0.863799 w_k1_AXP_s106
                                 + 1.01265 w_k1_DIS_s106
                                 + 0.968579 w_k1_FDX_s106
                                 + 0.829401 w_k1_IBM_s106
                                 + 0.926009 w_k1_INTC_s106
                                 + 0.907654 w_k1_KO_s106
                                 + 1.28943 w_k1_MAT_s106
                                 + 1.0036 w_k1_MSFT_s106 + 1.03019 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.864978 w_k1_ADM_s106
                                 + 0.719581 w_k1_AXP_s106
                                 + 0.574661 w_k1_DIS_s106
                                 + 0.692146 w_k1_FDX_s106
                                 + 1.02695 w_k1_IBM_s106
                                 + 1.04987 w_k1_INTC_s106
                                 + 0.823883 w_k1_KO_s106
                                 + 0.636673 w_k1_MAT_s106
                                 + 0.840377 w_k1_MSFT_s106
                                 + 0.93039 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.18161 w_k1_ADM_s106
                                 + 1.63176 w_k1_AXP_s106
                                 + 1.53202 w_k1_DIS_s106
                                 + 1.27537 w_k1_FDX_s106
                                 + 1.21663 w_k1_IBM_s106
                                 + 1.31343 w_k1_INTC_s106
                                 + 1.34959 w_k1_KO_s106 + 1.58595 w_k1_MAT_s106
                                 + 1.24213 w_k1_MSFT_s106
                                 + 1.37351 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.848189 w_k1_ADM_s106
                                 + 0.9967 w_k1_AXP_s106
                                 + 0.734994 w_k1_DIS_s106
                                 + 0.839085 w_k1_FDX_s106
                                 + 1.25519 w_k1_IBM_s106
                                 + 1.0427 w_k1_INTC_s106 + 1.01079 w_k1_KO_s106
                                 + 1.33775 w_k1_MAT_s106
                                 + 0.916304 w_k1_MSFT_s106
                                 + 1.08774 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.893882 w_k1_ADM_s107
                                 + 0.862705 w_k1_AXP_s107
                                 + 1.1007 w_k1_DIS_s107
                                 + 0.889011 w_k1_FDX_s107
                                 + 1.18702 w_k1_IBM_s107
                                 + 1.05702 w_k1_INTC_s107
                                 + 1.09734 w_k1_KO_s107
                                 + 0.755574 w_k1_MAT_s107
                                 + 0.989138 w_k1_MSFT_s107
                                 + 0.838099 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.14389 w_k1_ADM_s107
                                 + 1.20688 w_k1_AXP_s107 + 1.1391 w_k1_DIS_s107
                                 + 1.10902 w_k1_FDX_s107
                                 + 0.888064 w_k1_IBM_s107
                                 + 1.14041 w_k1_INTC_s107 + 1.0108 w_k1_KO_s107
                                 + 0.779195 w_k1_MAT_s107
                                 + 0.914092 w_k1_MSFT_s107
                                 + 0.921268 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.790281 w_k1_ADM_s107
                                 + 0.765755 w_k1_AXP_s107
                                 + 0.706878 w_k1_DIS_s107
                                 + 0.72577 w_k1_FDX_s107
                                 + 1.01772 w_k1_IBM_s107
                                 + 1.02805 w_k1_INTC_s107
                                 + 0.871224 w_k1_KO_s107
                                 + 0.569967 w_k1_MAT_s107
                                 + 0.691723 w_k1_MSFT_s107
                                 + 0.828509 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.06011 w_k1_ADM_s107
                                 + 1.06657 w_k1_AXP_s107
                                 + 1.13447 w_k1_DIS_s107
                                 + 0.99103 w_k1_FDX_s107
                                 + 1.60654 w_k1_IBM_s107
                                 + 0.927682 w_k1_INTC_s107
                                 + 1.14619 w_k1_KO_s107 + 1.11784 w_k1_MAT_s107
                                 + 0.969601 w_k1_MSFT_s107
                                 + 0.961777 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18683 w_k1_ADM_s107
                                 + 1.37141 w_k1_AXP_s107
                                 + 1.46658 w_k1_DIS_s107
                                 + 1.34229 w_k1_FDX_s107
                                 + 0.973233 w_k1_IBM_s107
                                 + 1.38491 w_k1_INTC_s107
                                 + 1.17007 w_k1_KO_s107 + 1.07973 w_k1_MAT_s107
                                 + 1.29271 w_k1_MSFT_s107
                                 + 1.13787 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.49231 w_k1_ADM_s107
                                 + 1.84747 w_k1_AXP_s107
                                 + 1.66844 w_k1_DIS_s107
                                 + 1.32047 w_k1_FDX_s107
                                 + 1.45172 w_k1_IBM_s107
                                 + 1.34642 w_k1_INTC_s107
                                 + 1.28974 w_k1_KO_s107 + 1.46505 w_k1_MAT_s107
                                 + 1.20249 w_k1_MSFT_s107
                                 + 1.12803 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.949628 w_k1_ADM_s107
                                 + 1.00464 w_k1_AXP_s107
                                 + 0.827447 w_k1_DIS_s107
                                 + 0.973674 w_k1_FDX_s107
                                 + 0.853496 w_k1_IBM_s107
                                 + 1.0048 w_k1_INTC_s107
                                 + 0.819614 w_k1_KO_s107
                                 + 0.94467 w_k1_MAT_s107
                                 + 0.95764 w_k1_MSFT_s107
                                 + 0.875177 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.755887 w_k1_ADM_s107
                                 + 0.695931 w_k1_AXP_s107
                                 + 0.711726 w_k1_DIS_s107
                                 + 0.752739 w_k1_FDX_s107
                                 + 0.677516 w_k1_IBM_s107
                                 + 0.869032 w_k1_INTC_s107
                                 + 0.707048 w_k1_KO_s107
                                 + 0.432472 w_k1_MAT_s107
                                 + 0.801016 w_k1_MSFT_s107
                                 + 0.741385 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.52333 w_k1_ADM_s110
                                 + 1.63417 w_k1_AXP_s110
                                 + 1.78318 w_k1_DIS_s110
                                 + 1.53838 w_k1_FDX_s110
                                 + 1.12241 w_k1_IBM_s110
                                 + 1.34621 w_k1_INTC_s110
                                 + 1.29928 w_k1_KO_s110 + 1.4934 w_k1_MAT_s110
                                 + 1.29618 w_k1_MSFT_s110
                                 + 1.20196 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.945678 w_k1_ADM_s110
                                 + 1.37128 w_k1_AXP_s110
                                 + 2.02056 w_k1_DIS_s110
                                 + 1.54634 w_k1_FDX_s110
                                 + 1.48478 w_k1_IBM_s110
                                 + 1.45842 w_k1_INTC_s110
                                 + 1.68342 w_k1_KO_s110 + 2.07814 w_k1_MAT_s110
                                 + 1.65128 w_k1_MSFT_s110
                                 + 1.38713 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.917657 w_k1_ADM_s110
                                 + 0.826974 w_k1_AXP_s110
                                 + 1.04819 w_k1_DIS_s110
                                 + 1.04295 w_k1_FDX_s110
                                 + 0.894113 w_k1_IBM_s110
                                 + 0.912625 w_k1_INTC_s110
                                 + 1.04885 w_k1_KO_s110
                                 + 0.937551 w_k1_MAT_s110
                                 + 0.900987 w_k1_MSFT_s110
                                 + 1.09995 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.796891 w_k1_ADM_s110
                                 + 0.679797 w_k1_AXP_s110
                                 + 0.667514 w_k1_DIS_s110
                                 + 0.725542 w_k1_FDX_s110
                                 + 0.740658 w_k1_IBM_s110
                                 + 0.79324 w_k1_INTC_s110
                                 + 0.876834 w_k1_KO_s110
                                 + 0.841887 w_k1_MAT_s110
                                 + 0.718759 w_k1_MSFT_s110
                                 + 0.876865 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.25899 w_k1_ADM_s110
                                 + 1.39667 w_k1_AXP_s110
                                 + 1.12255 w_k1_DIS_s110
                                 + 1.18834 w_k1_FDX_s110
                                 + 1.12154 w_k1_IBM_s110
                                 + 1.09793 w_k1_INTC_s110
                                 + 0.955324 w_k1_KO_s110
                                 + 1.14769 w_k1_MAT_s110
                                 + 0.962801 w_k1_MSFT_s110
                                 + 1.11631 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.02719 w_k1_ADM_s110
                                 + 1.05641 w_k1_AXP_s110
                                 + 1.19079 w_k1_DIS_s110
                                 + 1.06423 w_k1_FDX_s110
                                 + 1.26299 w_k1_IBM_s110
                                 + 1.11413 w_k1_INTC_s110
                                 + 1.14781 w_k1_KO_s110 + 1.14384 w_k1_MAT_s110
                                 + 0.984932 w_k1_MSFT_s110
                                 + 1.07082 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.86442 w_k1_ADM_s110
                                 + 0.914179 w_k1_AXP_s110
                                 + 0.902283 w_k1_DIS_s110
                                 + 0.8711 w_k1_FDX_s110 + 1.40564 w_k1_IBM_s110
                                 + 0.875984 w_k1_INTC_s110
                                 + 0.913993 w_k1_KO_s110
                                 + 1.42639 w_k1_MAT_s110
                                 + 0.736464 w_k1_MSFT_s110
                                 + 0.901003 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.18705 w_k1_ADM_s110
                                 + 1.12204 w_k1_AXP_s110
                                 + 1.51113 w_k1_DIS_s110
                                 + 1.20864 w_k1_FDX_s110
                                 + 0.787913 w_k1_IBM_s110
                                 + 1.21165 w_k1_INTC_s110 + 1.106 w_k1_KO_s110
                                 + 0.808866 w_k1_MAT_s110
                                 + 1.02036 w_k1_MSFT_s110
                                 + 1.03614 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.722835 w_k1_ADM_s111
                                 + 0.58449 w_k1_AXP_s111
                                 + 0.669132 w_k1_DIS_s111
                                 + 0.779917 w_k1_FDX_s111
                                 + 0.708691 w_k1_IBM_s111
                                 + 0.873359 w_k1_INTC_s111
                                 + 0.693716 w_k1_KO_s111
                                 + 0.704641 w_k1_MAT_s111
                                 + 0.966122 w_k1_MSFT_s111
                                 + 0.820684 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.892533 w_k1_ADM_s111
                                 + 0.999761 w_k1_AXP_s111
                                 + 0.752061 w_k1_DIS_s111
                                 + 0.868754 w_k1_FDX_s111
                                 + 0.887506 w_k1_IBM_s111
                                 + 0.925737 w_k1_INTC_s111
                                 + 0.873412 w_k1_KO_s111
                                 + 0.906849 w_k1_MAT_s111
                                 + 0.841495 w_k1_MSFT_s111
                                 + 0.996088 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.755347 w_k1_ADM_s111
                                 + 0.747089 w_k1_AXP_s111
                                 + 0.801225 w_k1_DIS_s111
                                 + 0.735423 w_k1_FDX_s111
                                 + 1.14469 w_k1_IBM_s111
                                 + 1.17851 w_k1_INTC_s111 + 1.0761 w_k1_KO_s111
                                 + 1.1017 w_k1_MAT_s111
                                 + 0.969367 w_k1_MSFT_s111
                                 + 0.94385 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.31501 w_k1_ADM_s111
                                 + 1.15684 w_k1_AXP_s111
                                 + 1.76577 w_k1_DIS_s111
                                 + 1.41143 w_k1_FDX_s111
                                 + 1.04757 w_k1_IBM_s111
                                 + 1.15292 w_k1_INTC_s111 + 1.4498 w_k1_KO_s111
                                 + 1.18766 w_k1_MAT_s111
                                 + 1.41257 w_k1_MSFT_s111
                                 + 1.23553 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.990257 w_k1_ADM_s111
                                 + 0.845614 w_k1_AXP_s111
                                 + 0.993641 w_k1_DIS_s111
                                 + 0.909851 w_k1_FDX_s111
                                 + 0.925606 w_k1_IBM_s111
                                 + 1.02532 w_k1_INTC_s111
                                 + 0.999715 w_k1_KO_s111
                                 + 0.941417 w_k1_MAT_s111
                                 + 1.06812 w_k1_MSFT_s111
                                 + 1.05381 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.872795 w_k1_ADM_s111
                                 + 1.12512 w_k1_AXP_s111
                                 + 1.25751 w_k1_DIS_s111
                                 + 1.09817 w_k1_FDX_s111
                                 + 0.957959 w_k1_IBM_s111
                                 + 1.06886 w_k1_INTC_s111
                                 + 1.11895 w_k1_KO_s111 + 1.1187 w_k1_MAT_s111
                                 + 1.22148 w_k1_MSFT_s111
                                 + 1.09654 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.983356 w_k1_ADM_s111
                                 + 1.01743 w_k1_AXP_s111
                                 + 1.22609 w_k1_DIS_s111
                                 + 1.05891 w_k1_FDX_s111
                                 + 1.42888 w_k1_IBM_s111
                                 + 0.999656 w_k1_INTC_s111
                                 + 1.24551 w_k1_KO_s111 + 1.11504 w_k1_MAT_s111
                                 + 1.07643 w_k1_MSFT_s111
                                 + 1.19849 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.887833 w_k1_ADM_s111
                                 + 1.13135 w_k1_AXP_s111
                                 + 0.964655 w_k1_DIS_s111
                                 + 1.01883 w_k1_FDX_s111
                                 + 1.23482 w_k1_IBM_s111
                                 + 1.13114 w_k1_INTC_s111
                                 + 1.08565 w_k1_KO_s111 + 1.38037 w_k1_MAT_s111
                                 + 0.940404 w_k1_MSFT_s111
                                 + 0.997302 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.935691 w_k1_ADM_s112
                                 + 0.846839 w_k1_AXP_s112
                                 + 0.876861 w_k1_DIS_s112
                                 + 0.918522 w_k1_FDX_s112
                                 + 1.06063 w_k1_IBM_s112
                                 + 0.816095 w_k1_INTC_s112
                                 + 1.01601 w_k1_KO_s112 + 1.26605 w_k1_MAT_s112
                                 + 0.954132 w_k1_MSFT_s112
                                 + 1.00353 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.14156 w_k1_ADM_s112
                                 + 1.38888 w_k1_AXP_s112
                                 + 1.63863 w_k1_DIS_s112
                                 + 1.30511 w_k1_FDX_s112
                                 + 1.27304 w_k1_IBM_s112
                                 + 1.37736 w_k1_INTC_s112
                                 + 1.43933 w_k1_KO_s112 + 1.53875 w_k1_MAT_s112
                                 + 1.40553 w_k1_MSFT_s112
                                 + 1.22576 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.059 w_k1_ADM_s112
                                 + 1.11275 w_k1_AXP_s112
                                 + 1.25223 w_k1_DIS_s112
                                 + 1.12479 w_k1_FDX_s112
                                 + 0.996095 w_k1_IBM_s112
                                 + 1.26205 w_k1_INTC_s112
                                 + 1.16171 w_k1_KO_s112 + 1.17646 w_k1_MAT_s112
                                 + 1.15697 w_k1_MSFT_s112
                                 + 1.10357 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.1068 w_k1_ADM_s112
                                 + 1.01831 w_k1_AXP_s112
                                 + 0.998253 w_k1_DIS_s112
                                 + 0.837648 w_k1_FDX_s112
                                 + 1.31043 w_k1_IBM_s112
                                 + 0.830836 w_k1_INTC_s112
                                 + 1.24924 w_k1_KO_s112 + 1.61763 w_k1_MAT_s112
                                 + 1.15288 w_k1_MSFT_s112
                                 + 1.04362 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.86736 w_k1_ADM_s112
                                 + 0.954188 w_k1_AXP_s112
                                 + 0.912006 w_k1_DIS_s112
                                 + 0.916854 w_k1_FDX_s112
                                 + 0.767307 w_k1_IBM_s112
                                 + 0.952999 w_k1_INTC_s112 + 1.019 w_k1_KO_s112
                                 + 0.871841 w_k1_MAT_s112
                                 + 1.05642 w_k1_MSFT_s112
                                 + 0.970627 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.806209 w_k1_ADM_s112
                                 + 0.711166 w_k1_AXP_s112
                                 + 0.571606 w_k1_DIS_s112
                                 + 0.702358 w_k1_FDX_s112
                                 + 0.920575 w_k1_IBM_s112
                                 + 0.864145 w_k1_INTC_s112
                                 + 0.926763 w_k1_KO_s112
                                 + 0.836913 w_k1_MAT_s112
                                 + 0.7851 w_k1_MSFT_s112
                                 + 0.841537 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.93464 w_k1_ADM_s112
                                 + 1.12756 w_k1_AXP_s112
                                 + 0.912664 w_k1_DIS_s112
                                 + 1.0485 w_k1_FDX_s112
                                 + 0.893884 w_k1_IBM_s112
                                 + 1.04242 w_k1_INTC_s112 + 1.0664 w_k1_KO_s112
                                 + 0.700917 w_k1_MAT_s112
                                 + 1.07071 w_k1_MSFT_s112
                                 + 1.00463 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.841683 w_k1_ADM_s112
                                 + 0.55808 w_k1_AXP_s112
                                 + 0.579256 w_k1_DIS_s112
                                 + 0.739468 w_k1_FDX_s112
                                 + 0.667319 w_k1_IBM_s112
                                 + 0.631175 w_k1_INTC_s112
                                 + 0.773203 w_k1_KO_s112
                                 + 0.730343 w_k1_MAT_s112
                                 + 0.823656 w_k1_MSFT_s112
                                 + 0.897702 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.88257 w_k1_ADM_s113
                                 + 1.29863 w_k1_AXP_s113
                                 + 1.32892 w_k1_DIS_s113
                                 + 1.05466 w_k1_FDX_s113
                                 + 1.47686 w_k1_IBM_s113
                                 + 0.99442 w_k1_INTC_s113
                                 + 1.19302 w_k1_KO_s113 + 1.33719 w_k1_MAT_s113
                                 + 1.11831 w_k1_MSFT_s113
                                 + 1.04032 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.05936 w_k1_ADM_s113
                                 + 1.02041 w_k1_AXP_s113
                                 + 0.864606 w_k1_DIS_s113
                                 + 0.897172 w_k1_FDX_s113
                                 + 0.947956 w_k1_IBM_s113
                                 + 0.694697 w_k1_INTC_s113
                                 + 0.925648 w_k1_KO_s113
                                 + 0.926889 w_k1_MAT_s113
                                 + 0.91552 w_k1_MSFT_s113
                                 + 1.01999 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.1185 w_k1_ADM_s113
                                 + 1.49023 w_k1_AXP_s113 + 1.4714 w_k1_DIS_s113
                                 + 1.3219 w_k1_FDX_s113 + 1.07175 w_k1_IBM_s113
                                 + 1.15798 w_k1_INTC_s113
                                 + 1.06927 w_k1_KO_s113 + 1.40696 w_k1_MAT_s113
                                 + 1.26184 w_k1_MSFT_s113
                                 + 1.28049 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.806891 w_k1_ADM_s113
                                 + 0.617187 w_k1_AXP_s113
                                 + 0.501335 w_k1_DIS_s113
                                 + 0.757397 w_k1_FDX_s113
                                 + 0.782079 w_k1_IBM_s113
                                 + 0.739945 w_k1_INTC_s113
                                 + 0.603261 w_k1_KO_s113
                                 + 0.59998 w_k1_MAT_s113
                                 + 0.838787 w_k1_MSFT_s113
                                 + 0.820211 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.11357 w_k1_ADM_s113
                                 + 0.911259 w_k1_AXP_s113
                                 + 0.596404 w_k1_DIS_s113
                                 + 0.782581 w_k1_FDX_s113
                                 + 0.978569 w_k1_IBM_s113
                                 + 0.771539 w_k1_INTC_s113
                                 + 0.802329 w_k1_KO_s113
                                 + 1.21875 w_k1_MAT_s113
                                 + 0.856986 w_k1_MSFT_s113
                                 + 0.952429 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.29521 w_k1_ADM_s113
                                 + 0.695613 w_k1_AXP_s113
                                 + 0.747002 w_k1_DIS_s113
                                 + 0.859101 w_k1_FDX_s113
                                 + 0.789747 w_k1_IBM_s113
                                 + 0.94062 w_k1_INTC_s113
                                 + 0.866637 w_k1_KO_s113
                                 + 0.636251 w_k1_MAT_s113
                                 + 0.966885 w_k1_MSFT_s113
                                 + 0.887003 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.28303 w_k1_ADM_s113
                                 + 0.890592 w_k1_AXP_s113
                                 + 1.11276 w_k1_DIS_s113
                                 + 1.00127 w_k1_FDX_s113
                                 + 0.900893 w_k1_IBM_s113
                                 + 1.05742 w_k1_INTC_s113
                                 + 0.961882 w_k1_KO_s113
                                 + 0.739031 w_k1_MAT_s113
                                 + 1.1619 w_k1_MSFT_s113 + 1.08185 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.38956 w_k1_ADM_s113
                                 + 1.10474 w_k1_AXP_s113
                                 + 0.849562 w_k1_DIS_s113
                                 + 1.04727 w_k1_FDX_s113
                                 + 0.933091 w_k1_IBM_s113
                                 + 0.932129 w_k1_INTC_s113
                                 + 0.80623 w_k1_KO_s113 + 1.10901 w_k1_MAT_s113
                                 + 0.881671 w_k1_MSFT_s113
                                 + 0.900337 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.02656 w_k1_ADM_s114
                                 + 0.989195 w_k1_AXP_s114
                                 + 1.29668 w_k1_DIS_s114
                                 + 1.02419 w_k1_FDX_s114
                                 + 1.40023 w_k1_IBM_s114
                                 + 1.07869 w_k1_INTC_s114
                                 + 1.26115 w_k1_KO_s114 + 1.09225 w_k1_MAT_s114
                                 + 1.0476 w_k1_MSFT_s114 + 1.12838 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.980883 w_k1_ADM_s114
                                 + 1.17977 w_k1_AXP_s114
                                 + 1.03919 w_k1_DIS_s114 + 1.1071 w_k1_FDX_s114
                                 + 0.81964 w_k1_IBM_s114
                                 + 1.00833 w_k1_INTC_s114
                                 + 1.11297 w_k1_KO_s114
                                 + 0.996673 w_k1_MAT_s114
                                 + 1.05702 w_k1_MSFT_s114
                                 + 1.11168 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.19892 w_k1_ADM_s114
                                 + 1.07457 w_k1_AXP_s114
                                 + 0.799143 w_k1_DIS_s114
                                 + 0.982958 w_k1_FDX_s114
                                 + 1.25954 w_k1_IBM_s114
                                 + 1.10235 w_k1_INTC_s114
                                 + 1.12079 w_k1_KO_s114 + 1.15259 w_k1_MAT_s114
                                 + 0.971427 w_k1_MSFT_s114
                                 + 1.11021 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.34938 w_k1_ADM_s114
                                 + 1.40757 w_k1_AXP_s114
                                 + 1.86901 w_k1_DIS_s114 + 1.5706 w_k1_FDX_s114
                                 + 0.946299 w_k1_IBM_s114
                                 + 1.3826 w_k1_INTC_s114 + 1.29367 w_k1_KO_s114
                                 + 0.915651 w_k1_MAT_s114
                                 + 1.47243 w_k1_MSFT_s114
                                 + 1.30558 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0661 w_k1_ADM_s114
                                 + 0.810225 w_k1_AXP_s114
                                 + 0.556423 w_k1_DIS_s114
                                 + 0.661636 w_k1_FDX_s114
                                 + 1.37559 w_k1_IBM_s114
                                 + 0.667943 w_k1_INTC_s114
                                 + 1.01329 w_k1_KO_s114 + 1.2402 w_k1_MAT_s114
                                 + 0.612847 w_k1_MSFT_s114
                                 + 0.935363 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.1122 w_k1_ADM_s114
                                 + 1.26092 w_k1_AXP_s114
                                 + 1.49725 w_k1_DIS_s114
                                 + 1.20864 w_k1_FDX_s114
                                 + 1.22625 w_k1_IBM_s114
                                 + 1.14729 w_k1_INTC_s114
                                 + 1.31559 w_k1_KO_s114 + 1.56742 w_k1_MAT_s114
                                 + 1.12199 w_k1_MSFT_s114
                                 + 1.24937 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.87819 w_k1_ADM_s114
                                 + 1.51087 w_k1_AXP_s114
                                 + 1.07364 w_k1_DIS_s114
                                 + 1.03341 w_k1_FDX_s114
                                 + 1.96454 w_k1_IBM_s114
                                 + 1.13128 w_k1_INTC_s114
                                 + 1.37943 w_k1_KO_s114 + 2.51068 w_k1_MAT_s114
                                 + 1.0523 w_k1_MSFT_s114 + 1.22209 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.803063 w_k1_ADM_s114
                                 + 0.762486 w_k1_AXP_s114
                                 + 0.850566 w_k1_DIS_s114
                                 + 0.901887 w_k1_FDX_s114
                                 + 0.78299 w_k1_IBM_s114
                                 + 1.05428 w_k1_INTC_s114
                                 + 0.987658 w_k1_KO_s114
                                 + 0.895522 w_k1_MAT_s114
                                 + 1.07351 w_k1_MSFT_s114
                                 + 0.905664 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.827627 w_k1_ADM_s115
                                 + 0.66986 w_k1_AXP_s115
                                 + 0.665389 w_k1_DIS_s115
                                 + 0.786468 w_k1_FDX_s115
                                 + 0.69457 w_k1_IBM_s115
                                 + 0.686211 w_k1_INTC_s115
                                 + 0.670133 w_k1_KO_s115
                                 + 0.714441 w_k1_MAT_s115
                                 + 0.783065 w_k1_MSFT_s115
                                 + 0.867335 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.1271 w_k1_ADM_s115
                                 + 1.23439 w_k1_AXP_s115
                                 + 1.07358 w_k1_DIS_s115
                                 + 1.11044 w_k1_FDX_s115
                                 + 0.755035 w_k1_IBM_s115
                                 + 1.15795 w_k1_INTC_s115
                                 + 0.804224 w_k1_KO_s115
                                 + 0.808008 w_k1_MAT_s115
                                 + 1.1194 w_k1_MSFT_s115
                                 + 0.931627 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.917574 w_k1_ADM_s115
                                 + 1.14724 w_k1_AXP_s115
                                 + 1.01985 w_k1_DIS_s115
                                 + 0.884187 w_k1_FDX_s115
                                 + 1.32425 w_k1_IBM_s115
                                 + 0.959974 w_k1_INTC_s115
                                 + 1.06924 w_k1_KO_s115 + 1.74017 w_k1_MAT_s115
                                 + 1.01538 w_k1_MSFT_s115 + 1.0448 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.958812 w_k1_ADM_s115
                                 + 0.768692 w_k1_AXP_s115
                                 + 0.755539 w_k1_DIS_s115
                                 + 0.765743 w_k1_FDX_s115
                                 + 0.948051 w_k1_IBM_s115
                                 + 0.855799 w_k1_INTC_s115
                                 + 0.840752 w_k1_KO_s115
                                 + 0.941051 w_k1_MAT_s115
                                 + 0.779535 w_k1_MSFT_s115
                                 + 0.83692 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.63083 w_k1_ADM_s115
                                 + 1.74276 w_k1_AXP_s115
                                 + 1.68107 w_k1_DIS_s115
                                 + 1.42462 w_k1_FDX_s115
                                 + 1.12973 w_k1_IBM_s115
                                 + 1.23081 w_k1_INTC_s115
                                 + 1.30706 w_k1_KO_s115 + 1.64103 w_k1_MAT_s115
                                 + 1.37003 w_k1_MSFT_s115
                                 + 1.03421 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0799 w_k1_ADM_s115
                                 + 0.964223 w_k1_AXP_s115
                                 + 0.788743 w_k1_DIS_s115
                                 + 0.892045 w_k1_FDX_s115
                                 + 0.698675 w_k1_IBM_s115
                                 + 0.77304 w_k1_INTC_s115
                                 + 0.717345 w_k1_KO_s115
                                 + 0.714538 w_k1_MAT_s115
                                 + 0.829054 w_k1_MSFT_s115
                                 + 0.78765 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.24625 w_k1_ADM_s115
                                 + 1.008 w_k1_AXP_s115 + 1.10992 w_k1_DIS_s115
                                 + 1.04371 w_k1_FDX_s115
                                 + 0.894837 w_k1_IBM_s115
                                 + 1.09137 w_k1_INTC_s115
                                 + 0.950586 w_k1_KO_s115
                                 + 0.854715 w_k1_MAT_s115
                                 + 1.01236 w_k1_MSFT_s115
                                 + 0.945773 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.16274 w_k1_ADM_s115
                                 + 1.0109 w_k1_AXP_s115 + 1.30623 w_k1_DIS_s115
                                 + 1.11919 w_k1_FDX_s115
                                 + 1.17335 w_k1_IBM_s115
                                 + 1.03515 w_k1_INTC_s115
                                 + 1.24095 w_k1_KO_s115 + 1.23889 w_k1_MAT_s115
                                 + 1.30282 w_k1_MSFT_s115
                                 + 1.06487 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.14485 w_k1_ADM_s116
                                 + 1.13142 w_k1_AXP_s116
                                 + 1.23557 w_k1_DIS_s116
                                 + 1.15615 w_k1_FDX_s116
                                 + 1.46014 w_k1_IBM_s116
                                 + 1.16761 w_k1_INTC_s116
                                 + 1.23792 w_k1_KO_s116 + 1.53209 w_k1_MAT_s116
                                 + 1.05328 w_k1_MSFT_s116
                                 + 1.06439 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.36414 w_k1_ADM_s116
                                 + 1.00407 w_k1_AXP_s116
                                 + 1.08101 w_k1_DIS_s116
                                 + 0.970002 w_k1_FDX_s116
                                 + 0.860501 w_k1_IBM_s116
                                 + 0.836116 w_k1_INTC_s116
                                 + 1.07674 w_k1_KO_s116
                                 + 0.961745 w_k1_MAT_s116
                                 + 0.952351 w_k1_MSFT_s116
                                 + 1.15562 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.24303 w_k1_ADM_s116
                                 + 0.775528 w_k1_AXP_s116
                                 + 0.848275 w_k1_DIS_s116
                                 + 0.955138 w_k1_FDX_s116
                                 + 0.603232 w_k1_IBM_s116
                                 + 0.774723 w_k1_INTC_s116
                                 + 0.761481 w_k1_KO_s116
                                 + 0.479015 w_k1_MAT_s116
                                 + 0.968051 w_k1_MSFT_s116
                                 + 0.903094 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.834637 w_k1_ADM_s116
                                 + 0.499677 w_k1_AXP_s116
                                 + 0.6394 w_k1_DIS_s116
                                 + 0.705638 w_k1_FDX_s116
                                 + 0.815919 w_k1_IBM_s116
                                 + 0.724498 w_k1_INTC_s116
                                 + 0.862985 w_k1_KO_s116
                                 + 0.584355 w_k1_MAT_s116
                                 + 0.738425 w_k1_MSFT_s116
                                 + 0.934954 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.14194 w_k1_ADM_s116
                                 + 0.881072 w_k1_AXP_s116
                                 + 0.76109 w_k1_DIS_s116
                                 + 0.78587 w_k1_FDX_s116
                                 + 1.11638 w_k1_IBM_s116
                                 + 0.813855 w_k1_INTC_s116
                                 + 1.00956 w_k1_KO_s116 + 1.1588 w_k1_MAT_s116
                                 + 0.772227 w_k1_MSFT_s116
                                 + 0.947048 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.16707 w_k1_ADM_s116
                                 + 1.19078 w_k1_AXP_s116
                                 + 1.47804 w_k1_DIS_s116
                                 + 1.19036 w_k1_FDX_s116
                                 + 1.02461 w_k1_IBM_s116
                                 + 1.00662 w_k1_INTC_s116
                                 + 1.13345 w_k1_KO_s116
                                 + 0.911501 w_k1_MAT_s116
                                 + 1.2262 w_k1_MSFT_s116 + 1.31124 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.881246 w_k1_ADM_s116
                                 + 0.765358 w_k1_AXP_s116
                                 + 0.64038 w_k1_DIS_s116
                                 + 0.773578 w_k1_FDX_s116
                                 + 0.823761 w_k1_IBM_s116
                                 + 0.637149 w_k1_INTC_s116
                                 + 0.781728 w_k1_KO_s116
                                 + 0.920574 w_k1_MAT_s116
                                 + 0.775892 w_k1_MSFT_s116
                                 + 0.933808 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.55766 w_k1_ADM_s116
                                 + 1.33795 w_k1_AXP_s116 + 1.6176 w_k1_DIS_s116
                                 + 1.49662 w_k1_FDX_s116
                                 + 0.719997 w_k1_IBM_s116
                                 + 1.11608 w_k1_INTC_s116
                                 + 1.09204 w_k1_KO_s116
                                 + 0.762175 w_k1_MAT_s116
                                 + 1.25664 w_k1_MSFT_s116
                                 + 1.18413 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10064 w_k1_ADM_s117
                                 + 0.919112 w_k1_AXP_s117
                                 + 1.14324 w_k1_DIS_s117
                                 + 1.06085 w_k1_FDX_s117
                                 + 0.821214 w_k1_IBM_s117
                                 + 0.857111 w_k1_INTC_s117
                                 + 0.913023 w_k1_KO_s117
                                 + 0.664005 w_k1_MAT_s117
                                 + 1.02377 w_k1_MSFT_s117
                                 + 0.966361 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.935192 w_k1_ADM_s117
                                 + 1.04115 w_k1_AXP_s117
                                 + 0.982598 w_k1_DIS_s117
                                 + 0.840196 w_k1_FDX_s117
                                 + 1.14166 w_k1_IBM_s117
                                 + 0.946038 w_k1_INTC_s117
                                 + 0.932077 w_k1_KO_s117
                                 + 1.22732 w_k1_MAT_s117
                                 + 0.893729 w_k1_MSFT_s117
                                 + 0.895249 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.08516 w_k1_ADM_s117
                                 + 1.28219 w_k1_AXP_s117
                                 + 1.36633 w_k1_DIS_s117
                                 + 1.11148 w_k1_FDX_s117
                                 + 1.13024 w_k1_IBM_s117
                                 + 0.924182 w_k1_INTC_s117
                                 + 1.18554 w_k1_KO_s117
                                 + 0.923972 w_k1_MAT_s117
                                 + 1.17574 w_k1_MSFT_s117
                                 + 1.14347 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.97631 w_k1_ADM_s117
                                 + 1.50228 w_k1_AXP_s117
                                 + 1.28125 w_k1_DIS_s117 + 1.2029 w_k1_FDX_s117
                                 + 0.82998 w_k1_IBM_s117
                                 + 1.07288 w_k1_INTC_s117
                                 + 0.978664 w_k1_KO_s117 + 1.0809 w_k1_MAT_s117
                                 + 1.10504 w_k1_MSFT_s117
                                 + 0.908951 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.10281 w_k1_ADM_s117
                                 + 1.32662 w_k1_AXP_s117
                                 + 1.51436 w_k1_DIS_s117
                                 + 1.15912 w_k1_FDX_s117
                                 + 1.44488 w_k1_IBM_s117
                                 + 1.04836 w_k1_INTC_s117
                                 + 1.60639 w_k1_KO_s117 + 1.24454 w_k1_MAT_s117
                                 + 1.18672 w_k1_MSFT_s117
                                 + 1.12687 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.807581 w_k1_ADM_s117
                                 + 0.784635 w_k1_AXP_s117
                                 + 0.817794 w_k1_DIS_s117
                                 + 0.735275 w_k1_FDX_s117
                                 + 0.813026 w_k1_IBM_s117
                                 + 0.667042 w_k1_INTC_s117
                                 + 0.727237 w_k1_KO_s117
                                 + 0.762621 w_k1_MAT_s117
                                 + 0.761257 w_k1_MSFT_s117
                                 + 0.787831 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.29304 w_k1_ADM_s117
                                 + 1.42421 w_k1_AXP_s117
                                 + 1.65867 w_k1_DIS_s117
                                 + 1.38133 w_k1_FDX_s117
                                 + 1.09709 w_k1_IBM_s117
                                 + 1.28381 w_k1_INTC_s117
                                 + 1.22607 w_k1_KO_s117 + 1.05184 w_k1_MAT_s117
                                 + 1.39086 w_k1_MSFT_s117
                                 + 1.02905 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.926226 w_k1_ADM_s117
                                 + 1.06079 w_k1_AXP_s117
                                 + 1.16989 w_k1_DIS_s117
                                 + 0.855617 w_k1_FDX_s117
                                 + 1.4578 w_k1_IBM_s117
                                 + 0.751397 w_k1_INTC_s117
                                 + 1.19632 w_k1_KO_s117 + 1.19974 w_k1_MAT_s117
                                 + 0.855537 w_k1_MSFT_s117
                                 + 0.994657 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0376263926293053 w_k1_ADM_s000
                                 + 0.0345769351430443 w_k1_ADM_s001
                                 + 0.0480843054603962 w_k1_ADM_s002
                                 + 0.0195982961297757 w_k1_ADM_s003
                                 + 0.0174707690608078 w_k1_ADM_s004
                                 + 0.0524711407675546 w_k1_ADM_s005
                                 + 0.016558847064012 w_k1_ADM_s006
                                 + 0.0126045912284177 w_k1_ADM_s007
                                 + 0.0285810895338875 w_k1_ADM_s010
                                 + 0.0260690701138431 w_k1_ADM_s011
                                 + 0.076835738221166 w_k1_ADM_s012
                                 + 0.0757680401536993 w_k1_ADM_s013
                                 + 0.0305415537107315 w_k1_ADM_s014
                                 + 0.0676021947938211 w_k1_ADM_s015
                                 + 0.028324410678665 w_k1_ADM_s016
                                 + 0.027307982253983 w_k1_ADM_s017
                                 + 0.0428348588147387 w_k1_ADM_s100
                                 + 0.027190473927969 w_k1_ADM_s101
                                 + 0.0487526986458478 w_k1_ADM_s102
                                 + 0.0347812273578961 w_k1_ADM_s103
                                 + 0.0649643588167895 w_k1_ADM_s104
                                 + 0.0229314289229296 w_k1_ADM_s105
                                 + 0.0271229288252143 w_k1_ADM_s106
                                 + 0.0384837764117178 w_k1_ADM_s107
                                 + 0.0183722528922069 w_k1_ADM_s110
                                 + 0.0105302241817918 w_k1_ADM_s111
                                 + 0.0119082210575079 w_k1_ADM_s112
                                 + 0.00831384603340084 w_k1_ADM_s113
                                 + 0.00864878928645919 w_k1_ADM_s114
                                 + 0.0402831689713575 w_k1_ADM_s115
                                 + 0.0294487325316707 w_k1_ADM_s116
                                 + 0.0152196249400336 w_k1_ADM_s117
                                 + 0.0413396882800938 w_k1_AXP_s000
                                 + 0.0372038981301441 w_k1_AXP_s001
                                 + 0.0512067870044213 w_k1_AXP_s002
                                 + 0.0226365653417706 w_k1_AXP_s003
                                 + 0.0175787634939488 w_k1_AXP_s004
                                 + 0.0500313912573612 w_k1_AXP_s005
                                 + 0.0156447692080233 w_k1_AXP_s006
                                 + 0.0124229232301921 w_k1_AXP_s007
                                 + 0.0277988392008805 w_k1_AXP_s010
                                 + 0.0253227747519928 w_k1_AXP_s011
                                 + 0.0764736914088931 w_k1_AXP_s012
                                 + 0.0832265891144498 w_k1_AXP_s013
                                 + 0.0311668750597746 w_k1_AXP_s014
                                 + 0.0537080334618859 w_k1_AXP_s015
                                 + 0.0278236950084322 w_k1_AXP_s016
                                 + 0.0257317267799164 w_k1_AXP_s017
                                 + 0.0446107021216286 w_k1_AXP_s100
                                 + 0.0228086679361753 w_k1_AXP_s101
                                 + 0.0441994786321258 w_k1_AXP_s102
                                 + 0.0373807956302525 w_k1_AXP_s103
                                 + 0.0602765790195616 w_k1_AXP_s104
                                 + 0.0219341667393273 w_k1_AXP_s105
                                 + 0.0283147650833079 w_k1_AXP_s106
                                 + 0.0407701255616923 w_k1_AXP_s107
                                 + 0.0185030090845694 w_k1_AXP_s110
                                 + 0.0106574065741007 w_k1_AXP_s111
                                 + 0.0124743075319248 w_k1_AXP_s112
                                 + 0.00811476540540712 w_k1_AXP_s113
                                 + 0.00893064095180219 w_k1_AXP_s114
                                 + 0.03829505527413 w_k1_AXP_s115
                                 + 0.0243299133117036 w_k1_AXP_s116
                                 + 0.0158742049855689 w_k1_AXP_s117
                                 + 0.0378798937928576 w_k1_DIS_s000
                                 + 0.0342615620126302 w_k1_DIS_s001
                                 + 0.052922594861747 w_k1_DIS_s002
                                 + 0.0225285054896751 w_k1_DIS_s003
                                 + 0.0168007065195897 w_k1_DIS_s004
                                 + 0.0553211891434872 w_k1_DIS_s005
                                 + 0.017352950421857 w_k1_DIS_s006
                                 + 0.012856142990458 w_k1_DIS_s007
                                 + 0.0281835092042292 w_k1_DIS_s010
                                 + 0.0238391587430493 w_k1_DIS_s011
                                 + 0.0742981461764676 w_k1_DIS_s012
                                 + 0.0845126583015791 w_k1_DIS_s013
                                 + 0.0313147612921695 w_k1_DIS_s014
                                 + 0.0579345418111046 w_k1_DIS_s015
                                 + 0.0270690555576471 w_k1_DIS_s016
                                 + 0.0246500260290324 w_k1_DIS_s017
                                 + 0.0461389750071472 w_k1_DIS_s100
                                 + 0.0242321753057253 w_k1_DIS_s101
                                 + 0.0473034276031 w_k1_DIS_s102
                                 + 0.0351329898496793 w_k1_DIS_s103
                                 + 0.0547714653379253 w_k1_DIS_s104
                                 + 0.0229201269767858 w_k1_DIS_s105
                                 + 0.0295511778191371 w_k1_DIS_s106
                                 + 0.0391150660406466 w_k1_DIS_s107
                                 + 0.0207478514503115 w_k1_DIS_s110
                                 + 0.0121802777034613 w_k1_DIS_s111
                                 + 0.0124397631779376 w_k1_DIS_s112
                                 + 0.00753464045108325 w_k1_DIS_s113
                                 + 0.00865253415561353 w_k1_DIS_s114
                                 + 0.0388688122816953 w_k1_DIS_s115
                                 + 0.0261422795874545 w_k1_DIS_s116
                                 + 0.0173536333215392 w_k1_DIS_s117
                                 + 0.0373373181257616 w_k1_FDX_s000
                                 + 0.0312379619167412 w_k1_FDX_s001
                                 + 0.0506990821881921 w_k1_FDX_s002
                                 + 0.0208854221856105 w_k1_FDX_s003
                                 + 0.0166929326527367 w_k1_FDX_s004
                                 + 0.0548434461386734 w_k1_FDX_s005
                                 + 0.0164398028190085 w_k1_FDX_s006
                                 + 0.0127629119936256 w_k1_FDX_s007
                                 + 0.0260359239612568 w_k1_FDX_s010
                                 + 0.0245315108957668 w_k1_FDX_s011
                                 + 0.071392736765107 w_k1_FDX_s012
                                 + 0.0807009920478384 w_k1_FDX_s013
                                 + 0.0292576975352543 w_k1_FDX_s014
                                 + 0.0574226327783478 w_k1_FDX_s015
                                 + 0.0271645716339632 w_k1_FDX_s016
                                 + 0.0243163594016866 w_k1_FDX_s017
                                 + 0.046272889540923 w_k1_FDX_s100
                                 + 0.0245562366855763 w_k1_FDX_s101
                                 + 0.0435627510313609 w_k1_FDX_s102
                                 + 0.0360710196218745 w_k1_FDX_s103
                                 + 0.0568220542936373 w_k1_FDX_s104
                                 + 0.0224100787900156 w_k1_FDX_s105
                                 + 0.0275277235580346 w_k1_FDX_s106
                                 + 0.0380260542890492 w_k1_FDX_s107
                                 + 0.0190424732037487 w_k1_FDX_s110
                                 + 0.01134249879212 w_k1_FDX_s111
                                 + 0.011974909719723 w_k1_FDX_s112
                                 + 0.00762516461596479 w_k1_FDX_s113
                                 + 0.00870257419171873 w_k1_FDX_s114
                                 + 0.0361940110391476 w_k1_FDX_s115
                                 + 0.0248385947831073 w_k1_FDX_s116
                                 + 0.0149474755841545 w_k1_FDX_s117
                                 + 0.0396349615079647 w_k1_IBM_s000
                                 + 0.0340451253012443 w_k1_IBM_s001
                                 + 0.0504433482971643 w_k1_IBM_s002
                                 + 0.0204111124863072 w_k1_IBM_s003
                                 + 0.0170220249775816 w_k1_IBM_s004
                                 + 0.057275037390922 w_k1_IBM_s005
                                 + 0.0158306084063375 w_k1_IBM_s006
                                 + 0.0119970991408658 w_k1_IBM_s007
                                 + 0.0272353330342558 w_k1_IBM_s010
                                 + 0.0255431098389312 w_k1_IBM_s011
                                 + 0.072350361503006 w_k1_IBM_s012
                                 + 0.0808038951759269 w_k1_IBM_s013
                                 + 0.0301698179933551 w_k1_IBM_s014
                                 + 0.0549580583732517 w_k1_IBM_s015
                                 + 0.0274362206456623 w_k1_IBM_s016
                                 + 0.0251329197557193 w_k1_IBM_s017
                                 + 0.0466132373229623 w_k1_IBM_s100
                                 + 0.0239107676005389 w_k1_IBM_s101
                                 + 0.0442051699426075 w_k1_IBM_s102
                                 + 0.0359516701481204 w_k1_IBM_s103
                                 + 0.0570678249635597 w_k1_IBM_s104
                                 + 0.0232669460759941 w_k1_IBM_s105
                                 + 0.0282286862389358 w_k1_IBM_s106
                                 + 0.037879541686526 w_k1_IBM_s107
                                 + 0.0188549992376572 w_k1_IBM_s110
                                 + 0.0117257962496036 w_k1_IBM_s111
                                 + 0.0122531565188537 w_k1_IBM_s112
                                 + 0.00782613769271299 w_k1_IBM_s113
                                 + 0.00913766027916832 w_k1_IBM_s114
                                 + 0.0389606662064257 w_k1_IBM_s115
                                 + 0.0251872468382487 w_k1_IBM_s116
                                 + 0.0152564541244079 w_k1_IBM_s117
                                 + 0.0460278751106957 w_k1_INTC_s000
                                 + 0.0370851493470657 w_k1_INTC_s001
                                 + 0.0464857612464114 w_k1_INTC_s002
                                 + 0.0226292526782228 w_k1_INTC_s003
                                 + 0.0166735478443201 w_k1_INTC_s004
                                 + 0.0620336350526155 w_k1_INTC_s005
                                 + 0.0169844736122639 w_k1_INTC_s006
                                 + 0.0118135606489935 w_k1_INTC_s007
                                 + 0.0243379307739442 w_k1_INTC_s010
                                 + 0.0248092788971239 w_k1_INTC_s011
                                 + 0.0764561123881494 w_k1_INTC_s012
                                 + 0.0808485788767499 w_k1_INTC_s013
                                 + 0.0313684787540757 w_k1_INTC_s014
                                 + 0.0510423589514293 w_k1_INTC_s015
                                 + 0.0281521319279404 w_k1_INTC_s016
                                 + 0.0234899505473857 w_k1_INTC_s017
                                 + 0.0457880627147508 w_k1_INTC_s100
                                 + 0.0245379653249262 w_k1_INTC_s101
                                 + 0.0468225524943349 w_k1_INTC_s102
                                 + 0.036734761371788 w_k1_INTC_s103
                                 + 0.0588957060274696 w_k1_INTC_s104
                                 + 0.0238548638457675 w_k1_INTC_s105
                                 + 0.030726416325546 w_k1_INTC_s106
                                 + 0.0406692977688561 w_k1_INTC_s107
                                 + 0.0187909061642629 w_k1_INTC_s110
                                 + 0.0118632968934538 w_k1_INTC_s111
                                 + 0.0123265401398 w_k1_INTC_s112
                                 + 0.0070929247552343 w_k1_INTC_s113
                                 + 0.00894015500748085 w_k1_INTC_s114
                                 + 0.0362409545025246 w_k1_INTC_s115
                                 + 0.0225978055555022 w_k1_INTC_s116
                                 + 0.0133400055061463 w_k1_INTC_s117
                                 + 0.0408726330485052 w_k1_KO_s000
                                 + 0.0324506817520634 w_k1_KO_s001
                                 + 0.0519372796716256 w_k1_KO_s002
                                 + 0.0204049233018227 w_k1_KO_s003
                                 + 0.0185542293147551 w_k1_KO_s004
                                 + 0.0579014150298892 w_k1_KO_s005
                                 + 0.016833090635327 w_k1_KO_s006
                                 + 0.0113367765138923 w_k1_KO_s007
                                 + 0.0279592774030118 w_k1_KO_s010
                                 + 0.0238840137045603 w_k1_KO_s011
                                 + 0.0699300756422417 w_k1_KO_s012
                                 + 0.0830847185743182 w_k1_KO_s013
                                 + 0.0287478680578155 w_k1_KO_s014
                                 + 0.0585016759914009 w_k1_KO_s015
                                 + 0.0292875104473036 w_k1_KO_s016
                                 + 0.0242865398674817 w_k1_KO_s017
                                 + 0.0461942462402301 w_k1_KO_s100
                                 + 0.0221420969997831 w_k1_KO_s101
                                 + 0.0481779441304047 w_k1_KO_s102
                                 + 0.033640521246377 w_k1_KO_s103
                                 + 0.0575884161069153 w_k1_KO_s104
                                 + 0.0211440206325577 w_k1_KO_s105
                                 + 0.0296102765002699 w_k1_KO_s106
                                 + 0.0365696343746962 w_k1_KO_s107
                                 + 0.0192134978583688 w_k1_KO_s110
                                 + 0.0120631173216514 w_k1_KO_s111
                                 + 0.0136060000050944 w_k1_KO_s112
                                 + 0.00704349068870989 w_k1_KO_s113
                                 + 0.00951615512208261 w_k1_KO_s114
                                 + 0.0372082098831982 w_k1_KO_s115
                                 + 0.0260196933825796 w_k1_KO_s116
                                 + 0.0151618146436994 w_k1_KO_s117
                                 + 0.0364459349817632 w_k1_MAT_s000
                                 + 0.0306051875810303 w_k1_MAT_s001
                                 + 0.0562009713605466 w_k1_MAT_s002
                                 + 0.0171425644963167 w_k1_MAT_s003
                                 + 0.0143819067135304 w_k1_MAT_s004
                                 + 0.0514110663824285 w_k1_MAT_s005
                                 + 0.0157242593683221 w_k1_MAT_s006
                                 + 0.0114286545516699 w_k1_MAT_s007
                                 + 0.0291639657213867 w_k1_MAT_s010
                                 + 0.0268891395306844 w_k1_MAT_s011
                                 + 0.0832632442947921 w_k1_MAT_s012
                                 + 0.0908353481979428 w_k1_MAT_s013
                                 + 0.0342270584041552 w_k1_MAT_s014
                                 + 0.0622026503087606 w_k1_MAT_s015
                                 + 0.0330394344366984 w_k1_MAT_s016
                                 + 0.0241493271139972 w_k1_MAT_s017
                                 + 0.0510842106323979 w_k1_MAT_s100
                                 + 0.0238245377360185 w_k1_MAT_s101
                                 + 0.054590506009758 w_k1_MAT_s102
                                 + 0.0434038433623542 w_k1_MAT_s103
                                 + 0.0655073911227275 w_k1_MAT_s104
                                 + 0.0203584895464813 w_k1_MAT_s105
                                 + 0.0332587787138316 w_k1_MAT_s106
                                 + 0.0327861676264268 w_k1_MAT_s107
                                 + 0.0200382837296122 w_k1_MAT_s110
                                 + 0.011727169382632 w_k1_MAT_s111
                                 + 0.0135326694821763 w_k1_MAT_s112
                                 + 0.00806428253866053 w_k1_MAT_s113
                                 + 0.00956742329329704 w_k1_MAT_s114
                                 + 0.0440968950572591 w_k1_MAT_s115
                                 + 0.0256002326219526 w_k1_MAT_s116
                                 + 0.0139324082436995 w_k1_MAT_s117
                                 + 0.041420930077168 w_k1_MSFT_s000
                                 + 0.032696650442576 w_k1_MSFT_s001
                                 + 0.0481281996579295 w_k1_MSFT_s002
                                 + 0.022058599754462 w_k1_MSFT_s003
                                 + 0.0163074251512691 w_k1_MSFT_s004
                                 + 0.0556779846066471 w_k1_MSFT_s005
                                 + 0.0159081113511646 w_k1_MSFT_s006
                                 + 0.0115549387215647 w_k1_MSFT_s007
                                 + 0.0261785768435045 w_k1_MSFT_s010
                                 + 0.0253018339023876 w_k1_MSFT_s011
                                 + 0.0761693433014299 w_k1_MSFT_s012
                                 + 0.0756564934353829 w_k1_MSFT_s013
                                 + 0.0322141319490871 w_k1_MSFT_s014
                                 + 0.0553564951131156 w_k1_MSFT_s015
                                 + 0.026920962365821 w_k1_MSFT_s016
                                 + 0.0226777948282538 w_k1_MSFT_s017
                                 + 0.0436255185976854 w_k1_MSFT_s100
                                 + 0.024836422483755 w_k1_MSFT_s101
                                 + 0.0451636554301453 w_k1_MSFT_s102
                                 + 0.0348595087946165 w_k1_MSFT_s103
                                 + 0.0569124239989293 w_k1_MSFT_s104
                                 + 0.0229037422328661 w_k1_MSFT_s105
                                 + 0.0295797917145731 w_k1_MSFT_s106
                                 + 0.0360132189502702 w_k1_MSFT_s107
                                 + 0.0170804845806599 w_k1_MSFT_s110
                                 + 0.0125691138042831 w_k1_MSFT_s111
                                 + 0.0132734030614582 w_k1_MSFT_s112
                                 + 0.00776267988063961 w_k1_MSFT_s113
                                 + 0.00879332536067369 w_k1_MSFT_s114
                                 + 0.038353980901809 w_k1_MSFT_s115
                                 + 0.0239118261348741 w_k1_MSFT_s116
                                 + 0.0149894188583904 w_k1_MSFT_s117
                                 + 0.0386540881880481 w_k1_PG_s000
                                 + 0.0321613714828342 w_k1_PG_s001
                                 + 0.0522611556685994 w_k1_PG_s002
                                 + 0.020615709583974 w_k1_PG_s003
                                 + 0.0167151683039311 w_k1_PG_s004
                                 + 0.056490139803255 w_k1_PG_s005
                                 + 0.0163918707735787 w_k1_PG_s006
                                 + 0.0123547143647242 w_k1_PG_s007
                                 + 0.0284477138998342 w_k1_PG_s010
                                 + 0.0270404631357528 w_k1_PG_s011
                                 + 0.0760216886229578 w_k1_PG_s012
                                 + 0.0760253868009649 w_k1_PG_s013
                                 + 0.029654428997291 w_k1_PG_s014
                                 + 0.0551297304296678 w_k1_PG_s015
                                 + 0.0284849203985683 w_k1_PG_s016
                                 + 0.0212355751228493 w_k1_PG_s017
                                 + 0.0471073573638323 w_k1_PG_s100
                                 + 0.0234996951318583 w_k1_PG_s101
                                 + 0.0456369494660554 w_k1_PG_s102
                                 + 0.0357709448373285 w_k1_PG_s103
                                 + 0.058383960673385 w_k1_PG_s104
                                 + 0.0226282620490426 w_k1_PG_s105
                                 + 0.0311863972271652 w_k1_PG_s106
                                 + 0.0344382067753547 w_k1_PG_s107
                                 + 0.0187679201085577 w_k1_PG_s110
                                 + 0.0120631985579372 w_k1_PG_s111
                                 + 0.0125291796774649 w_k1_PG_s112
                                 + 0.00772966703392555 w_k1_PG_s113
                                 + 0.00910675116835117 w_k1_PG_s114
                                 + 0.0359788118830857 w_k1_PG_s115
                                 + 0.0266422342648336 w_k1_PG_s116
                                 + 0.0143441869212611 w_k1_PG_s117 >= 1.05
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
