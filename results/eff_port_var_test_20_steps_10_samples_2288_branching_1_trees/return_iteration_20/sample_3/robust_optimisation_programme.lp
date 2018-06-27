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
 w_k1_ADM_s0_return_constraint:  1.14942 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.979699 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.29907 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.07033 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.07168 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.00329 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.982154 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.05863 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.01329 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.819855 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.684858 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.669723 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.03588 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.15263 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.933942 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.569472 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.949755 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.785292 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.980579 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.18391 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.784862 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.908619 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.42077 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.867725 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.642612 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.05834 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.914524 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.831722 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.916237 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.39423 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.06908 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.37832 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.626784 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.73209 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.881357 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.2694 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.999109 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.934561 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20041 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.960166 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.37928 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.992997 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.28032 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.822692 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.28116 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.67089 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.2137 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.909325 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.740182 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.755649 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.08003 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.22192 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.29621 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.549982 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.945405 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.756163 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.11398 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.45628 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.644079 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.946813 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.36823 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.95349 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.630165 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.17514 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.724441 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.822736 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.08569 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.42788 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.14051 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.59829 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.604544 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.804324 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01905 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.4442 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.14399 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.969075 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33965 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.924484 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.39259 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.890738 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.23779 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.904484 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.11258 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.78712 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.47691 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.899689 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.859535 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.637791 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.946399 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.11811 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.35214 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.551326 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.894748 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.798807 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.07628 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.66773 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.737649 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.25549 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.83383 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.887529 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.638906 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.1796 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.671575 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.876094 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.06141 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.50006 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.0229 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 2.07473 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.502667 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.637792 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.10097 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.5139 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.13749 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.819166 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.22435 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.927176 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.20766 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.949071 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.10826 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.95466 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.982983 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.49875 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.32094 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.879307 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.937719 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.716663 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.18255 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.09103 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.153 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.687194 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.868987 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.899288 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.09748 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.41058 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.720799 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.18866 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.49534 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.904902 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.70202 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.11386 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.816151 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.953713 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.940518 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.27491 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.00922 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.55791 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.711038 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.773412 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.07444 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.34167 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.20693 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.883976 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.03691 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.02259 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.42719 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.925202 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.32033 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.802738 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.61359 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.45254 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.818679 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.13519 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.726656 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.914741 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.670431 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.996783 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.55581 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.668339 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.33724 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.674241 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.995955 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.10444 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.00905 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.712934 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.03405 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.32158 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.851551 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.998905 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.581354 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.934856 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.84085 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.50271 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.47349 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.19018 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.544802 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.795295 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.10459 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.3752 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.808792 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.04191 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1067 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.982511 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.26421 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.938037 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.02752 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.928824 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.871473 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.62072 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.23531 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.818248 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.933356 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.772039 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.868414 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.03142 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.13826 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.847197 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01288 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.984567 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.06344 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.50714 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.788735 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.1786 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.41969 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.09776 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.892882 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0617 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.833247 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.928821 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.07458 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.36669 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.0569 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.27968 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.68557 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.827142 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.1592 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.26439 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.16612 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.98641 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15662 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.987511 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.31983 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.856561 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.18872 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.829039 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.2192 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.45518 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.03739 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.12409 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.750171 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.833163 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.770055 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.09511 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.37067 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.648929 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.0628 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.832374 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.08969 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.36778 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.931034 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11474 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.19455 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.11757 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.733919 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.04947 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.70073 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.903131 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.27104 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.45567 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.23076 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.4936 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.561216 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.800313 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.13867 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.3109 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.964903 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.01274 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15311 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.01994 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.58293 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.879402 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.38644 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.671562 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.61361 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.38361 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.867443 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.15386 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.584401 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.921635 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.84452 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.914277 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.82808 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.542749 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.2231 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.737001 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.922458 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.928666 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.895616 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.660835 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.820582 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.38533 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.732705 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.11568 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.530615 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.892482 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.64164 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.54371 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.34876 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.86135 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.67597 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.895639 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.2198 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.65555 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.18487 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.27127 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.23095 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.999043 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.17191 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.913804 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.01477 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.878592 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.814224 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.38859 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.45441 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.880385 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.921162 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.736862 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.95081 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.934194 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.13635 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.754315 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.944611 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.90229 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.0382 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.45957 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.840972 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.12583 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.39301 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.957897 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.826874 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.11765 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.803334 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.948045 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.01888 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.25988 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.06666 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.61706 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.721973 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.896337 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.1449 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.31181 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.18108 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.97263 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13959 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.952343 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.19581 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.972394 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.15067 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.951568 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.10823 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.24986 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00144 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.90283 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.900147 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.869609 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.01149 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08886 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.06656 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.709112 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.95302 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.819406 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02005 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.07317 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.793553 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.978007 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.20046 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.967845 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.806834 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.00691 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.818843 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.919821 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04485 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.17009 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08335 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.30303 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.73434 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.861845 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.04019 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.20578 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01099 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.971176 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0560002534690959 z_k1_s0000
                                 - 0.195000782829417 z_k1_s0001
                                 - 0.159000685199592 z_k1_s0002
                                 - 0.0500002371974584 z_k1_s0003
                                 - 0.146000903945574 z_k1_s0004
                                 - 0.171000717742867 z_k1_s0005
                                 - 0.107000544178733 z_k1_s0006
                                 - 0.132000357976026 z_k1_s0007
                                 - 0.0590000181226312 z_k1_s0010
                                 - 0.0969996484252154 z_k1_s0011
                                 - 0.18299977317349 z_k1_s0012
                                 - 0.0710000045340081 z_k1_s0013
                                 - 0.0519999933826505 z_k1_s0014
                                 - 0.125999818119444 z_k1_s0015
                                 - 0.0809997710767479 z_k1_s0016
                                 - 0.115000294442503 z_k1_s0017
                                 - 0.0610003377060973 z_k1_s0020
                                 - 0.0960001486042966 z_k1_s0021
                                 - 0.122000499811268 z_k1_s0022
                                 - 0.0980002431590354 z_k1_s0023
                                 - 0.149000898295609 z_k1_s0024
                                 - 0.0810004052488522 z_k1_s0025
                                 - 0.112000905042207 z_k1_s0026
                                 - 0.159000493064669 z_k1_s0027
                                 - 0.128000222033609 z_k1_s0030
                                 - 0.262000535490795 z_k1_s0031
                                 - 0.0720001637814659 z_k1_s0032
                                 - 0.0840001392282955 z_k1_s0033
                                 - 0.130000598175344 z_k1_s0034
                                 - 0.1150003696164 z_k1_s0035
                                 - 0.0740001251225499 z_k1_s0036
                                 - 0.172000201138759 z_k1_s0037
                                 - 0.17800090736644 z_k1_s0040
                                 - 0.148999957389903 z_k1_s0041
                                 - 0.0910000779396145 z_k1_s0042
                                 - 0.197999843934097 z_k1_s0043
                                 - 0.246000255100558 z_k1_s0044
                                 - 0.0700000495919969 z_k1_s0045
                                 - 0.140000637984736 z_k1_s0046
                                 - 0.274999858075238 z_k1_s0047
                                 - 0.250000352643639 z_k1_s0050
                                 - 0.116999708756507 z_k1_s0051
                                 - 0.180000668704072 z_k1_s0052
                                 - 0.094000153334041 z_k1_s0053
                                 - 0.150000211586184 z_k1_s0054
                                 - 0.0820001778872115 z_k1_s0055
                                 - 0.103000186769245 z_k1_s0056
                                 - 0.0610000653050155 z_k1_s0057
                                 - 0.050000078797515 z_k1_s0060
                                 - 0.0779998586038231 z_k1_s0061
                                 - 0.0870003188278825 z_k1_s0062
                                 - 0.0480000355255688 z_k1_s0063
                                 - 0.123000389722109 z_k1_s0064
                                 - 0.0649997838364076 z_k1_s0065
                                 - 0.0510000709334545 z_k1_s0066
                                 - 0.0880003699638891 z_k1_s0067
                                 - 0.274999762369651 z_k1_s0070
                                 - 0.10900005011199 z_k1_s0071
                                 - 0.104000054432542 z_k1_s0072
                                 - 0.466999596460461 z_k1_s0073
                                 - 0.119000041470887 z_k1_s0074
                                 - 0.104000054432542 z_k1_s0075
                                 - 0.132000030237452 z_k1_s0076
                                 - 0.133000029373342 z_k1_s0077
                                 - 0.310001767395079 z_k1_s0100
                                 - 0.421001193604192 z_k1_s0101
                                 - 0.121000324208041 z_k1_s0102
                                 - 0.279000457152454 z_k1_s0103
                                 - 0.182000474536853 z_k1_s0104
                                 - 0.360000589874134 z_k1_s0105
                                 - 0.103000294714335 z_k1_s0106
                                 - 0.491001812081104 z_k1_s0107
                                 - 0.217999082668264 z_k1_s0110
                                 - 0.140999969072321 z_k1_s0111
                                 - 0.177999973594839 z_k1_s0112
                                 - 0.328000107518504 z_k1_s0113
                                 - 0.203999697432419 z_k1_s0114
                                 - 0.382999728944055 z_k1_s0115
                                 - 0.161999937925658 z_k1_s0116
                                 - 0.167999929026612 z_k1_s0117
                                 - 0.594000334790788 z_k1_s0120
                                 - 0.728001201519979 z_k1_s0121
                                 - 0.354000752280739 z_k1_s0122
                                 - 0.225000614543183 z_k1_s0123
                                 - 0.511001431462608 z_k1_s0124
                                 - 0.31800065395383 z_k1_s0125
                                 - 0.191000485912124 z_k1_s0126
                                 - 0.298000635094525 z_k1_s0127
                                 - 0.484997662448653 z_k1_s0130
                                 - 0.695997784525851 z_k1_s0131
                                 - 0.885997413174632 z_k1_s0132
                                 - 0.508999967220126 z_k1_s0133
                                 - 0.269999322638073 z_k1_s0134
                                 - 0.302999343655829 z_k1_s0135
                                 - 0.70099809543695 z_k1_s0136
                                 - 0.211999371861219 z_k1_s0137
                                 - 0.282000649414623 z_k1_s0140
                                 - 0.625999770541344 z_k1_s0141
                                 - 0.835001839358742 z_k1_s0142
                                 - 0.183000342649765 z_k1_s0143
                                 - 0.407000662743278 z_k1_s0144
                                 - 0.253000484754068 z_k1_s0145
                                 - 0.305000571082942 z_k1_s0146
                                 - 0.475000723850169 z_k1_s0147
                                 - 0.387998458738454 z_k1_s0150
                                 - 0.407997764882064 z_k1_s0151
                                 - 0.173999355627883 z_k1_s0152
                                 - 0.446997405158139 z_k1_s0153
                                 - 0.131999393731966 z_k1_s0154
                                 - 0.555998873919853 z_k1_s0155
                                 - 0.535996729787575 z_k1_s0156
                                 - 0.196999267190202 z_k1_s0157
                                 - 0.243999901565492 z_k1_s0160
                                 - 0.32099922984637 z_k1_s0161
                                 - 0.473000898297101 z_k1_s0162
                                 - 0.13000001802752 z_k1_s0163
                                 - 0.326999964179989 z_k1_s0164
                                 - 0.138000085272294 z_k1_s0165
                                 - 0.14299998075027 z_k1_s0166
                                 - 0.178000030696143 z_k1_s0167
                                 - 0.222000756924737 z_k1_s0170
                                 - 0.344000744602473 z_k1_s0171
                                 - 0.266001570795221 z_k1_s0172
                                 - 0.130000524619299 z_k1_s0173
                                 - 0.138000586159059 z_k1_s0174
                                 - 0.211000910008468 z_k1_s0175
                                 - 0.488002327719958 z_k1_s0176
                                 - 0.578003020042259 z_k1_s0177
                                 - 0.186000000124668 z_k1_s1000
                                 - 0.383999011044509 z_k1_s1001
                                 - 0.518998834489839 z_k1_s1002
                                 - 0.120999882317664 z_k1_s1003
                                 - 0.265000059059337 z_k1_s1004
                                 - 0.353998746056669 z_k1_s1005
                                 - 0.547000359545114 z_k1_s1006
                                 - 0.361999181786747 z_k1_s1007
                                 - 0.336999950267406 z_k1_s1010
                                 - 0.521000097831946 z_k1_s1011
                                 - 0.334000011861514 z_k1_s1012
                                 - 0.293000362980593 z_k1_s1013
                                 - 0.714000060614162 z_k1_s1014
                                 - 0.233000122861567 z_k1_s1015
                                 - 0.350000664693726 z_k1_s1016
                                 - 0.162000206721015 z_k1_s1017
                                 - 1.04999929764223 z_k1_s1020
                                 - 0.234999820753286 z_k1_s1021
                                 - 0.747998247079924 z_k1_s1022
                                 - 0.543998304149527 z_k1_s1023
                                 - 0.241999569084522 z_k1_s1024
                                 - 0.881000411865309 z_k1_s1025
                                 - 0.59299885796548 z_k1_s1026
                                 - 0.337999623951673 z_k1_s1027
                                 - 0.259999552213835 z_k1_s1030
                                 - 0.67799785112321 z_k1_s1031
                                 - 0.405000727266285 z_k1_s1032
                                 - 0.710001039220142 z_k1_s1033
                                 - 0.362999789035826 z_k1_s1034
                                 - 0.639998839003346 z_k1_s1035
                                 - 0.417000013304538 z_k1_s1036
                                 - 0.345999527482904 z_k1_s1037
                                 - 0.594999361720939 z_k1_s1040
                                 - 0.672000818129654 z_k1_s1041
                                 - 0.368999514842191 z_k1_s1042
                                 - 0.26199954030763 z_k1_s1043
                                 - 0.520998492963742 z_k1_s1044
                                 - 0.302999668089833 z_k1_s1045
                                 - 0.961999029861328 z_k1_s1046
                                 - 0.403999284920172 z_k1_s1047
                                 - 0.461000056740956 z_k1_s1050
                                 - 1.29899684721287 z_k1_s1051
                                 - 1.49500133617046 z_k1_s1052
                                 - 0.761001592293869 z_k1_s1053
                                 - 0.418999525335637 z_k1_s1054
                                 - 0.829003457197437 z_k1_s1055
                                 - 1.18200139406778 z_k1_s1056
                                 - 1.08799970246555 z_k1_s1057
                                 - 0.357999445808955 z_k1_s1060
                                 - 0.494000124619445 z_k1_s1061
                                 - 0.124000129421118 z_k1_s1062
                                 - 0.103999997672502 z_k1_s1063
                                 - 0.169000109280365 z_k1_s1064
                                 - 0.281999704944545 z_k1_s1065
                                 - 0.134000104845386 z_k1_s1066
                                 - 0.144000080269655 z_k1_s1067
                                 - 0.195999353695591 z_k1_s1070
                                 - 0.252999215418643 z_k1_s1071
                                 - 0.749996196783234 z_k1_s1072
                                 - 0.531998918814487 z_k1_s1073
                                 - 0.240999376781795 z_k1_s1074
                                 - 0.5479991224623 z_k1_s1075
                                 - 0.456998356839129 z_k1_s1076
                                 - 0.163999574597989 z_k1_s1077
                                 - 0.421001218480691 z_k1_s1100
                                 - 1.53100752205999 z_k1_s1101
                                 - 0.59599957837409 z_k1_s1102
                                 - 0.582004341917995 z_k1_s1103
                                 - 0.485001377924226 z_k1_s1104
                                 - 0.378001151745325 z_k1_s1105
                                 - 0.862002131348807 z_k1_s1106
                                 - 0.315000873620818 z_k1_s1107
                                 - 0.739000738949047 z_k1_s1110
                                 - 0.259999935612883 z_k1_s1111
                                 - 0.510001300286999 z_k1_s1112
                                 - 0.152000034752156 z_k1_s1113
                                 - 0.464999150261403 z_k1_s1114
                                 - 0.175999981965647 z_k1_s1115
                                 - 0.245999943338343 z_k1_s1116
                                 - 0.220000115857077 z_k1_s1117
                                 - 0.233000131477183 z_k1_s1120
                                 - 0.444001015408976 z_k1_s1121
                                 - 0.151000031644005 z_k1_s1122
                                 - 0.447001207783384 z_k1_s1123
                                 - 0.383999663921125 z_k1_s1124
                                 - 0.148000088869628 z_k1_s1125
                                 - 0.217999918405236 z_k1_s1126
                                 - 0.471000250778338 z_k1_s1127
                                 - 0.459999140532964 z_k1_s1130
                                 - 0.571998864280223 z_k1_s1131
                                 - 1.79399348088343 z_k1_s1132
                                 - 1.9259970378653 z_k1_s1133
                                 - 0.635999317849145 z_k1_s1134
                                 - 1.34399497296975 z_k1_s1135
                                 - 0.563999556582956 z_k1_s1136
                                 - 1.26400189599707 z_k1_s1137
                                 - 0.291000545442975 z_k1_s1140
                                 - 0.466999165230377 z_k1_s1141
                                 - 0.234000049943969 z_k1_s1142
                                 - 0.178999915760599 z_k1_s1143
                                 - 0.144999993892643 z_k1_s1144
                                 - 0.169999986339652 z_k1_s1145
                                 - 0.117000024971984 z_k1_s1146
                                 - 0.281999673522351 z_k1_s1147
                                 - 0.719003334214758 z_k1_s1150
                                 - 0.253000682451204 z_k1_s1151
                                 - 0.465000225387293 z_k1_s1152
                                 - 0.360001227490273 z_k1_s1153
                                 - 0.928004688450866 z_k1_s1154
                                 - 0.526001134673144 z_k1_s1155
                                 - 0.369001165207235 z_k1_s1156
                                 - 0.512001874315462 z_k1_s1157
                                 - 0.248999300431706 z_k1_s1160
                                 - 0.888998202022062 z_k1_s1161
                                 - 0.873998447296918 z_k1_s1162
                                 - 0.716998391844832 z_k1_s1163
                                 - 1.216997709337 z_k1_s1164
                                 - 0.3929991937893 z_k1_s1165
                                 - 0.472999196987857 z_k1_s1166
                                 - 0.807997278510073 z_k1_s1167
                                 - 1.84999839806638 z_k1_s1170
                                 - 0.470998987740426 z_k1_s1171
                                 - 0.46299930858916 z_k1_s1172
                                 - 1.89499388454836 z_k1_s1173
                                 - 0.934000896723719 z_k1_s1174
                                 - 1.29799539948577 z_k1_s1175
                                 - 0.416999203173783 z_k1_s1176
                                 - 1.34000043271476 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.0558 w_k1_ADM_s000
                                 + 1.52293 w_k1_AXP_s000
                                 + 1.43661 w_k1_DIS_s000
                                 + 1.24959 w_k1_FDX_s000
                                 + 1.23828 w_k1_IBM_s000
                                 + 1.69617 w_k1_INTC_s000
                                 + 1.46945 w_k1_KO_s000 + 1.94727 w_k1_MAT_s000
                                 + 1.54482 w_k1_MSFT_s000
                                 + 1.24794 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.749609 w_k1_ADM_s000
                                 + 0.608246 w_k1_AXP_s000
                                 + 0.714617 w_k1_DIS_s000
                                 + 0.70633 w_k1_FDX_s000
                                 + 0.949943 w_k1_IBM_s000
                                 + 0.77863 w_k1_INTC_s000
                                 + 0.874536 w_k1_KO_s000
                                 + 0.684341 w_k1_MAT_s000
                                 + 0.885481 w_k1_MSFT_s000
                                 + 0.835147 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.964715 w_k1_ADM_s000
                                 + 0.894814 w_k1_AXP_s000
                                 + 1.10963 w_k1_DIS_s000
                                 + 1.11656 w_k1_FDX_s000
                                 + 1.11232 w_k1_IBM_s000
                                 + 1.15653 w_k1_INTC_s000
                                 + 1.25545 w_k1_KO_s000 + 1.00132 w_k1_MAT_s000
                                 + 1.23413 w_k1_MSFT_s000
                                 + 0.979978 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.24958 w_k1_ADM_s000
                                 + 1.28055 w_k1_AXP_s000
                                 + 1.62419 w_k1_DIS_s000
                                 + 1.45063 w_k1_FDX_s000
                                 + 0.962276 w_k1_IBM_s000
                                 + 1.56791 w_k1_INTC_s000
                                 + 1.32067 w_k1_KO_s000
                                 + 0.900831 w_k1_MAT_s000
                                 + 1.79199 w_k1_MSFT_s000
                                 + 1.25946 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.878379 w_k1_ADM_s000
                                 + 1.16446 w_k1_AXP_s000
                                 + 1.26636 w_k1_DIS_s000
                                 + 1.02618 w_k1_FDX_s000
                                 + 1.63379 w_k1_IBM_s000
                                 + 1.32616 w_k1_INTC_s000
                                 + 1.30897 w_k1_KO_s000 + 1.82775 w_k1_MAT_s000
                                 + 1.27335 w_k1_MSFT_s000
                                 + 1.13618 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.14959 w_k1_ADM_s000
                                 + 1.00228 w_k1_AXP_s000
                                 + 1.03342 w_k1_DIS_s000
                                 + 0.990048 w_k1_FDX_s000
                                 + 0.978187 w_k1_IBM_s000
                                 + 1.25448 w_k1_INTC_s000
                                 + 0.986348 w_k1_KO_s000
                                 + 0.978742 w_k1_MAT_s000
                                 + 1.30031 w_k1_MSFT_s000
                                 + 0.878613 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.848635 w_k1_ADM_s000
                                 + 0.759425 w_k1_AXP_s000
                                 + 0.801244 w_k1_DIS_s000
                                 + 1.0148 w_k1_FDX_s000
                                 + 0.668115 w_k1_IBM_s000
                                 + 0.97645 w_k1_INTC_s000
                                 + 0.827788 w_k1_KO_s000
                                 + 0.669098 w_k1_MAT_s000
                                 + 1.30343 w_k1_MSFT_s000
                                 + 0.895541 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.00588 w_k1_ADM_s000
                                 + 1.05755 w_k1_AXP_s000
                                 + 1.19008 w_k1_DIS_s000 + 1.2304 w_k1_FDX_s000
                                 + 0.821785 w_k1_IBM_s000
                                 + 1.44183 w_k1_INTC_s000
                                 + 0.948535 w_k1_KO_s000
                                 + 1.04364 w_k1_MAT_s000
                                 + 1.4003 w_k1_MSFT_s000 + 1.01664 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.776443 w_k1_ADM_s001
                                 + 0.663593 w_k1_AXP_s001
                                 + 0.848567 w_k1_DIS_s001
                                 + 0.755281 w_k1_FDX_s001
                                 + 0.971152 w_k1_IBM_s001
                                 + 0.697659 w_k1_INTC_s001
                                 + 0.928336 w_k1_KO_s001
                                 + 0.781575 w_k1_MAT_s001
                                 + 0.87347 w_k1_MSFT_s001
                                 + 0.949455 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.18537 w_k1_ADM_s001
                                 + 0.925448 w_k1_AXP_s001
                                 + 0.80655 w_k1_DIS_s001
                                 + 0.969576 w_k1_FDX_s001
                                 + 0.719809 w_k1_IBM_s001
                                 + 0.721511 w_k1_INTC_s001
                                 + 0.866625 w_k1_KO_s001
                                 + 0.660551 w_k1_MAT_s001
                                 + 0.846146 w_k1_MSFT_s001
                                 + 1.04093 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.00595 w_k1_ADM_s001
                                 + 0.943479 w_k1_AXP_s001
                                 + 1.17999 w_k1_DIS_s001
                                 + 1.00647 w_k1_FDX_s001
                                 + 0.996645 w_k1_IBM_s001
                                 + 0.764051 w_k1_INTC_s001
                                 + 1.17275 w_k1_KO_s001 + 1.05985 w_k1_MAT_s001
                                 + 1.01786 w_k1_MSFT_s001
                                 + 1.05249 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.27331 w_k1_ADM_s001
                                 + 1.49122 w_k1_AXP_s001
                                 + 1.66372 w_k1_DIS_s001
                                 + 1.39807 w_k1_FDX_s001
                                 + 1.33964 w_k1_IBM_s001
                                 + 1.11653 w_k1_INTC_s001
                                 + 1.38263 w_k1_KO_s001 + 1.3187 w_k1_MAT_s001
                                 + 1.22659 w_k1_MSFT_s001
                                 + 1.07059 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.19764 w_k1_ADM_s001
                                 + 0.675439 w_k1_AXP_s001
                                 + 0.990668 w_k1_DIS_s001
                                 + 0.858133 w_k1_FDX_s001
                                 + 0.701858 w_k1_IBM_s001
                                 + 0.554904 w_k1_INTC_s001
                                 + 0.919079 w_k1_KO_s001
                                 + 0.656356 w_k1_MAT_s001
                                 + 0.833023 w_k1_MSFT_s001
                                 + 0.907345 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.784832 w_k1_ADM_s001
                                 + 0.747943 w_k1_AXP_s001
                                 + 0.611519 w_k1_DIS_s001
                                 + 0.755378 w_k1_FDX_s001
                                 + 0.881052 w_k1_IBM_s001
                                 + 0.734102 w_k1_INTC_s001
                                 + 0.648391 w_k1_KO_s001
                                 + 0.92089 w_k1_MAT_s001
                                 + 0.777562 w_k1_MSFT_s001
                                 + 0.781899 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.26223 w_k1_ADM_s001
                                 + 1.45247 w_k1_AXP_s001
                                 + 0.902136 w_k1_DIS_s001
                                 + 1.00111 w_k1_FDX_s001
                                 + 1.25015 w_k1_IBM_s001
                                 + 0.742524 w_k1_INTC_s001
                                 + 1.03014 w_k1_KO_s001 + 1.46858 w_k1_MAT_s001
                                 + 0.904508 w_k1_MSFT_s001
                                 + 1.23185 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.10049 w_k1_ADM_s001
                                 + 1.02764 w_k1_AXP_s001
                                 + 1.67841 w_k1_DIS_s001
                                 + 1.08143 w_k1_FDX_s001
                                 + 1.24959 w_k1_IBM_s001
                                 + 0.899356 w_k1_INTC_s001
                                 + 1.55835 w_k1_KO_s001 + 1.29026 w_k1_MAT_s001
                                 + 1.40404 w_k1_MSFT_s001
                                 + 1.20952 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.01366 w_k1_ADM_s002
                                 + 0.6169 w_k1_AXP_s002
                                 + 0.845817 w_k1_DIS_s002
                                 + 0.78001 w_k1_FDX_s002
                                 + 0.738692 w_k1_IBM_s002
                                 + 0.727897 w_k1_INTC_s002
                                 + 0.884207 w_k1_KO_s002
                                 + 0.602141 w_k1_MAT_s002
                                 + 0.823519 w_k1_MSFT_s002
                                 + 0.8821 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.0258 w_k1_ADM_s002
                                 + 0.924803 w_k1_AXP_s002
                                 + 0.909871 w_k1_DIS_s002
                                 + 0.895082 w_k1_FDX_s002
                                 + 1.0987 w_k1_IBM_s002
                                 + 0.850647 w_k1_INTC_s002
                                 + 0.978824 w_k1_KO_s002
                                 + 0.818861 w_k1_MAT_s002
                                 + 0.748906 w_k1_MSFT_s002
                                 + 1.02543 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.1785 w_k1_ADM_s002
                                 + 1.1951 w_k1_AXP_s002 + 1.58299 w_k1_DIS_s002
                                 + 1.16404 w_k1_FDX_s002
                                 + 1.01173 w_k1_IBM_s002
                                 + 1.26024 w_k1_INTC_s002
                                 + 1.30564 w_k1_KO_s002 + 1.21189 w_k1_MAT_s002
                                 + 1.11798 w_k1_MSFT_s002
                                 + 1.14555 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20407 w_k1_ADM_s002
                                 + 1.19492 w_k1_AXP_s002
                                 + 1.08065 w_k1_DIS_s002
                                 + 1.04236 w_k1_FDX_s002
                                 + 1.27321 w_k1_IBM_s002
                                 + 0.950608 w_k1_INTC_s002
                                 + 1.25508 w_k1_KO_s002 + 1.50975 w_k1_MAT_s002
                                 + 0.889282 w_k1_MSFT_s002
                                 + 1.15232 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.09918 w_k1_ADM_s002
                                 + 0.70611 w_k1_AXP_s002
                                 + 0.680566 w_k1_DIS_s002
                                 + 0.82268 w_k1_FDX_s002
                                 + 0.860189 w_k1_IBM_s002
                                 + 0.649046 w_k1_INTC_s002
                                 + 0.739895 w_k1_KO_s002
                                 + 0.649293 w_k1_MAT_s002
                                 + 0.638301 w_k1_MSFT_s002
                                 + 0.823825 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.927857 w_k1_ADM_s002
                                 + 1.22515 w_k1_AXP_s002
                                 + 1.30885 w_k1_DIS_s002
                                 + 1.05701 w_k1_FDX_s002
                                 + 1.65991 w_k1_IBM_s002
                                 + 1.08106 w_k1_INTC_s002
                                 + 1.31271 w_k1_KO_s002 + 1.4741 w_k1_MAT_s002
                                 + 0.918436 w_k1_MSFT_s002
                                 + 1.28448 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.04012 w_k1_ADM_s002
                                 + 1.00784 w_k1_AXP_s002
                                 + 1.04078 w_k1_DIS_s002
                                 + 1.12967 w_k1_FDX_s002
                                 + 0.905583 w_k1_IBM_s002
                                 + 1.02082 w_k1_INTC_s002
                                 + 1.00628 w_k1_KO_s002 + 0.83381 w_k1_MAT_s002
                                 + 0.952748 w_k1_MSFT_s002
                                 + 1.02984 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.16087 w_k1_ADM_s002
                                 + 1.18138 w_k1_AXP_s002
                                 + 0.926093 w_k1_DIS_s002
                                 + 1.22068 w_k1_FDX_s002
                                 + 0.760973 w_k1_IBM_s002
                                 + 0.970924 w_k1_INTC_s002
                                 + 0.936234 w_k1_KO_s002
                                 + 0.838368 w_k1_MAT_s002
                                 + 0.964732 w_k1_MSFT_s002
                                 + 1.0646 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.13254 w_k1_ADM_s003
                                 + 0.93916 w_k1_AXP_s003
                                 + 0.836521 w_k1_DIS_s003
                                 + 1.0139 w_k1_FDX_s003
                                 + 0.695268 w_k1_IBM_s003
                                 + 1.11769 w_k1_INTC_s003
                                 + 0.71482 w_k1_KO_s003
                                 + 0.499789 w_k1_MAT_s003
                                 + 1.05661 w_k1_MSFT_s003
                                 + 0.991479 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.34254 w_k1_ADM_s003
                                 + 1.15899 w_k1_AXP_s003
                                 + 1.34842 w_k1_DIS_s003
                                 + 1.23083 w_k1_FDX_s003
                                 + 1.18512 w_k1_IBM_s003
                                 + 1.09269 w_k1_INTC_s003
                                 + 1.26135 w_k1_KO_s003 + 1.24882 w_k1_MAT_s003
                                 + 1.21562 w_k1_MSFT_s003
                                 + 1.20751 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.11523 w_k1_ADM_s003
                                 + 0.764784 w_k1_AXP_s003
                                 + 1.17894 w_k1_DIS_s003
                                 + 1.06749 w_k1_FDX_s003
                                 + 0.776505 w_k1_IBM_s003
                                 + 1.35383 w_k1_INTC_s003
                                 + 0.843363 w_k1_KO_s003
                                 + 0.400608 w_k1_MAT_s003
                                 + 1.19732 w_k1_MSFT_s003
                                 + 1.04218 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.879068 w_k1_ADM_s003
                                 + 0.902415 w_k1_AXP_s003
                                 + 0.663256 w_k1_DIS_s003
                                 + 0.731488 w_k1_FDX_s003
                                 + 1.06835 w_k1_IBM_s003
                                 + 0.822285 w_k1_INTC_s003
                                 + 0.658612 w_k1_KO_s003
                                 + 1.20695 w_k1_MAT_s003
                                 + 0.706496 w_k1_MSFT_s003
                                 + 0.926717 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.07704 w_k1_ADM_s003
                                 + 0.877497 w_k1_AXP_s003
                                 + 0.855424 w_k1_DIS_s003
                                 + 0.873002 w_k1_FDX_s003
                                 + 0.966234 w_k1_IBM_s003
                                 + 0.92611 w_k1_INTC_s003
                                 + 0.777474 w_k1_KO_s003
                                 + 0.82486 w_k1_MAT_s003
                                 + 0.899149 w_k1_MSFT_s003
                                 + 1.0099 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.15709 w_k1_ADM_s003
                                 + 1.26087 w_k1_AXP_s003
                                 + 1.18483 w_k1_DIS_s003
                                 + 1.01022 w_k1_FDX_s003
                                 + 1.67078 w_k1_IBM_s003
                                 + 1.03294 w_k1_INTC_s003
                                 + 1.09543 w_k1_KO_s003 + 1.34741 w_k1_MAT_s003
                                 + 0.923985 w_k1_MSFT_s003
                                 + 1.04092 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.724427 w_k1_ADM_s003
                                 + 0.67958 w_k1_AXP_s003
                                 + 0.659863 w_k1_DIS_s003
                                 + 0.720234 w_k1_FDX_s003
                                 + 0.782431 w_k1_IBM_s003
                                 + 0.819385 w_k1_INTC_s003
                                 + 0.594604 w_k1_KO_s003
                                 + 0.408434 w_k1_MAT_s003
                                 + 0.708916 w_k1_MSFT_s003
                                 + 0.845753 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.979664 w_k1_ADM_s003
                                 + 0.879067 w_k1_AXP_s003
                                 + 1.04563 w_k1_DIS_s003
                                 + 1.04139 w_k1_FDX_s003
                                 + 1.10284 w_k1_IBM_s003
                                 + 1.18892 w_k1_INTC_s003
                                 + 1.02876 w_k1_KO_s003
                                 + 0.836681 w_k1_MAT_s003
                                 + 1.19532 w_k1_MSFT_s003
                                 + 1.13564 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.18485 w_k1_ADM_s004
                                 + 0.816577 w_k1_AXP_s004
                                 + 0.796227 w_k1_DIS_s004
                                 + 0.922131 w_k1_FDX_s004
                                 + 0.913821 w_k1_IBM_s004
                                 + 1.02523 w_k1_INTC_s004
                                 + 1.06357 w_k1_KO_s004
                                 + 0.681406 w_k1_MAT_s004
                                 + 0.899436 w_k1_MSFT_s004
                                 + 0.798876 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.28743 w_k1_ADM_s004
                                 + 1.60354 w_k1_AXP_s004
                                 + 1.70492 w_k1_DIS_s004
                                 + 1.27477 w_k1_FDX_s004
                                 + 1.39069 w_k1_IBM_s004
                                 + 1.23839 w_k1_INTC_s004
                                 + 1.75589 w_k1_KO_s004 + 2.0859 w_k1_MAT_s004
                                 + 1.30429 w_k1_MSFT_s004
                                 + 1.18264 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.19246 w_k1_ADM_s004
                                 + 0.700405 w_k1_AXP_s004
                                 + 0.59623 w_k1_DIS_s004
                                 + 0.676112 w_k1_FDX_s004
                                 + 0.86406 w_k1_IBM_s004
                                 + 0.770283 w_k1_INTC_s004
                                 + 0.82201 w_k1_KO_s004
                                 + 0.745164 w_k1_MAT_s004
                                 + 0.758248 w_k1_MSFT_s004
                                 + 0.846284 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.21391 w_k1_ADM_s004
                                 + 1.12293 w_k1_AXP_s004
                                 + 1.07431 w_k1_DIS_s004
                                 + 0.931341 w_k1_FDX_s004
                                 + 1.41657 w_k1_IBM_s004
                                 + 0.972088 w_k1_INTC_s004
                                 + 1.1474 w_k1_KO_s004 + 1.42054 w_k1_MAT_s004
                                 + 0.911529 w_k1_MSFT_s004
                                 + 1.00754 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.33601 w_k1_ADM_s004
                                 + 1.13792 w_k1_AXP_s004
                                 + 1.02409 w_k1_DIS_s004
                                 + 1.09257 w_k1_FDX_s004
                                 + 0.677017 w_k1_IBM_s004
                                 + 0.9425 w_k1_INTC_s004
                                 + 0.894064 w_k1_KO_s004
                                 + 0.666299 w_k1_MAT_s004
                                 + 0.951847 w_k1_MSFT_s004
                                 + 0.875426 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.12558 w_k1_ADM_s004
                                 + 0.677275 w_k1_AXP_s004
                                 + 0.572125 w_k1_DIS_s004
                                 + 0.722977 w_k1_FDX_s004
                                 + 0.675634 w_k1_IBM_s004
                                 + 0.783914 w_k1_INTC_s004
                                 + 0.75784 w_k1_KO_s004
                                 + 0.467279 w_k1_MAT_s004
                                 + 0.786514 w_k1_MSFT_s004
                                 + 0.803586 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.00131 w_k1_ADM_s004
                                 + 0.904778 w_k1_AXP_s004
                                 + 0.891889 w_k1_DIS_s004
                                 + 0.822888 w_k1_FDX_s004
                                 + 0.98946 w_k1_IBM_s004
                                 + 0.902726 w_k1_INTC_s004
                                 + 1.08343 w_k1_KO_s004 + 1.06065 w_k1_MAT_s004
                                 + 0.818923 w_k1_MSFT_s004
                                 + 0.878239 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.35814 w_k1_ADM_s004
                                 + 1.28139 w_k1_AXP_s004
                                 + 1.28575 w_k1_DIS_s004
                                 + 1.11179 w_k1_FDX_s004
                                 + 1.06138 w_k1_IBM_s004
                                 + 1.18265 w_k1_INTC_s004 + 1.1968 w_k1_KO_s004
                                 + 1.11335 w_k1_MAT_s004
                                 + 1.08958 w_k1_MSFT_s004
                                 + 1.01072 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.20112 w_k1_ADM_s005
                                 + 1.11014 w_k1_AXP_s005
                                 + 1.05773 w_k1_DIS_s005
                                 + 1.11341 w_k1_FDX_s005 + 1.0711 w_k1_IBM_s005
                                 + 0.969841 w_k1_INTC_s005
                                 + 1.13172 w_k1_KO_s005 + 1.17236 w_k1_MAT_s005
                                 + 1.07435 w_k1_MSFT_s005
                                 + 1.02136 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.912527 w_k1_ADM_s005
                                 + 0.892745 w_k1_AXP_s005
                                 + 0.792103 w_k1_DIS_s005
                                 + 0.902632 w_k1_FDX_s005
                                 + 1.08875 w_k1_IBM_s005
                                 + 0.827711 w_k1_INTC_s005
                                 + 0.892818 w_k1_KO_s005
                                 + 0.923984 w_k1_MAT_s005
                                 + 0.878685 w_k1_MSFT_s005
                                 + 0.888197 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.36321 w_k1_ADM_s005
                                 + 1.15374 w_k1_AXP_s005
                                 + 1.25736 w_k1_DIS_s005
                                 + 1.08171 w_k1_FDX_s005
                                 + 1.68722 w_k1_IBM_s005
                                 + 1.05173 w_k1_INTC_s005
                                 + 1.10957 w_k1_KO_s005
                                 + 0.954262 w_k1_MAT_s005
                                 + 1.11021 w_k1_MSFT_s005
                                 + 0.935706 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.784199 w_k1_ADM_s005
                                 + 0.628084 w_k1_AXP_s005
                                 + 0.705162 w_k1_DIS_s005
                                 + 0.786173 w_k1_FDX_s005
                                 + 0.696392 w_k1_IBM_s005
                                 + 0.896582 w_k1_INTC_s005
                                 + 0.750476 w_k1_KO_s005
                                 + 0.642123 w_k1_MAT_s005
                                 + 0.875226 w_k1_MSFT_s005
                                 + 0.750596 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.07859 w_k1_ADM_s005
                                 + 1.20749 w_k1_AXP_s005
                                 + 1.44843 w_k1_DIS_s005
                                 + 1.35137 w_k1_FDX_s005
                                 + 0.960264 w_k1_IBM_s005
                                 + 1.04198 w_k1_INTC_s005 + 1.0596 w_k1_KO_s005
                                 + 1.062 w_k1_MAT_s005 + 1.31124 w_k1_MSFT_s005
                                 + 0.964975 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.16003 w_k1_ADM_s005
                                 + 0.704963 w_k1_AXP_s005
                                 + 0.756915 w_k1_DIS_s005
                                 + 0.80337 w_k1_FDX_s005
                                 + 1.21613 w_k1_IBM_s005
                                 + 0.848092 w_k1_INTC_s005
                                 + 1.06464 w_k1_KO_s005
                                 + 0.902925 w_k1_MAT_s005
                                 + 0.920467 w_k1_MSFT_s005
                                 + 0.920196 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.04054 w_k1_ADM_s005
                                 + 0.826395 w_k1_AXP_s005
                                 + 0.785351 w_k1_DIS_s005
                                 + 0.876657 w_k1_FDX_s005
                                 + 0.791416 w_k1_IBM_s005
                                 + 0.882241 w_k1_INTC_s005
                                 + 0.852772 w_k1_KO_s005
                                 + 0.753352 w_k1_MAT_s005
                                 + 0.991453 w_k1_MSFT_s005
                                 + 0.898364 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.16976 w_k1_ADM_s005
                                 + 0.950032 w_k1_AXP_s005
                                 + 0.812712 w_k1_DIS_s005
                                 + 0.952616 w_k1_FDX_s005
                                 + 0.627733 w_k1_IBM_s005
                                 + 0.964202 w_k1_INTC_s005
                                 + 0.845534 w_k1_KO_s005
                                 + 0.660317 w_k1_MAT_s005
                                 + 0.852087 w_k1_MSFT_s005
                                 + 0.941332 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.21489 w_k1_ADM_s006
                                 + 1.49201 w_k1_AXP_s006 + 1.3139 w_k1_DIS_s006
                                 + 1.2229 w_k1_FDX_s006 + 1.25799 w_k1_IBM_s006
                                 + 1.23904 w_k1_INTC_s006
                                 + 1.38114 w_k1_KO_s006 + 1.3185 w_k1_MAT_s006
                                 + 1.08349 w_k1_MSFT_s006
                                 + 1.02281 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.755643 w_k1_ADM_s006
                                 + 0.72384 w_k1_AXP_s006
                                 + 0.67741 w_k1_DIS_s006
                                 + 0.734667 w_k1_FDX_s006
                                 + 0.776683 w_k1_IBM_s006
                                 + 0.921568 w_k1_INTC_s006
                                 + 0.747971 w_k1_KO_s006
                                 + 0.570061 w_k1_MAT_s006
                                 + 0.734239 w_k1_MSFT_s006
                                 + 0.68437 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.3239 w_k1_ADM_s006
                                 + 1.15567 w_k1_AXP_s006
                                 + 1.02519 w_k1_DIS_s006
                                 + 1.11639 w_k1_FDX_s006
                                 + 0.938721 w_k1_IBM_s006
                                 + 1.29176 w_k1_INTC_s006
                                 + 1.22543 w_k1_KO_s006 + 1.12378 w_k1_MAT_s006
                                 + 1.23006 w_k1_MSFT_s006
                                 + 1.05173 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.25226 w_k1_ADM_s006
                                 + 1.49579 w_k1_AXP_s006 + 1.5535 w_k1_DIS_s006
                                 + 1.34967 w_k1_FDX_s006
                                 + 1.22924 w_k1_IBM_s006
                                 + 1.53117 w_k1_INTC_s006
                                 + 1.40454 w_k1_KO_s006 + 1.11819 w_k1_MAT_s006
                                 + 1.28323 w_k1_MSFT_s006
                                 + 1.17416 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.838927 w_k1_ADM_s006
                                 + 1.13438 w_k1_AXP_s006
                                 + 0.858269 w_k1_DIS_s006
                                 + 0.911762 w_k1_FDX_s006
                                 + 0.778855 w_k1_IBM_s006
                                 + 1.15019 w_k1_INTC_s006
                                 + 0.871193 w_k1_KO_s006
                                 + 0.777339 w_k1_MAT_s006
                                 + 1.04113 w_k1_MSFT_s006
                                 + 0.947563 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.925251 w_k1_ADM_s006
                                 + 0.875423 w_k1_AXP_s006
                                 + 0.779621 w_k1_DIS_s006
                                 + 0.78752 w_k1_FDX_s006
                                 + 1.26502 w_k1_IBM_s006
                                 + 1.10648 w_k1_INTC_s006
                                 + 1.11838 w_k1_KO_s006 + 1.00401 w_k1_MAT_s006
                                 + 0.934749 w_k1_MSFT_s006
                                 + 0.744094 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.10831 w_k1_ADM_s006
                                 + 1.35061 w_k1_AXP_s006
                                 + 0.953403 w_k1_DIS_s006
                                 + 0.905895 w_k1_FDX_s006
                                 + 1.41682 w_k1_IBM_s006
                                 + 1.10105 w_k1_INTC_s006
                                 + 1.12997 w_k1_KO_s006 + 1.26932 w_k1_MAT_s006
                                 + 0.970285 w_k1_MSFT_s006
                                 + 0.943512 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.916161 w_k1_ADM_s006
                                 + 1.06702 w_k1_AXP_s006
                                 + 1.20254 w_k1_DIS_s006
                                 + 1.05286 w_k1_FDX_s006
                                 + 0.919577 w_k1_IBM_s006
                                 + 1.53063 w_k1_INTC_s006
                                 + 1.09814 w_k1_KO_s006
                                 + 0.671122 w_k1_MAT_s006
                                 + 1.26798 w_k1_MSFT_s006
                                 + 0.964157 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.03682 w_k1_ADM_s007
                                 + 0.901635 w_k1_AXP_s007
                                 + 0.88665 w_k1_DIS_s007
                                 + 1.02088 w_k1_FDX_s007
                                 + 0.856936 w_k1_IBM_s007
                                 + 0.924207 w_k1_INTC_s007
                                 + 0.846154 w_k1_KO_s007
                                 + 0.815862 w_k1_MAT_s007
                                 + 1.03555 w_k1_MSFT_s007
                                 + 0.897621 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.712269 w_k1_ADM_s007
                                 + 0.898309 w_k1_AXP_s007
                                 + 0.779563 w_k1_DIS_s007
                                 + 0.813512 w_k1_FDX_s007
                                 + 1.20164 w_k1_IBM_s007
                                 + 0.906309 w_k1_INTC_s007
                                 + 1.00093 w_k1_KO_s007 + 1.50352 w_k1_MAT_s007
                                 + 0.928246 w_k1_MSFT_s007
                                 + 0.810776 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.943869 w_k1_ADM_s007
                                 + 0.936855 w_k1_AXP_s007
                                 + 1.06789 w_k1_DIS_s007
                                 + 0.992889 w_k1_FDX_s007
                                 + 0.930032 w_k1_IBM_s007
                                 + 0.830349 w_k1_INTC_s007
                                 + 0.931428 w_k1_KO_s007
                                 + 1.03792 w_k1_MAT_s007
                                 + 1.17521 w_k1_MSFT_s007
                                 + 0.903041 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.15647 w_k1_ADM_s007
                                 + 1.24558 w_k1_AXP_s007
                                 + 1.29404 w_k1_DIS_s007
                                 + 1.25237 w_k1_FDX_s007
                                 + 1.22634 w_k1_IBM_s007
                                 + 1.01281 w_k1_INTC_s007
                                 + 1.13683 w_k1_KO_s007 + 1.31874 w_k1_MAT_s007
                                 + 1.34708 w_k1_MSFT_s007
                                 + 1.12107 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.547041 w_k1_ADM_s007
                                 + 0.727181 w_k1_AXP_s007
                                 + 0.681907 w_k1_DIS_s007
                                 + 0.776083 w_k1_FDX_s007
                                 + 0.878472 w_k1_IBM_s007
                                 + 0.749891 w_k1_INTC_s007
                                 + 0.735021 w_k1_KO_s007
                                 + 0.978316 w_k1_MAT_s007
                                 + 0.843695 w_k1_MSFT_s007
                                 + 0.73284 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.704115 w_k1_ADM_s007
                                 + 0.749631 w_k1_AXP_s007
                                 + 0.644311 w_k1_DIS_s007
                                 + 0.842938 w_k1_FDX_s007
                                 + 0.636642 w_k1_IBM_s007
                                 + 0.870019 w_k1_INTC_s007
                                 + 0.742131 w_k1_KO_s007
                                 + 0.69928 w_k1_MAT_s007
                                 + 0.906695 w_k1_MSFT_s007
                                 + 0.770265 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.03365 w_k1_ADM_s007
                                 + 0.896164 w_k1_AXP_s007
                                 + 1.13139 w_k1_DIS_s007
                                 + 1.02799 w_k1_FDX_s007
                                 + 1.26116 w_k1_IBM_s007
                                 + 0.827049 w_k1_INTC_s007
                                 + 1.0476 w_k1_KO_s007 + 1.60931 w_k1_MAT_s007
                                 + 1.04577 w_k1_MSFT_s007
                                 + 1.01552 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.851999 w_k1_ADM_s007
                                 + 0.933384 w_k1_AXP_s007
                                 + 1.30518 w_k1_DIS_s007
                                 + 1.09663 w_k1_FDX_s007
                                 + 0.878284 w_k1_IBM_s007
                                 + 0.912679 w_k1_INTC_s007
                                 + 0.929569 w_k1_KO_s007
                                 + 0.998601 w_k1_MAT_s007
                                 + 1.17155 w_k1_MSFT_s007
                                 + 1.05265 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.715009 w_k1_ADM_s010
                                 + 0.802644 w_k1_AXP_s010
                                 + 0.619108 w_k1_DIS_s010
                                 + 0.813362 w_k1_FDX_s010
                                 + 0.71466 w_k1_IBM_s010
                                 + 0.937392 w_k1_INTC_s010
                                 + 0.854148 w_k1_KO_s010
                                 + 0.737301 w_k1_MAT_s010
                                 + 0.825678 w_k1_MSFT_s010
                                 + 0.865549 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.96427 w_k1_ADM_s010
                                 + 0.930093 w_k1_AXP_s010
                                 + 0.780016 w_k1_DIS_s010
                                 + 0.769524 w_k1_FDX_s010
                                 + 1.21381 w_k1_IBM_s010
                                 + 0.765128 w_k1_INTC_s010
                                 + 0.946192 w_k1_KO_s010
                                 + 1.18094 w_k1_MAT_s010
                                 + 0.729403 w_k1_MSFT_s010
                                 + 0.992083 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.38165 w_k1_ADM_s010
                                 + 1.38405 w_k1_AXP_s010
                                 + 1.27358 w_k1_DIS_s010
                                 + 1.35247 w_k1_FDX_s010
                                 + 1.03791 w_k1_IBM_s010
                                 + 1.13969 w_k1_INTC_s010
                                 + 1.05534 w_k1_KO_s010
                                 + 0.795434 w_k1_MAT_s010
                                 + 1.15406 w_k1_MSFT_s010
                                 + 1.12661 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.25714 w_k1_ADM_s010
                                 + 1.10244 w_k1_AXP_s010
                                 + 1.22496 w_k1_DIS_s010
                                 + 1.11404 w_k1_FDX_s010
                                 + 0.996169 w_k1_IBM_s010
                                 + 1.01573 w_k1_INTC_s010
                                 + 1.10486 w_k1_KO_s010
                                 + 0.901624 w_k1_MAT_s010
                                 + 0.957803 w_k1_MSFT_s010
                                 + 1.0696 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.22051 w_k1_ADM_s010
                                 + 0.841599 w_k1_AXP_s010
                                 + 1.22223 w_k1_DIS_s010
                                 + 0.974205 w_k1_FDX_s010
                                 + 1.17357 w_k1_IBM_s010
                                 + 1.1816 w_k1_INTC_s010 + 1.22051 w_k1_KO_s010
                                 + 0.759831 w_k1_MAT_s010
                                 + 1.03252 w_k1_MSFT_s010
                                 + 1.08154 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.43108 w_k1_ADM_s010
                                 + 1.24336 w_k1_AXP_s010
                                 + 1.25907 w_k1_DIS_s010
                                 + 1.07122 w_k1_FDX_s010
                                 + 1.38664 w_k1_IBM_s010
                                 + 1.05034 w_k1_INTC_s010
                                 + 1.29069 w_k1_KO_s010
                                 + 0.998139 w_k1_MAT_s010
                                 + 1.06929 w_k1_MSFT_s010
                                 + 1.22465 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.61919 w_k1_ADM_s010
                                 + 1.48681 w_k1_AXP_s010
                                 + 1.83272 w_k1_DIS_s010
                                 + 1.32618 w_k1_FDX_s010
                                 + 1.99673 w_k1_IBM_s010
                                 + 1.35834 w_k1_INTC_s010
                                 + 1.75937 w_k1_KO_s010 + 1.53384 w_k1_MAT_s010
                                 + 1.28464 w_k1_MSFT_s010
                                 + 1.45041 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.910171 w_k1_ADM_s010
                                 + 0.969555 w_k1_AXP_s010
                                 + 0.898445 w_k1_DIS_s010
                                 + 0.944804 w_k1_FDX_s010
                                 + 0.853501 w_k1_IBM_s010
                                 + 0.958919 w_k1_INTC_s010
                                 + 0.923608 w_k1_KO_s010
                                 + 0.99265 w_k1_MAT_s010
                                 + 0.970904 w_k1_MSFT_s010
                                 + 0.995416 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.21412 w_k1_ADM_s011
                                 + 1.07895 w_k1_AXP_s011
                                 + 1.41878 w_k1_DIS_s011
                                 + 1.11854 w_k1_FDX_s011
                                 + 1.27366 w_k1_IBM_s011
                                 + 0.96392 w_k1_INTC_s011
                                 + 1.50621 w_k1_KO_s011 + 1.76302 w_k1_MAT_s011
                                 + 1.06549 w_k1_MSFT_s011 + 1.2129 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.979787 w_k1_ADM_s011
                                 + 1.14792 w_k1_AXP_s011
                                 + 1.22641 w_k1_DIS_s011
                                 + 0.962887 w_k1_FDX_s011
                                 + 1.53058 w_k1_IBM_s011
                                 + 0.876937 w_k1_INTC_s011
                                 + 1.31868 w_k1_KO_s011 + 1.82516 w_k1_MAT_s011
                                 + 0.949833 w_k1_MSFT_s011
                                 + 1.09033 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.10617 w_k1_ADM_s011
                                 + 1.27336 w_k1_AXP_s011
                                 + 0.932716 w_k1_DIS_s011
                                 + 0.969092 w_k1_FDX_s011
                                 + 1.10082 w_k1_IBM_s011
                                 + 0.98409 w_k1_INTC_s011
                                 + 0.995228 w_k1_KO_s011
                                 + 1.22887 w_k1_MAT_s011
                                 + 0.898819 w_k1_MSFT_s011
                                 + 0.922978 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.985234 w_k1_ADM_s011
                                 + 0.93647 w_k1_AXP_s011
                                 + 0.794254 w_k1_DIS_s011
                                 + 0.914692 w_k1_FDX_s011
                                 + 0.750257 w_k1_IBM_s011
                                 + 0.985226 w_k1_INTC_s011
                                 + 0.856282 w_k1_KO_s011
                                 + 0.763334 w_k1_MAT_s011
                                 + 0.866767 w_k1_MSFT_s011
                                 + 0.98366 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.33269 w_k1_ADM_s011
                                 + 0.998756 w_k1_AXP_s011
                                 + 1.18379 w_k1_DIS_s011
                                 + 1.07499 w_k1_FDX_s011
                                 + 1.02128 w_k1_IBM_s011
                                 + 0.986785 w_k1_INTC_s011
                                 + 1.10803 w_k1_KO_s011
                                 + 0.945512 w_k1_MAT_s011
                                 + 1.14048 w_k1_MSFT_s011
                                 + 1.10098 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.907543 w_k1_ADM_s011
                                 + 0.733024 w_k1_AXP_s011
                                 + 0.603065 w_k1_DIS_s011
                                 + 0.75059 w_k1_FDX_s011
                                 + 0.91204 w_k1_IBM_s011
                                 + 0.844399 w_k1_INTC_s011
                                 + 0.856397 w_k1_KO_s011
                                 + 0.734738 w_k1_MAT_s011
                                 + 0.823977 w_k1_MSFT_s011
                                 + 0.797411 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.47466 w_k1_ADM_s011
                                 + 1.10157 w_k1_AXP_s011
                                 + 1.12366 w_k1_DIS_s011 + 1.1166 w_k1_FDX_s011
                                 + 0.781829 w_k1_IBM_s011
                                 + 1.07544 w_k1_INTC_s011
                                 + 1.05653 w_k1_KO_s011
                                 + 0.588205 w_k1_MAT_s011
                                 + 0.997826 w_k1_MSFT_s011
                                 + 1.10476 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.43646 w_k1_ADM_s011
                                 + 1.38888 w_k1_AXP_s011
                                 + 1.85896 w_k1_DIS_s011
                                 + 1.35057 w_k1_FDX_s011
                                 + 1.12399 w_k1_IBM_s011 + 1.396 w_k1_INTC_s011
                                 + 1.46308 w_k1_KO_s011 + 1.06683 w_k1_MAT_s011
                                 + 1.43369 w_k1_MSFT_s011
                                 + 1.20836 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.00292 w_k1_ADM_s012
                                 + 1.17235 w_k1_AXP_s012
                                 + 1.01744 w_k1_DIS_s012
                                 + 1.08343 w_k1_FDX_s012
                                 + 1.01376 w_k1_IBM_s012
                                 + 1.18501 w_k1_INTC_s012
                                 + 1.00427 w_k1_KO_s012 + 1.01424 w_k1_MAT_s012
                                 + 0.914944 w_k1_MSFT_s012
                                 + 1.01208 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.739078 w_k1_ADM_s012
                                 + 0.775601 w_k1_AXP_s012
                                 + 0.800691 w_k1_DIS_s012
                                 + 0.848465 w_k1_FDX_s012
                                 + 0.666941 w_k1_IBM_s012
                                 + 0.873931 w_k1_INTC_s012
                                 + 0.791644 w_k1_KO_s012
                                 + 0.51479 w_k1_MAT_s012
                                 + 0.821587 w_k1_MSFT_s012
                                 + 0.874037 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.776204 w_k1_ADM_s012
                                 + 1.46749 w_k1_AXP_s012
                                 + 1.17256 w_k1_DIS_s012
                                 + 1.00331 w_k1_FDX_s012
                                 + 1.69209 w_k1_IBM_s012
                                 + 1.08907 w_k1_INTC_s012 + 1.5793 w_k1_KO_s012
                                 + 2.14366 w_k1_MAT_s012
                                 + 1.17372 w_k1_MSFT_s012
                                 + 1.25061 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.05494 w_k1_ADM_s012
                                 + 1.1489 w_k1_AXP_s012 + 1.51295 w_k1_DIS_s012
                                 + 1.17184 w_k1_FDX_s012
                                 + 0.890041 w_k1_IBM_s012
                                 + 1.08842 w_k1_INTC_s012
                                 + 1.25037 w_k1_KO_s012 + 1.06444 w_k1_MAT_s012
                                 + 1.21611 w_k1_MSFT_s012
                                 + 1.16125 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.644529 w_k1_ADM_s012
                                 + 0.746064 w_k1_AXP_s012
                                 + 0.85386 w_k1_DIS_s012
                                 + 0.840453 w_k1_FDX_s012
                                 + 0.915111 w_k1_IBM_s012
                                 + 0.726489 w_k1_INTC_s012
                                 + 0.967477 w_k1_KO_s012
                                 + 0.987844 w_k1_MAT_s012
                                 + 0.788214 w_k1_MSFT_s012
                                 + 0.922405 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.743983 w_k1_ADM_s012
                                 + 0.810998 w_k1_AXP_s012
                                 + 0.839336 w_k1_DIS_s012
                                 + 0.817354 w_k1_FDX_s012
                                 + 1.25987 w_k1_IBM_s012
                                 + 0.934572 w_k1_INTC_s012
                                 + 1.11949 w_k1_KO_s012 + 1.05465 w_k1_MAT_s012
                                 + 0.851032 w_k1_MSFT_s012
                                 + 1.04297 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.769022 w_k1_ADM_s012
                                 + 0.91074 w_k1_AXP_s012
                                 + 1.36652 w_k1_DIS_s012
                                 + 1.07707 w_k1_FDX_s012
                                 + 1.29009 w_k1_IBM_s012
                                 + 1.0328 w_k1_INTC_s012 + 1.28412 w_k1_KO_s012
                                 + 1.08683 w_k1_MAT_s012
                                 + 1.04419 w_k1_MSFT_s012
                                 + 1.08391 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.10642 w_k1_ADM_s012
                                 + 1.38626 w_k1_AXP_s012
                                 + 1.58197 w_k1_DIS_s012
                                 + 1.34384 w_k1_FDX_s012
                                 + 1.08089 w_k1_IBM_s012
                                 + 1.16035 w_k1_INTC_s012
                                 + 1.51318 w_k1_KO_s012 + 1.37694 w_k1_MAT_s012
                                 + 1.24891 w_k1_MSFT_s012
                                 + 1.35044 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.21149 w_k1_ADM_s013
                                 + 1.36443 w_k1_AXP_s013
                                 + 1.43999 w_k1_DIS_s013
                                 + 1.35752 w_k1_FDX_s013
                                 + 1.30366 w_k1_IBM_s013
                                 + 1.19752 w_k1_INTC_s013
                                 + 1.31452 w_k1_KO_s013 + 1.15543 w_k1_MAT_s013
                                 + 1.24547 w_k1_MSFT_s013
                                 + 1.18282 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.912802 w_k1_ADM_s013
                                 + 0.905187 w_k1_AXP_s013
                                 + 0.696776 w_k1_DIS_s013
                                 + 0.831327 w_k1_FDX_s013
                                 + 0.886775 w_k1_IBM_s013
                                 + 0.862221 w_k1_INTC_s013
                                 + 0.700998 w_k1_KO_s013
                                 + 0.617007 w_k1_MAT_s013
                                 + 0.869792 w_k1_MSFT_s013
                                 + 0.885436 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.09432 w_k1_ADM_s013
                                 + 1.27059 w_k1_AXP_s013
                                 + 1.10987 w_k1_DIS_s013
                                 + 1.04772 w_k1_FDX_s013 + 1.3291 w_k1_IBM_s013
                                 + 1.0296 w_k1_INTC_s013
                                 + 0.968534 w_k1_KO_s013
                                 + 1.37968 w_k1_MAT_s013
                                 + 1.02832 w_k1_MSFT_s013
                                 + 1.08025 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.11141 w_k1_ADM_s013
                                 + 0.922011 w_k1_AXP_s013
                                 + 0.706554 w_k1_DIS_s013
                                 + 0.971955 w_k1_FDX_s013
                                 + 0.626238 w_k1_IBM_s013
                                 + 0.873531 w_k1_INTC_s013
                                 + 0.804858 w_k1_KO_s013
                                 + 0.609002 w_k1_MAT_s013
                                 + 0.898341 w_k1_MSFT_s013
                                 + 0.990696 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.893939 w_k1_ADM_s013
                                 + 0.771792 w_k1_AXP_s013
                                 + 0.771468 w_k1_DIS_s013
                                 + 0.781946 w_k1_FDX_s013
                                 + 1.30565 w_k1_IBM_s013
                                 + 0.832313 w_k1_INTC_s013
                                 + 0.968501 w_k1_KO_s013
                                 + 1.09586 w_k1_MAT_s013
                                 + 0.867618 w_k1_MSFT_s013
                                 + 1.00991 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.09789 w_k1_ADM_s013
                                 + 0.913477 w_k1_AXP_s013
                                 + 0.975515 w_k1_DIS_s013
                                 + 0.9479 w_k1_FDX_s013 + 1.17653 w_k1_IBM_s013
                                 + 0.975553 w_k1_INTC_s013
                                 + 1.1521 w_k1_KO_s013 + 0.843844 w_k1_MAT_s013
                                 + 0.949209 w_k1_MSFT_s013
                                 + 1.08503 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.09196 w_k1_ADM_s013
                                 + 0.950434 w_k1_AXP_s013
                                 + 1.08305 w_k1_DIS_s013
                                 + 1.08462 w_k1_FDX_s013
                                 + 0.817218 w_k1_IBM_s013
                                 + 1.0577 w_k1_INTC_s013 + 1.02773 w_k1_KO_s013
                                 + 0.721014 w_k1_MAT_s013
                                 + 1.06986 w_k1_MSFT_s013
                                 + 1.10398 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.808888 w_k1_ADM_s013
                                 + 0.730183 w_k1_AXP_s013
                                 + 0.512869 w_k1_DIS_s013
                                 + 0.68465 w_k1_FDX_s013
                                 + 0.741364 w_k1_IBM_s013
                                 + 0.741622 w_k1_INTC_s013
                                 + 0.618275 w_k1_KO_s013
                                 + 0.578584 w_k1_MAT_s013
                                 + 0.698335 w_k1_MSFT_s013
                                 + 0.690289 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.963456 w_k1_ADM_s014
                                 + 1.23037 w_k1_AXP_s014
                                 + 1.03509 w_k1_DIS_s014
                                 + 0.994056 w_k1_FDX_s014
                                 + 1.15849 w_k1_IBM_s014
                                 + 1.10427 w_k1_INTC_s014
                                 + 1.11058 w_k1_KO_s014 + 2.01845 w_k1_MAT_s014
                                 + 1.0514 w_k1_MSFT_s014 + 1.02641 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.09739 w_k1_ADM_s014
                                 + 0.997664 w_k1_AXP_s014
                                 + 0.883648 w_k1_DIS_s014
                                 + 1.04353 w_k1_FDX_s014
                                 + 0.756037 w_k1_IBM_s014
                                 + 0.852748 w_k1_INTC_s014
                                 + 0.778772 w_k1_KO_s014
                                 + 0.919356 w_k1_MAT_s014
                                 + 0.904156 w_k1_MSFT_s014
                                 + 0.97597 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.947662 w_k1_ADM_s014
                                 + 0.775968 w_k1_AXP_s014
                                 + 0.874421 w_k1_DIS_s014
                                 + 0.824839 w_k1_FDX_s014
                                 + 1.02882 w_k1_IBM_s014
                                 + 0.87574 w_k1_INTC_s014
                                 + 1.05541 w_k1_KO_s014 + 1.16218 w_k1_MAT_s014
                                 + 0.905117 w_k1_MSFT_s014
                                 + 0.962959 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.23628 w_k1_ADM_s014
                                 + 1.60919 w_k1_AXP_s014
                                 + 1.36447 w_k1_DIS_s014
                                 + 1.35551 w_k1_FDX_s014
                                 + 1.15513 w_k1_IBM_s014
                                 + 1.3231 w_k1_INTC_s014 + 1.24224 w_k1_KO_s014
                                 + 1.92538 w_k1_MAT_s014
                                 + 1.12601 w_k1_MSFT_s014
                                 + 1.09112 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.03843 w_k1_ADM_s014
                                 + 1.40876 w_k1_AXP_s014
                                 + 1.29364 w_k1_DIS_s014
                                 + 1.09871 w_k1_FDX_s014
                                 + 1.41262 w_k1_IBM_s014
                                 + 1.15948 w_k1_INTC_s014
                                 + 1.29893 w_k1_KO_s014 + 1.85773 w_k1_MAT_s014
                                 + 1.08484 w_k1_MSFT_s014 + 1.1343 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.29145 w_k1_ADM_s014
                                 + 0.947826 w_k1_AXP_s014
                                 + 1.66814 w_k1_DIS_s014
                                 + 1.06225 w_k1_FDX_s014
                                 + 1.49557 w_k1_IBM_s014
                                 + 1.47134 w_k1_INTC_s014
                                 + 1.57023 w_k1_KO_s014 + 1.32372 w_k1_MAT_s014
                                 + 1.38557 w_k1_MSFT_s014
                                 + 1.06294 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.13729 w_k1_ADM_s014
                                 + 0.852061 w_k1_AXP_s014
                                 + 1.30673 w_k1_DIS_s014
                                 + 1.11097 w_k1_FDX_s014
                                 + 1.02183 w_k1_IBM_s014
                                 + 1.31906 w_k1_INTC_s014
                                 + 1.26481 w_k1_KO_s014 + 1.15417 w_k1_MAT_s014
                                 + 1.27162 w_k1_MSFT_s014
                                 + 1.05253 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.758144 w_k1_ADM_s014
                                 + 0.57102 w_k1_AXP_s014
                                 + 0.666143 w_k1_DIS_s014
                                 + 0.774956 w_k1_FDX_s014
                                 + 0.712578 w_k1_IBM_s014
                                 + 0.883773 w_k1_INTC_s014
                                 + 0.728467 w_k1_KO_s014 + 0.7101 w_k1_MAT_s014
                                 + 0.951301 w_k1_MSFT_s014
                                 + 0.842259 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.03002 w_k1_ADM_s015
                                 + 0.919941 w_k1_AXP_s015
                                 + 0.697036 w_k1_DIS_s015
                                 + 0.797757 w_k1_FDX_s015
                                 + 0.886593 w_k1_IBM_s015
                                 + 0.829352 w_k1_INTC_s015
                                 + 0.779261 w_k1_KO_s015 + 1.3834 w_k1_MAT_s015
                                 + 0.749088 w_k1_MSFT_s015
                                 + 0.966366 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.09961 w_k1_ADM_s015
                                 + 1.10919 w_k1_AXP_s015
                                 + 1.34526 w_k1_DIS_s015
                                 + 1.10911 w_k1_FDX_s015
                                 + 1.68536 w_k1_IBM_s015
                                 + 0.967952 w_k1_INTC_s015
                                 + 1.35079 w_k1_KO_s015 + 1.8729 w_k1_MAT_s015
                                 + 1.03833 w_k1_MSFT_s015
                                 + 1.28032 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.890464 w_k1_ADM_s015
                                 + 0.907067 w_k1_AXP_s015
                                 + 0.694014 w_k1_DIS_s015
                                 + 0.900795 w_k1_FDX_s015
                                 + 0.601818 w_k1_IBM_s015
                                 + 0.66031 w_k1_INTC_s015
                                 + 0.742715 w_k1_KO_s015
                                 + 1.03404 w_k1_MAT_s015
                                 + 0.714333 w_k1_MSFT_s015
                                 + 0.901718 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.01603 w_k1_ADM_s015
                                 + 1.1209 w_k1_AXP_s015 + 1.6095 w_k1_DIS_s015
                                 + 1.24296 w_k1_FDX_s015
                                 + 1.12735 w_k1_IBM_s015
                                 + 1.11553 w_k1_INTC_s015
                                 + 1.41331 w_k1_KO_s015 + 1.70194 w_k1_MAT_s015
                                 + 1.17447 w_k1_MSFT_s015 + 1.2332 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.647183 w_k1_ADM_s015
                                 + 0.573562 w_k1_AXP_s015
                                 + 0.535615 w_k1_DIS_s015
                                 + 0.72444 w_k1_FDX_s015
                                 + 0.612112 w_k1_IBM_s015
                                 + 0.764192 w_k1_INTC_s015
                                 + 0.647015 w_k1_KO_s015
                                 + 0.66895 w_k1_MAT_s015
                                 + 0.712755 w_k1_MSFT_s015
                                 + 0.858387 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.946189 w_k1_ADM_s015
                                 + 0.765356 w_k1_AXP_s015
                                 + 0.931035 w_k1_DIS_s015
                                 + 0.859944 w_k1_FDX_s015
                                 + 1.00552 w_k1_IBM_s015
                                 + 0.904112 w_k1_INTC_s015
                                 + 0.974485 w_k1_KO_s015
                                 + 1.38089 w_k1_MAT_s015
                                 + 0.784466 w_k1_MSFT_s015
                                 + 0.98171 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.337 w_k1_ADM_s015
                                 + 1.30134 w_k1_AXP_s015
                                 + 0.959922 w_k1_DIS_s015
                                 + 1.14025 w_k1_FDX_s015
                                 + 0.902712 w_k1_IBM_s015
                                 + 0.910798 w_k1_INTC_s015
                                 + 0.888547 w_k1_KO_s015
                                 + 1.58301 w_k1_MAT_s015
                                 + 0.921223 w_k1_MSFT_s015
                                 + 1.17528 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.991633 w_k1_ADM_s015
                                 + 0.90657 w_k1_AXP_s015
                                 + 1.15776 w_k1_DIS_s015
                                 + 1.02155 w_k1_FDX_s015
                                 + 0.744005 w_k1_IBM_s015
                                 + 0.916355 w_k1_INTC_s015
                                 + 0.971012 w_k1_KO_s015
                                 + 1.04946 w_k1_MAT_s015
                                 + 0.949318 w_k1_MSFT_s015
                                 + 1.15389 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.08292 w_k1_ADM_s016
                                 + 1.27764 w_k1_AXP_s016
                                 + 1.23183 w_k1_DIS_s016
                                 + 1.23318 w_k1_FDX_s016
                                 + 1.33363 w_k1_IBM_s016
                                 + 1.06239 w_k1_INTC_s016
                                 + 1.34795 w_k1_KO_s016 + 1.64739 w_k1_MAT_s016
                                 + 1.05198 w_k1_MSFT_s016
                                 + 1.33014 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02131 w_k1_ADM_s016
                                 + 1.1096 w_k1_AXP_s016
                                 + 0.921172 w_k1_DIS_s016
                                 + 0.887312 w_k1_FDX_s016
                                 + 1.35407 w_k1_IBM_s016
                                 + 0.925206 w_k1_INTC_s016
                                 + 1.00402 w_k1_KO_s016 + 1.32187 w_k1_MAT_s016
                                 + 0.872462 w_k1_MSFT_s016
                                 + 0.985482 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.03629 w_k1_ADM_s016
                                 + 1.20592 w_k1_AXP_s016
                                 + 0.928696 w_k1_DIS_s016
                                 + 0.996143 w_k1_FDX_s016
                                 + 1.02113 w_k1_IBM_s016
                                 + 1.00859 w_k1_INTC_s016
                                 + 1.09584 w_k1_KO_s016 + 1.02154 w_k1_MAT_s016
                                 + 0.91248 w_k1_MSFT_s016
                                 + 1.08021 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.887881 w_k1_ADM_s016
                                 + 0.703004 w_k1_AXP_s016
                                 + 0.872899 w_k1_DIS_s016
                                 + 0.955011 w_k1_FDX_s016
                                 + 0.723801 w_k1_IBM_s016
                                 + 1.03295 w_k1_INTC_s016
                                 + 0.920741 w_k1_KO_s016
                                 + 0.526021 w_k1_MAT_s016
                                 + 1.00498 w_k1_MSFT_s016
                                 + 0.96867 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.667734 w_k1_ADM_s016
                                 + 0.742569 w_k1_AXP_s016
                                 + 0.638461 w_k1_DIS_s016
                                 + 0.826984 w_k1_FDX_s016
                                 + 0.812369 w_k1_IBM_s016
                                 + 1.02458 w_k1_INTC_s016
                                 + 0.682793 w_k1_KO_s016
                                 + 0.773757 w_k1_MAT_s016
                                 + 0.720971 w_k1_MSFT_s016
                                 + 0.800833 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.1521 w_k1_ADM_s016
                                 + 0.911569 w_k1_AXP_s016
                                 + 1.23177 w_k1_DIS_s016
                                 + 1.17949 w_k1_FDX_s016
                                 + 0.891353 w_k1_IBM_s016
                                 + 1.30828 w_k1_INTC_s016
                                 + 1.16817 w_k1_KO_s016
                                 + 0.903912 w_k1_MAT_s016
                                 + 1.17947 w_k1_MSFT_s016
                                 + 1.23295 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.09443 w_k1_ADM_s016
                                 + 1.00466 w_k1_AXP_s016
                                 + 0.855992 w_k1_DIS_s016
                                 + 1.1509 w_k1_FDX_s016
                                 + 0.798318 w_k1_IBM_s016
                                 + 1.14101 w_k1_INTC_s016
                                 + 0.757824 w_k1_KO_s016
                                 + 0.589132 w_k1_MAT_s016
                                 + 0.909335 w_k1_MSFT_s016
                                 + 1.06144 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.34405 w_k1_ADM_s016
                                 + 1.60898 w_k1_AXP_s016
                                 + 1.72022 w_k1_DIS_s016 + 1.4515 w_k1_FDX_s016
                                 + 1.21526 w_k1_IBM_s016
                                 + 1.30644 w_k1_INTC_s016
                                 + 1.34741 w_k1_KO_s016 + 1.02886 w_k1_MAT_s016
                                 + 1.29919 w_k1_MSFT_s016
                                 + 1.25466 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.914233 w_k1_ADM_s017
                                 + 0.709395 w_k1_AXP_s017
                                 + 0.610948 w_k1_DIS_s017
                                 + 0.702728 w_k1_FDX_s017
                                 + 0.94129 w_k1_IBM_s017
                                 + 0.738149 w_k1_INTC_s017
                                 + 0.74377 w_k1_KO_s017 + 1.07396 w_k1_MAT_s017
                                 + 0.758062 w_k1_MSFT_s017
                                 + 0.925686 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.61024 w_k1_ADM_s017
                                 + 1.19308 w_k1_AXP_s017
                                 + 1.56314 w_k1_DIS_s017
                                 + 1.32138 w_k1_FDX_s017
                                 + 1.21848 w_k1_IBM_s017
                                 + 1.13953 w_k1_INTC_s017
                                 + 1.04253 w_k1_KO_s017 + 1.20213 w_k1_MAT_s017
                                 + 1.34984 w_k1_MSFT_s017
                                 + 1.23244 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.936711 w_k1_ADM_s017
                                 + 1.01938 w_k1_AXP_s017
                                 + 0.806847 w_k1_DIS_s017
                                 + 0.824597 w_k1_FDX_s017
                                 + 1.26787 w_k1_IBM_s017
                                 + 0.951806 w_k1_INTC_s017
                                 + 0.93575 w_k1_KO_s017 + 1.56881 w_k1_MAT_s017
                                 + 0.829683 w_k1_MSFT_s017
                                 + 0.97238 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.51692 w_k1_ADM_s017
                                 + 1.50353 w_k1_AXP_s017
                                 + 1.13216 w_k1_DIS_s017
                                 + 1.30409 w_k1_FDX_s017
                                 + 1.01815 w_k1_IBM_s017
                                 + 1.02134 w_k1_INTC_s017
                                 + 0.956388 w_k1_KO_s017
                                 + 1.58932 w_k1_MAT_s017
                                 + 1.18375 w_k1_MSFT_s017 + 1.2171 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.897211 w_k1_ADM_s017
                                 + 0.727631 w_k1_AXP_s017
                                 + 0.714208 w_k1_DIS_s017
                                 + 0.866029 w_k1_FDX_s017
                                 + 0.736471 w_k1_IBM_s017
                                 + 0.924591 w_k1_INTC_s017
                                 + 0.708693 w_k1_KO_s017
                                 + 0.784898 w_k1_MAT_s017
                                 + 0.926545 w_k1_MSFT_s017
                                 + 0.881648 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.19485 w_k1_ADM_s017
                                 + 0.93708 w_k1_AXP_s017
                                 + 0.655916 w_k1_DIS_s017
                                 + 0.953199 w_k1_FDX_s017
                                 + 0.587357 w_k1_IBM_s017
                                 + 0.846065 w_k1_INTC_s017
                                 + 0.714998 w_k1_KO_s017
                                 + 0.71119 w_k1_MAT_s017
                                 + 0.930108 w_k1_MSFT_s017
                                 + 0.987559 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.29393 w_k1_ADM_s017
                                 + 1.02836 w_k1_AXP_s017
                                 + 0.959878 w_k1_DIS_s017
                                 + 1.1158 w_k1_FDX_s017 + 0.93218 w_k1_IBM_s017
                                 + 0.911549 w_k1_INTC_s017
                                 + 0.960043 w_k1_KO_s017
                                 + 1.41759 w_k1_MAT_s017
                                 + 1.00439 w_k1_MSFT_s017
                                 + 1.16499 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.14512 w_k1_ADM_s017
                                 + 0.927131 w_k1_AXP_s017
                                 + 0.945931 w_k1_DIS_s017
                                 + 0.94367 w_k1_FDX_s017
                                 + 1.14518 w_k1_IBM_s017
                                 + 0.907792 w_k1_INTC_s017
                                 + 0.906965 w_k1_KO_s017
                                 + 1.01541 w_k1_MAT_s017
                                 + 1.0188 w_k1_MSFT_s017
                                 + 0.918415 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.06207 w_k1_ADM_s100
                                 + 0.771298 w_k1_AXP_s100
                                 + 0.780642 w_k1_DIS_s100
                                 + 1.01148 w_k1_FDX_s100
                                 + 0.67243 w_k1_IBM_s100
                                 + 0.883472 w_k1_INTC_s100
                                 + 0.791223 w_k1_KO_s100
                                 + 1.00297 w_k1_MAT_s100
                                 + 0.996753 w_k1_MSFT_s100
                                 + 0.960612 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.31915 w_k1_ADM_s100
                                 + 1.64801 w_k1_AXP_s100
                                 + 1.59632 w_k1_DIS_s100
                                 + 1.36742 w_k1_FDX_s100
                                 + 1.51214 w_k1_IBM_s100
                                 + 1.20941 w_k1_INTC_s100
                                 + 1.38143 w_k1_KO_s100 + 2.71067 w_k1_MAT_s100
                                 + 1.60387 w_k1_MSFT_s100
                                 + 1.28829 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.18697 w_k1_ADM_s100
                                 + 0.960417 w_k1_AXP_s100
                                 + 1.16616 w_k1_DIS_s100
                                 + 1.02354 w_k1_FDX_s100
                                 + 1.14578 w_k1_IBM_s100
                                 + 0.803667 w_k1_INTC_s100
                                 + 1.12401 w_k1_KO_s100 + 1.70343 w_k1_MAT_s100
                                 + 1.12827 w_k1_MSFT_s100
                                 + 1.14077 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.627349 w_k1_ADM_s100
                                 + 0.842264 w_k1_AXP_s100
                                 + 0.862688 w_k1_DIS_s100
                                 + 0.995449 w_k1_FDX_s100
                                 + 0.883512 w_k1_IBM_s100
                                 + 1.02675 w_k1_INTC_s100
                                 + 1.00412 w_k1_KO_s100 + 1.45748 w_k1_MAT_s100
                                 + 1.09839 w_k1_MSFT_s100
                                 + 1.05903 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.719503 w_k1_ADM_s100
                                 + 0.535914 w_k1_AXP_s100
                                 + 0.633341 w_k1_DIS_s100
                                 + 0.708439 w_k1_FDX_s100
                                 + 0.851638 w_k1_IBM_s100
                                 + 0.703161 w_k1_INTC_s100
                                 + 0.881047 w_k1_KO_s100
                                 + 1.10475 w_k1_MAT_s100
                                 + 0.82236 w_k1_MSFT_s100
                                 + 0.833845 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.00336 w_k1_ADM_s100
                                 + 1.29102 w_k1_AXP_s100
                                 + 0.996738 w_k1_DIS_s100
                                 + 1.23195 w_k1_FDX_s100
                                 + 0.857374 w_k1_IBM_s100
                                 + 0.950113 w_k1_INTC_s100
                                 + 0.9875 w_k1_KO_s100 + 1.53586 w_k1_MAT_s100
                                 + 1.01069 w_k1_MSFT_s100
                                 + 1.07932 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.809555 w_k1_ADM_s100
                                 + 0.837597 w_k1_AXP_s100
                                 + 0.931561 w_k1_DIS_s100
                                 + 0.861278 w_k1_FDX_s100
                                 + 1.21884 w_k1_IBM_s100
                                 + 0.989748 w_k1_INTC_s100
                                 + 1.19575 w_k1_KO_s100 + 1.27413 w_k1_MAT_s100
                                 + 1.15281 w_k1_MSFT_s100
                                 + 0.945292 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.930512 w_k1_ADM_s100
                                 + 0.908029 w_k1_AXP_s100
                                 + 0.673378 w_k1_DIS_s100
                                 + 0.928049 w_k1_FDX_s100
                                 + 0.799331 w_k1_IBM_s100
                                 + 0.819858 w_k1_INTC_s100
                                 + 0.855679 w_k1_KO_s100
                                 + 0.905082 w_k1_MAT_s100
                                 + 0.840002 w_k1_MSFT_s100
                                 + 0.853592 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.05771 w_k1_ADM_s101
                                 + 0.951936 w_k1_AXP_s101
                                 + 0.962743 w_k1_DIS_s101
                                 + 1.01918 w_k1_FDX_s101
                                 + 0.847246 w_k1_IBM_s101
                                 + 0.883786 w_k1_INTC_s101
                                 + 0.796445 w_k1_KO_s101
                                 + 0.739627 w_k1_MAT_s101
                                 + 0.850439 w_k1_MSFT_s101
                                 + 0.854702 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.08149 w_k1_ADM_s101
                                 + 1.14078 w_k1_AXP_s101 + 1.2244 w_k1_DIS_s101
                                 + 1.14997 w_k1_FDX_s101
                                 + 1.32121 w_k1_IBM_s101
                                 + 1.00667 w_k1_INTC_s101
                                 + 1.17706 w_k1_KO_s101 + 1.27031 w_k1_MAT_s101
                                 + 1.12442 w_k1_MSFT_s101
                                 + 1.18339 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.01077 w_k1_ADM_s101
                                 + 1.20694 w_k1_AXP_s101 + 1.5771 w_k1_DIS_s101
                                 + 1.34007 w_k1_FDX_s101
                                 + 1.01411 w_k1_IBM_s101
                                 + 1.18095 w_k1_INTC_s101
                                 + 1.18067 w_k1_KO_s101 + 1.28409 w_k1_MAT_s101
                                 + 1.35827 w_k1_MSFT_s101 + 1.1486 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.990677 w_k1_ADM_s101
                                 + 0.741397 w_k1_AXP_s101 + 0.798 w_k1_DIS_s101
                                 + 0.789221 w_k1_FDX_s101
                                 + 0.912292 w_k1_IBM_s101
                                 + 0.741833 w_k1_INTC_s101
                                 + 0.90635 w_k1_KO_s101
                                 + 0.719884 w_k1_MAT_s101
                                 + 0.872067 w_k1_MSFT_s101
                                 + 0.980346 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.897522 w_k1_ADM_s101
                                 + 0.81 w_k1_AXP_s101 + 1.03317 w_k1_DIS_s101
                                 + 0.925663 w_k1_FDX_s101
                                 + 1.07445 w_k1_IBM_s101
                                 + 0.766572 w_k1_INTC_s101
                                 + 0.986982 w_k1_KO_s101
                                 + 0.962965 w_k1_MAT_s101
                                 + 0.890925 w_k1_MSFT_s101
                                 + 1.04021 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.1226 w_k1_ADM_s101
                                 + 0.762103 w_k1_AXP_s101
                                 + 1.01565 w_k1_DIS_s101 + 0.9639 w_k1_FDX_s101
                                 + 0.655893 w_k1_IBM_s101
                                 + 0.77963 w_k1_INTC_s101
                                 + 0.715032 w_k1_KO_s101
                                 + 0.524448 w_k1_MAT_s101
                                 + 0.873185 w_k1_MSFT_s101
                                 + 0.857763 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.760017 w_k1_ADM_s101
                                 + 0.557142 w_k1_AXP_s101
                                 + 0.653252 w_k1_DIS_s101
                                 + 0.644943 w_k1_FDX_s101
                                 + 0.812611 w_k1_IBM_s101
                                 + 0.697505 w_k1_INTC_s101
                                 + 0.787849 w_k1_KO_s101
                                 + 0.776844 w_k1_MAT_s101
                                 + 0.71098 w_k1_MSFT_s101
                                 + 0.785251 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.24213 w_k1_ADM_s101
                                 + 1.28148 w_k1_AXP_s101
                                 + 1.62824 w_k1_DIS_s101
                                 + 1.07533 w_k1_FDX_s101
                                 + 1.68251 w_k1_IBM_s101
                                 + 0.906891 w_k1_INTC_s101
                                 + 1.32058 w_k1_KO_s101 + 1.31856 w_k1_MAT_s101
                                 + 1.04704 w_k1_MSFT_s101 + 1.1512 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.832947 w_k1_ADM_s102
                                 + 0.963196 w_k1_AXP_s102
                                 + 0.712456 w_k1_DIS_s102
                                 + 0.820422 w_k1_FDX_s102
                                 + 1.08627 w_k1_IBM_s102
                                 + 0.824985 w_k1_INTC_s102
                                 + 0.921589 w_k1_KO_s102
                                 + 1.11882 w_k1_MAT_s102
                                 + 0.83169 w_k1_MSFT_s102
                                 + 0.844784 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.50384 w_k1_ADM_s102
                                 + 1.72458 w_k1_AXP_s102 + 1.9538 w_k1_DIS_s102
                                 + 1.60579 w_k1_FDX_s102
                                 + 1.15671 w_k1_IBM_s102
                                 + 1.37452 w_k1_INTC_s102
                                 + 1.38232 w_k1_KO_s102
                                 + 0.965834 w_k1_MAT_s102
                                 + 1.20571 w_k1_MSFT_s102
                                 + 1.24586 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.0012 w_k1_ADM_s102
                                 + 0.654286 w_k1_AXP_s102
                                 + 0.602732 w_k1_DIS_s102
                                 + 0.782221 w_k1_FDX_s102
                                 + 0.716559 w_k1_IBM_s102
                                 + 0.687541 w_k1_INTC_s102
                                 + 0.784628 w_k1_KO_s102
                                 + 0.860724 w_k1_MAT_s102
                                 + 0.744934 w_k1_MSFT_s102
                                 + 0.886019 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.16124 w_k1_ADM_s102
                                 + 1.26654 w_k1_AXP_s102
                                 + 1.23954 w_k1_DIS_s102
                                 + 1.20375 w_k1_FDX_s102
                                 + 1.16754 w_k1_IBM_s102
                                 + 1.27141 w_k1_INTC_s102
                                 + 1.23254 w_k1_KO_s102 + 1.32908 w_k1_MAT_s102
                                 + 1.04816 w_k1_MSFT_s102
                                 + 1.05719 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.13199 w_k1_ADM_s102
                                 + 1.18792 w_k1_AXP_s102
                                 + 1.00951 w_k1_DIS_s102
                                 + 1.00704 w_k1_FDX_s102
                                 + 1.27867 w_k1_IBM_s102
                                 + 0.986435 w_k1_INTC_s102
                                 + 1.09178 w_k1_KO_s102
                                 + 0.974686 w_k1_MAT_s102
                                 + 0.854469 w_k1_MSFT_s102
                                 + 0.932514 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.19089 w_k1_ADM_s102
                                 + 1.01048 w_k1_AXP_s102
                                 + 1.05052 w_k1_DIS_s102
                                 + 1.07192 w_k1_FDX_s102
                                 + 0.871109 w_k1_IBM_s102
                                 + 0.899451 w_k1_INTC_s102
                                 + 1.18567 w_k1_KO_s102
                                 + 0.840563 w_k1_MAT_s102
                                 + 0.915483 w_k1_MSFT_s102
                                 + 0.975033 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.23969 w_k1_ADM_s102
                                 + 1.36593 w_k1_AXP_s102
                                 + 1.27839 w_k1_DIS_s102
                                 + 1.12092 w_k1_FDX_s102
                                 + 1.62336 w_k1_IBM_s102
                                 + 1.02585 w_k1_INTC_s102
                                 + 1.50279 w_k1_KO_s102 + 1.91225 w_k1_MAT_s102
                                 + 1.0665 w_k1_MSFT_s102 + 1.07166 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.58073 w_k1_ADM_s102
                                 + 1.40101 w_k1_AXP_s102
                                 + 1.43387 w_k1_DIS_s102
                                 + 1.30528 w_k1_FDX_s102
                                 + 0.818753 w_k1_IBM_s102
                                 + 1.20441 w_k1_INTC_s102
                                 + 1.12065 w_k1_KO_s102 + 0.87538 w_k1_MAT_s102
                                 + 1.17579 w_k1_MSFT_s102
                                 + 1.04816 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.47571 w_k1_ADM_s103
                                 + 1.15255 w_k1_AXP_s103
                                 + 0.810848 w_k1_DIS_s103
                                 + 1.17666 w_k1_FDX_s103
                                 + 0.758646 w_k1_IBM_s103
                                 + 1.00617 w_k1_INTC_s103
                                 + 0.829789 w_k1_KO_s103
                                 + 0.880839 w_k1_MAT_s103
                                 + 0.901797 w_k1_MSFT_s103
                                 + 1.06962 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.13025 w_k1_ADM_s103
                                 + 0.994317 w_k1_AXP_s103
                                 + 1.12027 w_k1_DIS_s103
                                 + 1.09722 w_k1_FDX_s103
                                 + 0.952982 w_k1_IBM_s103
                                 + 1.0867 w_k1_INTC_s103 + 1.07873 w_k1_KO_s103
                                 + 0.819509 w_k1_MAT_s103
                                 + 0.931521 w_k1_MSFT_s103
                                 + 1.05086 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.01981 w_k1_ADM_s103
                                 + 1.49854 w_k1_AXP_s103
                                 + 1.61131 w_k1_DIS_s103
                                 + 1.35039 w_k1_FDX_s103
                                 + 1.14631 w_k1_IBM_s103
                                 + 1.30075 w_k1_INTC_s103
                                 + 1.30095 w_k1_KO_s103 + 1.37189 w_k1_MAT_s103
                                 + 1.31501 w_k1_MSFT_s103
                                 + 1.27344 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.862628 w_k1_ADM_s103
                                 + 1.08949 w_k1_AXP_s103
                                 + 1.12214 w_k1_DIS_s103
                                 + 1.03059 w_k1_FDX_s103
                                 + 1.42223 w_k1_IBM_s103
                                 + 1.11505 w_k1_INTC_s103
                                 + 1.24232 w_k1_KO_s103 + 1.39362 w_k1_MAT_s103
                                 + 1.01388 w_k1_MSFT_s103
                                 + 1.03403 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.13922 w_k1_ADM_s103
                                 + 0.821654 w_k1_AXP_s103
                                 + 0.87632 w_k1_DIS_s103
                                 + 1.02447 w_k1_FDX_s103
                                 + 0.608365 w_k1_IBM_s103
                                 + 1.18405 w_k1_INTC_s103
                                 + 0.740539 w_k1_KO_s103
                                 + 0.46135 w_k1_MAT_s103
                                 + 0.948847 w_k1_MSFT_s103
                                 + 0.902841 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.885995 w_k1_ADM_s103
                                 + 0.814885 w_k1_AXP_s103
                                 + 0.773406 w_k1_DIS_s103
                                 + 0.826841 w_k1_FDX_s103
                                 + 1.06094 w_k1_IBM_s103
                                 + 0.986744 w_k1_INTC_s103
                                 + 0.896512 w_k1_KO_s103
                                 + 0.811023 w_k1_MAT_s103
                                 + 0.86171 w_k1_MSFT_s103
                                 + 0.896736 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.811954 w_k1_ADM_s103
                                 + 0.661907 w_k1_AXP_s103
                                 + 0.7191 w_k1_DIS_s103
                                 + 0.817872 w_k1_FDX_s103
                                 + 0.832871 w_k1_IBM_s103
                                 + 0.809048 w_k1_INTC_s103
                                 + 0.889533 w_k1_KO_s103
                                 + 0.553068 w_k1_MAT_s103
                                 + 0.784045 w_k1_MSFT_s103
                                 + 0.792638 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.20313 w_k1_ADM_s103
                                 + 1.40038 w_k1_AXP_s103
                                 + 1.14594 w_k1_DIS_s103
                                 + 1.27534 w_k1_FDX_s103
                                 + 1.07917 w_k1_IBM_s103
                                 + 1.2126 w_k1_INTC_s103 + 1.03511 w_k1_KO_s103
                                 + 1.16143 w_k1_MAT_s103
                                 + 1.08595 w_k1_MSFT_s103
                                 + 1.05897 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.911268 w_k1_ADM_s104
                                 + 1.36402 w_k1_AXP_s104
                                 + 1.48364 w_k1_DIS_s104
                                 + 1.16174 w_k1_FDX_s104 + 1.4838 w_k1_IBM_s104
                                 + 1.00173 w_k1_INTC_s104
                                 + 1.30255 w_k1_KO_s104 + 1.63101 w_k1_MAT_s104
                                 + 1.15992 w_k1_MSFT_s104
                                 + 1.12729 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.823952 w_k1_ADM_s104
                                 + 1.02751 w_k1_AXP_s104
                                 + 1.11199 w_k1_DIS_s104
                                 + 1.14047 w_k1_FDX_s104
                                 + 0.786528 w_k1_IBM_s104
                                 + 1.21963 w_k1_INTC_s104
                                 + 0.924814 w_k1_KO_s104
                                 + 0.709173 w_k1_MAT_s104
                                 + 1.21226 w_k1_MSFT_s104
                                 + 1.03106 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.27031 w_k1_ADM_s104
                                 + 1.31901 w_k1_AXP_s104 + 1.2012 w_k1_DIS_s104
                                 + 1.19072 w_k1_FDX_s104
                                 + 1.22932 w_k1_IBM_s104
                                 + 1.26321 w_k1_INTC_s104
                                 + 1.19576 w_k1_KO_s104 + 1.72507 w_k1_MAT_s104
                                 + 1.06994 w_k1_MSFT_s104
                                 + 1.26379 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.26438 w_k1_ADM_s104
                                 + 1.96194 w_k1_AXP_s104
                                 + 1.78708 w_k1_DIS_s104
                                 + 1.67578 w_k1_FDX_s104
                                 + 1.12944 w_k1_IBM_s104
                                 + 1.85549 w_k1_INTC_s104
                                 + 1.26538 w_k1_KO_s104 + 1.75024 w_k1_MAT_s104
                                 + 1.64326 w_k1_MSFT_s104
                                 + 1.35617 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.630228 w_k1_ADM_s104
                                 + 0.701143 w_k1_AXP_s104
                                 + 0.571136 w_k1_DIS_s104
                                 + 0.755067 w_k1_FDX_s104
                                 + 0.75673 w_k1_IBM_s104
                                 + 0.871567 w_k1_INTC_s104
                                 + 0.701972 w_k1_KO_s104
                                 + 0.676902 w_k1_MAT_s104
                                 + 0.766881 w_k1_MSFT_s104
                                 + 0.751605 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.884642 w_k1_ADM_s104
                                 + 1.00279 w_k1_AXP_s104
                                 + 1.10252 w_k1_DIS_s104
                                 + 0.973038 w_k1_FDX_s104
                                 + 1.18529 w_k1_IBM_s104 + 1.089 w_k1_INTC_s104
                                 + 1.033 w_k1_KO_s104 + 1.16153 w_k1_MAT_s104
                                 + 1.02949 w_k1_MSFT_s104
                                 + 1.04201 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.850035 w_k1_ADM_s104
                                 + 0.779018 w_k1_AXP_s104
                                 + 0.724189 w_k1_DIS_s104
                                 + 0.833132 w_k1_FDX_s104
                                 + 1.08112 w_k1_IBM_s104
                                 + 0.817725 w_k1_INTC_s104
                                 + 0.939299 w_k1_KO_s104
                                 + 1.23466 w_k1_MAT_s104
                                 + 0.940534 w_k1_MSFT_s104
                                 + 0.929203 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.10618 w_k1_ADM_s104
                                 + 1.42674 w_k1_AXP_s104
                                 + 0.935355 w_k1_DIS_s104
                                 + 1.19595 w_k1_FDX_s104
                                 + 1.07933 w_k1_IBM_s104
                                 + 1.23542 w_k1_INTC_s104
                                 + 0.985451 w_k1_KO_s104 + 1.1932 w_k1_MAT_s104
                                 + 1.03262 w_k1_MSFT_s104 + 1.0213 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.03776 w_k1_ADM_s105
                                 + 1.61931 w_k1_AXP_s105
                                 + 1.52562 w_k1_DIS_s105
                                 + 1.10322 w_k1_FDX_s105
                                 + 1.67295 w_k1_IBM_s105
                                 + 1.09687 w_k1_INTC_s105
                                 + 1.42715 w_k1_KO_s105 + 2.27878 w_k1_MAT_s105
                                 + 1.14812 w_k1_MSFT_s105 + 1.1685 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.29662 w_k1_ADM_s105
                                 + 1.06774 w_k1_AXP_s105
                                 + 1.08614 w_k1_DIS_s105
                                 + 1.09021 w_k1_FDX_s105
                                 + 0.713568 w_k1_IBM_s105
                                 + 1.09876 w_k1_INTC_s105
                                 + 0.987833 w_k1_KO_s105
                                 + 0.905068 w_k1_MAT_s105
                                 + 1.0884 w_k1_MSFT_s105 + 1.05372 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.27222 w_k1_ADM_s105
                                 + 1.16322 w_k1_AXP_s105
                                 + 0.982112 w_k1_DIS_s105
                                 + 1.03741 w_k1_FDX_s105
                                 + 1.11302 w_k1_IBM_s105
                                 + 1.0742 w_k1_INTC_s105 + 1.17179 w_k1_KO_s105
                                 + 1.27782 w_k1_MAT_s105
                                 + 0.976177 w_k1_MSFT_s105
                                 + 0.997954 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.925354 w_k1_ADM_s105
                                 + 0.878693 w_k1_AXP_s105
                                 + 0.962761 w_k1_DIS_s105
                                 + 0.808595 w_k1_FDX_s105
                                 + 1.34918 w_k1_IBM_s105
                                 + 1.10618 w_k1_INTC_s105
                                 + 1.09914 w_k1_KO_s105 + 1.34661 w_k1_MAT_s105
                                 + 0.963587 w_k1_MSFT_s105
                                 + 0.91261 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.36446 w_k1_ADM_s105
                                 + 1.43196 w_k1_AXP_s105
                                 + 1.39328 w_k1_DIS_s105
                                 + 1.31256 w_k1_FDX_s105
                                 + 1.25737 w_k1_IBM_s105
                                 + 1.14062 w_k1_INTC_s105
                                 + 1.15166 w_k1_KO_s105 + 1.57401 w_k1_MAT_s105
                                 + 1.10296 w_k1_MSFT_s105
                                 + 1.21753 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.846823 w_k1_ADM_s105
                                 + 0.914655 w_k1_AXP_s105
                                 + 0.711954 w_k1_DIS_s105
                                 + 0.839995 w_k1_FDX_s105
                                 + 0.713672 w_k1_IBM_s105
                                 + 0.857893 w_k1_INTC_s105
                                 + 0.727394 w_k1_KO_s105
                                 + 0.711966 w_k1_MAT_s105
                                 + 0.872347 w_k1_MSFT_s105
                                 + 0.831087 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.42423 w_k1_ADM_s105
                                 + 1.43732 w_k1_AXP_s105
                                 + 1.48376 w_k1_DIS_s105
                                 + 1.21219 w_k1_FDX_s105
                                 + 0.922793 w_k1_IBM_s105
                                 + 1.17554 w_k1_INTC_s105 + 1.2289 w_k1_KO_s105
                                 + 1.22186 w_k1_MAT_s105
                                 + 1.19593 w_k1_MSFT_s105
                                 + 1.23627 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.959907 w_k1_ADM_s105
                                 + 0.776452 w_k1_AXP_s105
                                 + 0.812591 w_k1_DIS_s105
                                 + 0.769468 w_k1_FDX_s105
                                 + 1.01736 w_k1_IBM_s105
                                 + 0.783471 w_k1_INTC_s105
                                 + 0.979105 w_k1_KO_s105
                                 + 0.978663 w_k1_MAT_s105
                                 + 0.845899 w_k1_MSFT_s105
                                 + 0.887977 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.1481 w_k1_ADM_s106
                                 + 1.29867 w_k1_AXP_s106
                                 + 1.33233 w_k1_DIS_s106
                                 + 1.33104 w_k1_FDX_s106
                                 + 0.968618 w_k1_IBM_s106
                                 + 1.11913 w_k1_INTC_s106
                                 + 1.04348 w_k1_KO_s106 + 1.01866 w_k1_MAT_s106
                                 + 1.09865 w_k1_MSFT_s106
                                 + 1.13406 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.0868 w_k1_ADM_s106
                                 + 1.00852 w_k1_AXP_s106
                                 + 0.967772 w_k1_DIS_s106
                                 + 1.01301 w_k1_FDX_s106
                                 + 1.11737 w_k1_IBM_s106
                                 + 1.05494 w_k1_INTC_s106
                                 + 1.04181 w_k1_KO_s106
                                 + 0.969169 w_k1_MAT_s106
                                 + 1.06715 w_k1_MSFT_s106
                                 + 0.961151 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.962873 w_k1_ADM_s106
                                 + 0.837848 w_k1_AXP_s106
                                 + 0.775463 w_k1_DIS_s106
                                 + 0.94493 w_k1_FDX_s106
                                 + 0.669463 w_k1_IBM_s106
                                 + 0.899309 w_k1_INTC_s106
                                 + 0.777097 w_k1_KO_s106
                                 + 0.760463 w_k1_MAT_s106
                                 + 0.849975 w_k1_MSFT_s106
                                 + 0.996637 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.86417 w_k1_ADM_s106
                                 + 0.780868 w_k1_AXP_s106
                                 + 1.02986 w_k1_DIS_s106
                                 + 1.13188 w_k1_FDX_s106
                                 + 0.746831 w_k1_IBM_s106
                                 + 1.0328 w_k1_INTC_s106
                                 + 0.980806 w_k1_KO_s106
                                 + 0.726174 w_k1_MAT_s106
                                 + 1.19528 w_k1_MSFT_s106
                                 + 0.978891 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.64229 w_k1_ADM_s106
                                 + 1.64094 w_k1_AXP_s106
                                 + 1.99661 w_k1_DIS_s106
                                 + 1.39628 w_k1_FDX_s106
                                 + 1.74378 w_k1_IBM_s106
                                 + 1.25779 w_k1_INTC_s106
                                 + 1.55166 w_k1_KO_s106 + 1.6506 w_k1_MAT_s106
                                 + 1.18975 w_k1_MSFT_s106
                                 + 1.49253 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.777074 w_k1_ADM_s106
                                 + 0.846311 w_k1_AXP_s106
                                 + 0.848156 w_k1_DIS_s106
                                 + 0.849315 w_k1_FDX_s106
                                 + 1.25186 w_k1_IBM_s106
                                 + 0.758937 w_k1_INTC_s106
                                 + 1.02813 w_k1_KO_s106 + 1.25498 w_k1_MAT_s106
                                 + 0.801001 w_k1_MSFT_s106
                                 + 1.07325 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.894567 w_k1_ADM_s106
                                 + 0.69241 w_k1_AXP_s106
                                 + 0.810203 w_k1_DIS_s106
                                 + 0.897461 w_k1_FDX_s106
                                 + 0.84521 w_k1_IBM_s106
                                 + 0.849893 w_k1_INTC_s106
                                 + 0.860743 w_k1_KO_s106
                                 + 0.74205 w_k1_MAT_s106
                                 + 0.903365 w_k1_MSFT_s106
                                 + 0.889444 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.777365 w_k1_ADM_s106
                                 + 0.586874 w_k1_AXP_s106
                                 + 0.523012 w_k1_DIS_s106
                                 + 0.650054 w_k1_FDX_s106
                                 + 0.716626 w_k1_IBM_s106
                                 + 0.652399 w_k1_INTC_s106
                                 + 0.649501 w_k1_KO_s106
                                 + 0.722277 w_k1_MAT_s106
                                 + 0.644671 w_k1_MSFT_s106
                                 + 0.824395 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.02074 w_k1_ADM_s107
                                 + 1.26407 w_k1_AXP_s107
                                 + 1.77107 w_k1_DIS_s107
                                 + 1.25046 w_k1_FDX_s107
                                 + 1.74603 w_k1_IBM_s107
                                 + 0.838823 w_k1_INTC_s107
                                 + 1.40656 w_k1_KO_s107 + 1.96865 w_k1_MAT_s107
                                 + 1.09864 w_k1_MSFT_s107 + 1.2679 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.681976 w_k1_ADM_s107
                                 + 0.736073 w_k1_AXP_s107
                                 + 0.722918 w_k1_DIS_s107
                                 + 0.841735 w_k1_FDX_s107
                                 + 0.628735 w_k1_IBM_s107
                                 + 0.738821 w_k1_INTC_s107
                                 + 0.617826 w_k1_KO_s107
                                 + 0.543422 w_k1_MAT_s107
                                 + 0.806302 w_k1_MSFT_s107
                                 + 0.794435 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.96057 w_k1_ADM_s107
                                 + 1.18368 w_k1_AXP_s107
                                 + 1.25573 w_k1_DIS_s107
                                 + 1.18802 w_k1_FDX_s107
                                 + 1.06923 w_k1_IBM_s107
                                 + 1.04764 w_k1_INTC_s107 + 1.1257 w_k1_KO_s107
                                 + 1.23201 w_k1_MAT_s107
                                 + 1.09739 w_k1_MSFT_s107
                                 + 1.20886 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.887453 w_k1_ADM_s107
                                 + 0.896455 w_k1_AXP_s107
                                 + 0.773317 w_k1_DIS_s107
                                 + 0.838956 w_k1_FDX_s107
                                 + 1.03189 w_k1_IBM_s107
                                 + 0.746177 w_k1_INTC_s107
                                 + 0.872541 w_k1_KO_s107
                                 + 1.17035 w_k1_MAT_s107
                                 + 0.893052 w_k1_MSFT_s107
                                 + 0.944245 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.783106 w_k1_ADM_s107
                                 + 0.707955 w_k1_AXP_s107
                                 + 0.647104 w_k1_DIS_s107
                                 + 0.808936 w_k1_FDX_s107
                                 + 0.863555 w_k1_IBM_s107
                                 + 0.842771 w_k1_INTC_s107
                                 + 0.70036 w_k1_KO_s107
                                 + 0.852754 w_k1_MAT_s107
                                 + 0.772471 w_k1_MSFT_s107
                                 + 0.898989 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.942818 w_k1_ADM_s107
                                 + 0.856158 w_k1_AXP_s107
                                 + 1.17078 w_k1_DIS_s107
                                 + 1.05733 w_k1_FDX_s107
                                 + 0.917914 w_k1_IBM_s107
                                 + 1.07672 w_k1_INTC_s107 + 1.054 w_k1_KO_s107
                                 + 0.889452 w_k1_MAT_s107
                                 + 1.1439 w_k1_MSFT_s107 + 1.08767 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.696294 w_k1_ADM_s107
                                 + 0.753851 w_k1_AXP_s107
                                 + 0.892112 w_k1_DIS_s107
                                 + 0.990075 w_k1_FDX_s107
                                 + 0.648837 w_k1_IBM_s107
                                 + 0.941586 w_k1_INTC_s107
                                 + 0.770792 w_k1_KO_s107
                                 + 0.709606 w_k1_MAT_s107
                                 + 1.0179 w_k1_MSFT_s107 + 1.0439 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.12226 w_k1_ADM_s107
                                 + 1.62927 w_k1_AXP_s107 + 1.395 w_k1_DIS_s107
                                 + 1.2879 w_k1_FDX_s107 + 1.47322 w_k1_IBM_s107
                                 + 1.26391 w_k1_INTC_s107
                                 + 1.35175 w_k1_KO_s107 + 2.33073 w_k1_MAT_s107
                                 + 1.37806 w_k1_MSFT_s107
                                 + 1.18712 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13275 w_k1_ADM_s110
                                 + 1.30139 w_k1_AXP_s110
                                 + 1.60397 w_k1_DIS_s110
                                 + 1.31854 w_k1_FDX_s110
                                 + 1.14474 w_k1_IBM_s110
                                 + 1.12832 w_k1_INTC_s110
                                 + 1.35024 w_k1_KO_s110 + 1.60708 w_k1_MAT_s110
                                 + 1.20332 w_k1_MSFT_s110
                                 + 1.19239 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.683708 w_k1_ADM_s110
                                 + 0.875731 w_k1_AXP_s110
                                 + 0.937695 w_k1_DIS_s110
                                 + 0.880817 w_k1_FDX_s110
                                 + 0.860403 w_k1_IBM_s110
                                 + 1.03735 w_k1_INTC_s110
                                 + 0.983379 w_k1_KO_s110
                                 + 1.20977 w_k1_MAT_s110
                                 + 0.938481 w_k1_MSFT_s110
                                 + 1.0242 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.705365 w_k1_ADM_s110
                                 + 0.809802 w_k1_AXP_s110
                                 + 0.895864 w_k1_DIS_s110
                                 + 0.840636 w_k1_FDX_s110
                                 + 1.39429 w_k1_IBM_s110
                                 + 0.833421 w_k1_INTC_s110
                                 + 1.15128 w_k1_KO_s110 + 1.5248 w_k1_MAT_s110
                                 + 0.776814 w_k1_MSFT_s110
                                 + 0.997299 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03822 w_k1_ADM_s110
                                 + 1.45109 w_k1_AXP_s110
                                 + 1.19137 w_k1_DIS_s110
                                 + 1.27324 w_k1_FDX_s110
                                 + 0.90293 w_k1_IBM_s110
                                 + 1.07081 w_k1_INTC_s110
                                 + 1.15907 w_k1_KO_s110 + 1.55976 w_k1_MAT_s110
                                 + 1.22393 w_k1_MSFT_s110
                                 + 1.19071 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.824528 w_k1_ADM_s110
                                 + 1.02162 w_k1_AXP_s110
                                 + 1.09591 w_k1_DIS_s110
                                 + 0.957707 w_k1_FDX_s110
                                 + 1.26508 w_k1_IBM_s110
                                 + 1.06298 w_k1_INTC_s110
                                 + 1.23587 w_k1_KO_s110 + 1.56442 w_k1_MAT_s110
                                 + 1.08937 w_k1_MSFT_s110
                                 + 1.08917 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.36106 w_k1_ADM_s110
                                 + 1.63594 w_k1_AXP_s110
                                 + 1.75733 w_k1_DIS_s110
                                 + 1.42316 w_k1_FDX_s110
                                 + 1.51427 w_k1_IBM_s110
                                 + 1.0187 w_k1_INTC_s110 + 1.46391 w_k1_KO_s110
                                 + 1.89294 w_k1_MAT_s110
                                 + 1.16736 w_k1_MSFT_s110
                                 + 1.21415 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.998739 w_k1_ADM_s110
                                 + 0.969451 w_k1_AXP_s110
                                 + 0.908171 w_k1_DIS_s110
                                 + 1.0652 w_k1_FDX_s110
                                 + 0.625355 w_k1_IBM_s110
                                 + 0.901204 w_k1_INTC_s110
                                 + 0.759005 w_k1_KO_s110
                                 + 0.890302 w_k1_MAT_s110
                                 + 0.916335 w_k1_MSFT_s110
                                 + 0.937961 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.894364 w_k1_ADM_s110
                                 + 1.09125 w_k1_AXP_s110
                                 + 1.22594 w_k1_DIS_s110 + 1.0919 w_k1_FDX_s110
                                 + 1.08972 w_k1_IBM_s110
                                 + 1.00928 w_k1_INTC_s110
                                 + 1.00464 w_k1_KO_s110 + 1.08409 w_k1_MAT_s110
                                 + 1.06424 w_k1_MSFT_s110
                                 + 0.888632 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.848638 w_k1_ADM_s111
                                 + 0.958728 w_k1_AXP_s111
                                 + 0.956682 w_k1_DIS_s111
                                 + 0.93051 w_k1_FDX_s111
                                 + 1.12971 w_k1_IBM_s111
                                 + 1.08294 w_k1_INTC_s111
                                 + 1.13684 w_k1_KO_s111 + 1.14082 w_k1_MAT_s111
                                 + 0.886053 w_k1_MSFT_s111
                                 + 1.09296 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.739934 w_k1_ADM_s111
                                 + 0.759121 w_k1_AXP_s111
                                 + 0.721932 w_k1_DIS_s111
                                 + 0.782742 w_k1_FDX_s111
                                 + 0.849448 w_k1_IBM_s111
                                 + 0.87537 w_k1_INTC_s111
                                 + 0.884393 w_k1_KO_s111
                                 + 0.76924 w_k1_MAT_s111
                                 + 0.858182 w_k1_MSFT_s111
                                 + 0.823031 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.1159 w_k1_ADM_s111
                                 + 1.33472 w_k1_AXP_s111
                                 + 1.32066 w_k1_DIS_s111
                                 + 1.19331 w_k1_FDX_s111
                                 + 1.46128 w_k1_IBM_s111
                                 + 1.25063 w_k1_INTC_s111
                                 + 1.37447 w_k1_KO_s111 + 1.61289 w_k1_MAT_s111
                                 + 1.26373 w_k1_MSFT_s111
                                 + 1.24974 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.17162 w_k1_ADM_s111
                                 + 1.6697 w_k1_AXP_s111 + 1.97946 w_k1_DIS_s111
                                 + 1.55479 w_k1_FDX_s111
                                 + 1.23444 w_k1_IBM_s111
                                 + 1.59975 w_k1_INTC_s111
                                 + 1.49086 w_k1_KO_s111
                                 + 0.907094 w_k1_MAT_s111
                                 + 1.54924 w_k1_MSFT_s111
                                 + 1.23061 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.954677 w_k1_ADM_s111
                                 + 1.18722 w_k1_AXP_s111
                                 + 1.06411 w_k1_DIS_s111
                                 + 1.20285 w_k1_FDX_s111
                                 + 0.877546 w_k1_IBM_s111
                                 + 1.08143 w_k1_INTC_s111
                                 + 0.870144 w_k1_KO_s111
                                 + 0.752252 w_k1_MAT_s111
                                 + 1.10301 w_k1_MSFT_s111
                                 + 0.902011 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.838928 w_k1_ADM_s111
                                 + 0.833308 w_k1_AXP_s111
                                 + 0.994619 w_k1_DIS_s111
                                 + 0.998803 w_k1_FDX_s111
                                 + 0.750184 w_k1_IBM_s111
                                 + 1.22897 w_k1_INTC_s111
                                 + 0.832601 w_k1_KO_s111
                                 + 0.429501 w_k1_MAT_s111
                                 + 1.00414 w_k1_MSFT_s111
                                 + 0.979115 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.739213 w_k1_ADM_s111
                                 + 0.662094 w_k1_AXP_s111
                                 + 0.654041 w_k1_DIS_s111
                                 + 0.721759 w_k1_FDX_s111
                                 + 0.997755 w_k1_IBM_s111
                                 + 0.802431 w_k1_INTC_s111
                                 + 0.856106 w_k1_KO_s111
                                 + 0.724918 w_k1_MAT_s111
                                 + 0.69487 w_k1_MSFT_s111
                                 + 0.865027 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.889615 w_k1_ADM_s111
                                 + 0.714235 w_k1_AXP_s111
                                 + 0.631992 w_k1_DIS_s111
                                 + 0.882485 w_k1_FDX_s111
                                 + 0.574415 w_k1_IBM_s111
                                 + 0.875813 w_k1_INTC_s111
                                 + 0.697475 w_k1_KO_s111
                                 + 0.401928 w_k1_MAT_s111
                                 + 0.817768 w_k1_MSFT_s111
                                 + 0.84719 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.23308 w_k1_ADM_s112
                                 + 1.10361 w_k1_AXP_s112
                                 + 1.21182 w_k1_DIS_s112
                                 + 1.21829 w_k1_FDX_s112
                                 + 0.742068 w_k1_IBM_s112
                                 + 1.31966 w_k1_INTC_s112
                                 + 1.15637 w_k1_KO_s112
                                 + 0.706666 w_k1_MAT_s112
                                 + 1.26012 w_k1_MSFT_s112 + 1.0723 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.946372 w_k1_ADM_s112
                                 + 0.97245 w_k1_AXP_s112
                                 + 0.833125 w_k1_DIS_s112
                                 + 0.840205 w_k1_FDX_s112
                                 + 0.898373 w_k1_IBM_s112
                                 + 0.927818 w_k1_INTC_s112
                                 + 1.04077 w_k1_KO_s112
                                 + 0.715312 w_k1_MAT_s112
                                 + 0.820739 w_k1_MSFT_s112
                                 + 0.956155 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.830913 w_k1_ADM_s112
                                 + 0.899871 w_k1_AXP_s112
                                 + 0.766456 w_k1_DIS_s112
                                 + 0.731893 w_k1_FDX_s112
                                 + 1.49501 w_k1_IBM_s112
                                 + 0.964795 w_k1_INTC_s112
                                 + 1.13737 w_k1_KO_s112 + 1.35754 w_k1_MAT_s112
                                 + 0.828803 w_k1_MSFT_s112
                                 + 0.953408 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.61407 w_k1_ADM_s112
                                 + 1.55693 w_k1_AXP_s112
                                 + 1.67909 w_k1_DIS_s112
                                 + 1.33424 w_k1_FDX_s112
                                 + 1.17602 w_k1_IBM_s112
                                 + 1.42105 w_k1_INTC_s112
                                 + 1.56959 w_k1_KO_s112 + 1.35661 w_k1_MAT_s112
                                 + 1.37932 w_k1_MSFT_s112 + 1.3501 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.947578 w_k1_ADM_s112
                                 + 0.834242 w_k1_AXP_s112
                                 + 0.833645 w_k1_DIS_s112
                                 + 0.965326 w_k1_FDX_s112
                                 + 0.695673 w_k1_IBM_s112
                                 + 0.931722 w_k1_INTC_s112
                                 + 0.804422 w_k1_KO_s112
                                 + 0.505306 w_k1_MAT_s112
                                 + 1.03472 w_k1_MSFT_s112
                                 + 0.85745 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.02442 w_k1_ADM_s112
                                 + 0.976429 w_k1_AXP_s112
                                 + 1.34658 w_k1_DIS_s112 + 1.0026 w_k1_FDX_s112
                                 + 1.60992 w_k1_IBM_s112
                                 + 1.19215 w_k1_INTC_s112
                                 + 1.51596 w_k1_KO_s112 + 1.07333 w_k1_MAT_s112
                                 + 1.1337 w_k1_MSFT_s112 + 1.30294 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.28035 w_k1_ADM_s112
                                 + 1.13364 w_k1_AXP_s112
                                 + 1.02581 w_k1_DIS_s112
                                 + 1.06227 w_k1_FDX_s112
                                 + 0.90773 w_k1_IBM_s112
                                 + 1.17061 w_k1_INTC_s112
                                 + 1.14822 w_k1_KO_s112
                                 + 0.831628 w_k1_MAT_s112
                                 + 1.03219 w_k1_MSFT_s112
                                 + 1.04325 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.808157 w_k1_ADM_s112
                                 + 0.597521 w_k1_AXP_s112
                                 + 0.597457 w_k1_DIS_s112
                                 + 0.722549 w_k1_FDX_s112
                                 + 1.0749 w_k1_IBM_s112
                                 + 0.894732 w_k1_INTC_s112
                                 + 0.909086 w_k1_KO_s112
                                 + 0.659727 w_k1_MAT_s112
                                 + 0.807325 w_k1_MSFT_s112
                                 + 0.897078 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.03405 w_k1_ADM_s113
                                 + 0.894524 w_k1_AXP_s113
                                 + 0.843039 w_k1_DIS_s113
                                 + 1.04074 w_k1_FDX_s113
                                 + 0.932826 w_k1_IBM_s113
                                 + 1.14943 w_k1_INTC_s113 + 1.0511 w_k1_KO_s113
                                 + 0.680725 w_k1_MAT_s113
                                 + 0.956456 w_k1_MSFT_s113
                                 + 0.971538 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.12131 w_k1_ADM_s113
                                 + 1.10197 w_k1_AXP_s113
                                 + 0.822362 w_k1_DIS_s113
                                 + 0.842524 w_k1_FDX_s113
                                 + 1.25533 w_k1_IBM_s113
                                 + 0.978129 w_k1_INTC_s113
                                 + 0.997943 w_k1_KO_s113
                                 + 1.44121 w_k1_MAT_s113
                                 + 0.814803 w_k1_MSFT_s113
                                 + 0.933863 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.838912 w_k1_ADM_s113
                                 + 0.699487 w_k1_AXP_s113
                                 + 0.783208 w_k1_DIS_s113
                                 + 0.766469 w_k1_FDX_s113
                                 + 0.812112 w_k1_IBM_s113
                                 + 0.78591 w_k1_INTC_s113
                                 + 0.791529 w_k1_KO_s113
                                 + 0.604244 w_k1_MAT_s113
                                 + 0.758077 w_k1_MSFT_s113
                                 + 0.777251 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.24916 w_k1_ADM_s113
                                 + 0.940109 w_k1_AXP_s113
                                 + 1.13099 w_k1_DIS_s113
                                 + 1.20029 w_k1_FDX_s113
                                 + 0.810263 w_k1_IBM_s113
                                 + 1.27762 w_k1_INTC_s113
                                 + 0.993897 w_k1_KO_s113
                                 + 0.62525 w_k1_MAT_s113
                                 + 1.06976 w_k1_MSFT_s113
                                 + 1.01954 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.06988 w_k1_ADM_s113
                                 + 1.1121 w_k1_AXP_s113
                                 + 0.798519 w_k1_DIS_s113
                                 + 1.04456 w_k1_FDX_s113
                                 + 0.66883 w_k1_IBM_s113
                                 + 1.18215 w_k1_INTC_s113
                                 + 0.746005 w_k1_KO_s113
                                 + 0.75279 w_k1_MAT_s113
                                 + 0.843054 w_k1_MSFT_s113
                                 + 0.882426 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.42563 w_k1_ADM_s113
                                 + 1.44193 w_k1_AXP_s113 + 1.618 w_k1_DIS_s113
                                 + 1.3928 w_k1_FDX_s113 + 1.20192 w_k1_IBM_s113
                                 + 1.23992 w_k1_INTC_s113
                                 + 1.30311 w_k1_KO_s113 + 1.26979 w_k1_MAT_s113
                                 + 1.30297 w_k1_MSFT_s113
                                 + 1.15781 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.14638 w_k1_ADM_s113
                                 + 0.958379 w_k1_AXP_s113
                                 + 1.28803 w_k1_DIS_s113
                                 + 1.01038 w_k1_FDX_s113
                                 + 1.77018 w_k1_IBM_s113
                                 + 1.19649 w_k1_INTC_s113
                                 + 1.37947 w_k1_KO_s113 + 1.37921 w_k1_MAT_s113
                                 + 1.01726 w_k1_MSFT_s113
                                 + 1.04286 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.02517 w_k1_ADM_s113
                                 + 0.754198 w_k1_AXP_s113
                                 + 0.967169 w_k1_DIS_s113
                                 + 0.832841 w_k1_FDX_s113
                                 + 1.26495 w_k1_IBM_s113
                                 + 0.875235 w_k1_INTC_s113
                                 + 1.14025 w_k1_KO_s113
                                 + 0.850811 w_k1_MAT_s113
                                 + 0.797517 w_k1_MSFT_s113
                                 + 0.957086 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.08293 w_k1_ADM_s114
                                 + 1.29229 w_k1_AXP_s114
                                 + 1.49277 w_k1_DIS_s114
                                 + 1.29619 w_k1_FDX_s114
                                 + 1.04321 w_k1_IBM_s114
                                 + 0.976761 w_k1_INTC_s114
                                 + 1.13886 w_k1_KO_s114 + 1.2492 w_k1_MAT_s114
                                 + 1.1769 w_k1_MSFT_s114 + 1.05468 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04978 w_k1_ADM_s114
                                 + 1.02412 w_k1_AXP_s114
                                 + 1.03893 w_k1_DIS_s114
                                 + 0.964483 w_k1_FDX_s114
                                 + 0.867842 w_k1_IBM_s114
                                 + 1.01081 w_k1_INTC_s114
                                 + 0.954262 w_k1_KO_s114
                                 + 0.944595 w_k1_MAT_s114
                                 + 0.976078 w_k1_MSFT_s114
                                 + 0.929666 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.920703 w_k1_ADM_s114
                                 + 0.767387 w_k1_AXP_s114
                                 + 0.68766 w_k1_DIS_s114
                                 + 0.772502 w_k1_FDX_s114
                                 + 0.792122 w_k1_IBM_s114
                                 + 0.828511 w_k1_INTC_s114
                                 + 0.86674 w_k1_KO_s114
                                 + 0.750281 w_k1_MAT_s114
                                 + 0.814199 w_k1_MSFT_s114
                                 + 0.791024 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.00073 w_k1_ADM_s114
                                 + 0.857973 w_k1_AXP_s114
                                 + 0.811484 w_k1_DIS_s114
                                 + 0.931154 w_k1_FDX_s114
                                 + 0.972913 w_k1_IBM_s114
                                 + 0.930362 w_k1_INTC_s114
                                 + 0.822241 w_k1_KO_s114
                                 + 0.678483 w_k1_MAT_s114
                                 + 0.877132 w_k1_MSFT_s114
                                 + 0.994844 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.830177 w_k1_ADM_s114
                                 + 1.09146 w_k1_AXP_s114
                                 + 1.16186 w_k1_DIS_s114 + 1.0322 w_k1_FDX_s114
                                 + 1.52811 w_k1_IBM_s114
                                 + 1.10676 w_k1_INTC_s114
                                 + 1.34783 w_k1_KO_s114 + 1.49617 w_k1_MAT_s114
                                 + 1.12135 w_k1_MSFT_s114 + 1.132 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.18507 w_k1_ADM_s114
                                 + 0.804552 w_k1_AXP_s114
                                 + 0.763703 w_k1_DIS_s114
                                 + 0.931994 w_k1_FDX_s114
                                 + 0.650866 w_k1_IBM_s114
                                 + 0.919957 w_k1_INTC_s114
                                 + 0.801005 w_k1_KO_s114
                                 + 0.503893 w_k1_MAT_s114
                                 + 0.917597 w_k1_MSFT_s114
                                 + 0.839688 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.922545 w_k1_ADM_s114
                                 + 0.995501 w_k1_AXP_s114
                                 + 0.830827 w_k1_DIS_s114
                                 + 0.822919 w_k1_FDX_s114
                                 + 1.04514 w_k1_IBM_s114
                                 + 0.901689 w_k1_INTC_s114
                                 + 0.811037 w_k1_KO_s114
                                 + 0.793347 w_k1_MAT_s114
                                 + 0.788085 w_k1_MSFT_s114
                                 + 0.836086 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.05581 w_k1_ADM_s114
                                 + 0.950731 w_k1_AXP_s114
                                 + 1.1983 w_k1_DIS_s114 + 1.06643 w_k1_FDX_s114
                                 + 1.10135 w_k1_IBM_s114
                                 + 1.03924 w_k1_INTC_s114
                                 + 1.14612 w_k1_KO_s114
                                 + 0.962944 w_k1_MAT_s114
                                 + 1.15258 w_k1_MSFT_s114
                                 + 0.963083 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.757721 w_k1_ADM_s115
                                 + 0.814105 w_k1_AXP_s115
                                 + 0.724298 w_k1_DIS_s115
                                 + 0.797746 w_k1_FDX_s115
                                 + 0.844233 w_k1_IBM_s115
                                 + 0.968298 w_k1_INTC_s115
                                 + 0.994569 w_k1_KO_s115
                                 + 0.777662 w_k1_MAT_s115
                                 + 0.81589 w_k1_MSFT_s115
                                 + 0.871123 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.32875 w_k1_ADM_s115
                                 + 1.83528 w_k1_AXP_s115 + 1.6261 w_k1_DIS_s115
                                 + 1.47489 w_k1_FDX_s115 + 1.0629 w_k1_IBM_s115
                                 + 1.19557 w_k1_INTC_s115
                                 + 1.37925 w_k1_KO_s115 + 1.60865 w_k1_MAT_s115
                                 + 1.35199 w_k1_MSFT_s115 + 1.3035 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.14949 w_k1_ADM_s115
                                 + 1.30724 w_k1_AXP_s115
                                 + 1.48736 w_k1_DIS_s115 + 1.2705 w_k1_FDX_s115
                                 + 1.22148 w_k1_IBM_s115
                                 + 1.25037 w_k1_INTC_s115
                                 + 1.54994 w_k1_KO_s115 + 1.39695 w_k1_MAT_s115
                                 + 1.23109 w_k1_MSFT_s115
                                 + 1.35007 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.886099 w_k1_ADM_s115
                                 + 0.926893 w_k1_AXP_s115
                                 + 1.08595 w_k1_DIS_s115
                                 + 0.893176 w_k1_FDX_s115
                                 + 1.64585 w_k1_IBM_s115
                                 + 1.11345 w_k1_INTC_s115
                                 + 1.46025 w_k1_KO_s115 + 1.38372 w_k1_MAT_s115
                                 + 1.16582 w_k1_MSFT_s115
                                 + 0.961684 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04824 w_k1_ADM_s115
                                 + 0.90624 w_k1_AXP_s115
                                 + 0.95181 w_k1_DIS_s115
                                 + 1.05411 w_k1_FDX_s115
                                 + 0.695874 w_k1_IBM_s115
                                 + 0.958334 w_k1_INTC_s115
                                 + 0.898241 w_k1_KO_s115
                                 + 0.775178 w_k1_MAT_s115
                                 + 1.14311 w_k1_MSFT_s115
                                 + 1.04848 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.827618 w_k1_ADM_s115
                                 + 1.12081 w_k1_AXP_s115
                                 + 0.867109 w_k1_DIS_s115
                                 + 0.972197 w_k1_FDX_s115
                                 + 0.917094 w_k1_IBM_s115
                                 + 0.91674 w_k1_INTC_s115
                                 + 0.977177 w_k1_KO_s115
                                 + 0.918645 w_k1_MAT_s115
                                 + 0.904903 w_k1_MSFT_s115
                                 + 0.971688 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.975045 w_k1_ADM_s115
                                 + 1.27032 w_k1_AXP_s115
                                 + 1.27897 w_k1_DIS_s115
                                 + 1.15062 w_k1_FDX_s115
                                 + 1.13603 w_k1_IBM_s115
                                 + 1.1336 w_k1_INTC_s115 + 1.15686 w_k1_KO_s115
                                 + 1.06127 w_k1_MAT_s115
                                 + 1.13854 w_k1_MSFT_s115 + 1.0268 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.95426 w_k1_ADM_s115
                                 + 0.855876 w_k1_AXP_s115
                                 + 1.01221 w_k1_DIS_s115
                                 + 0.899785 w_k1_FDX_s115
                                 + 1.10144 w_k1_IBM_s115
                                 + 0.9707 w_k1_INTC_s115 + 1.12436 w_k1_KO_s115
                                 + 1.01378 w_k1_MAT_s115
                                 + 0.964433 w_k1_MSFT_s115
                                 + 0.989453 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.08095 w_k1_ADM_s116
                                 + 1.57235 w_k1_AXP_s116
                                 + 1.46448 w_k1_DIS_s116 + 1.1541 w_k1_FDX_s116
                                 + 1.50444 w_k1_IBM_s116
                                 + 1.13566 w_k1_INTC_s116
                                 + 1.28846 w_k1_KO_s116 + 1.28202 w_k1_MAT_s116
                                 + 1.11028 w_k1_MSFT_s116
                                 + 1.16659 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.818159 w_k1_ADM_s116
                                 + 0.748282 w_k1_AXP_s116
                                 + 0.715172 w_k1_DIS_s116
                                 + 0.768059 w_k1_FDX_s116
                                 + 0.764733 w_k1_IBM_s116
                                 + 0.800368 w_k1_INTC_s116
                                 + 0.79014 w_k1_KO_s116
                                 + 0.609503 w_k1_MAT_s116
                                 + 0.809408 w_k1_MSFT_s116
                                 + 0.82743 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.849578 w_k1_ADM_s116
                                 + 0.853262 w_k1_AXP_s116
                                 + 0.967656 w_k1_DIS_s116
                                 + 0.90836 w_k1_FDX_s116
                                 + 1.06796 w_k1_IBM_s116
                                 + 0.940768 w_k1_INTC_s116
                                 + 1.04699 w_k1_KO_s116
                                 + 0.906451 w_k1_MAT_s116
                                 + 1.00831 w_k1_MSFT_s116
                                 + 1.00523 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.49973 w_k1_ADM_s116
                                 + 1.30442 w_k1_AXP_s116
                                 + 1.62662 w_k1_DIS_s116
                                 + 1.43388 w_k1_FDX_s116
                                 + 1.07202 w_k1_IBM_s116
                                 + 1.13048 w_k1_INTC_s116 + 1.3373 w_k1_KO_s116
                                 + 0.442823 w_k1_MAT_s116
                                 + 1.21208 w_k1_MSFT_s116
                                 + 1.21124 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07582 w_k1_ADM_s116
                                 + 1.22075 w_k1_AXP_s116
                                 + 1.07576 w_k1_DIS_s116
                                 + 1.01477 w_k1_FDX_s116
                                 + 1.01116 w_k1_IBM_s116
                                 + 0.894097 w_k1_INTC_s116
                                 + 1.0843 w_k1_KO_s116 + 0.799192 w_k1_MAT_s116
                                 + 0.970604 w_k1_MSFT_s116
                                 + 1.00087 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.03552 w_k1_ADM_s116
                                 + 0.903152 w_k1_AXP_s116
                                 + 1.19707 w_k1_DIS_s116
                                 + 0.947186 w_k1_FDX_s116
                                 + 1.51964 w_k1_IBM_s116
                                 + 0.902659 w_k1_INTC_s116
                                 + 1.26764 w_k1_KO_s116 + 1.09865 w_k1_MAT_s116
                                 + 1.01023 w_k1_MSFT_s116
                                 + 0.930755 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.21343 w_k1_ADM_s116
                                 + 1.31511 w_k1_AXP_s116
                                 + 1.20722 w_k1_DIS_s116
                                 + 1.20739 w_k1_FDX_s116
                                 + 1.12518 w_k1_IBM_s116
                                 + 1.08819 w_k1_INTC_s116
                                 + 1.19427 w_k1_KO_s116
                                 + 0.737325 w_k1_MAT_s116
                                 + 1.12299 w_k1_MSFT_s116
                                 + 1.15497 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.981904 w_k1_ADM_s116
                                 + 0.928964 w_k1_AXP_s116
                                 + 0.759373 w_k1_DIS_s116
                                 + 0.97418 w_k1_FDX_s116
                                 + 0.661854 w_k1_IBM_s116
                                 + 0.838927 w_k1_INTC_s116
                                 + 0.830977 w_k1_KO_s116
                                 + 0.589247 w_k1_MAT_s116
                                 + 0.879698 w_k1_MSFT_s116
                                 + 1.02988 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.818894 w_k1_ADM_s117
                                 + 0.869146 w_k1_AXP_s117
                                 + 0.735535 w_k1_DIS_s117
                                 + 0.840056 w_k1_FDX_s117
                                 + 0.951914 w_k1_IBM_s117
                                 + 0.865127 w_k1_INTC_s117
                                 + 0.763163 w_k1_KO_s117
                                 + 0.656445 w_k1_MAT_s117
                                 + 0.732347 w_k1_MSFT_s117
                                 + 0.837486 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.828821 w_k1_ADM_s117
                                 + 0.80932 w_k1_AXP_s117
                                 + 0.711143 w_k1_DIS_s117
                                 + 0.73619 w_k1_FDX_s117
                                 + 1.60208 w_k1_IBM_s117
                                 + 0.803803 w_k1_INTC_s117
                                 + 1.09302 w_k1_KO_s117 + 1.20279 w_k1_MAT_s117
                                 + 0.750596 w_k1_MSFT_s117
                                 + 1.01068 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.05431 w_k1_ADM_s117
                                 + 1.43747 w_k1_AXP_s117
                                 + 1.62506 w_k1_DIS_s117
                                 + 1.43551 w_k1_FDX_s117
                                 + 1.29226 w_k1_IBM_s117
                                 + 1.21948 w_k1_INTC_s117
                                 + 1.30452 w_k1_KO_s117 + 1.58937 w_k1_MAT_s117
                                 + 1.39875 w_k1_MSFT_s117
                                 + 1.28864 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.91189 w_k1_ADM_s117
                                 + 0.981913 w_k1_AXP_s117
                                 + 1.04213 w_k1_DIS_s117
                                 + 1.00133 w_k1_FDX_s117
                                 + 1.11267 w_k1_IBM_s117
                                 + 0.88547 w_k1_INTC_s117
                                 + 0.99691 w_k1_KO_s117
                                 + 0.932039 w_k1_MAT_s117
                                 + 0.869395 w_k1_MSFT_s117
                                 + 0.976128 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.11614 w_k1_ADM_s117
                                 + 1.24446 w_k1_AXP_s117 + 1.2816 w_k1_DIS_s117
                                 + 1.10256 w_k1_FDX_s117 + 1.4552 w_k1_IBM_s117
                                 + 1.00213 w_k1_INTC_s117 + 1.2718 w_k1_KO_s117
                                 + 1.26769 w_k1_MAT_s117
                                 + 0.995131 w_k1_MSFT_s117
                                 + 1.12084 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.7657 w_k1_ADM_s117
                                 + 0.823651 w_k1_AXP_s117
                                 + 0.74423 w_k1_DIS_s117
                                 + 0.91998 w_k1_FDX_s117
                                 + 0.625508 w_k1_IBM_s117
                                 + 0.719262 w_k1_INTC_s117
                                 + 0.692442 w_k1_KO_s117
                                 + 0.63274 w_k1_MAT_s117
                                 + 0.837166 w_k1_MSFT_s117
                                 + 0.887081 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.17112 w_k1_ADM_s117
                                 + 1.41605 w_k1_AXP_s117
                                 + 1.42511 w_k1_DIS_s117
                                 + 1.42355 w_k1_FDX_s117
                                 + 0.903402 w_k1_IBM_s117
                                 + 1.1446 w_k1_INTC_s117 + 1.19111 w_k1_KO_s117
                                 + 1.17154 w_k1_MAT_s117
                                 + 1.35741 w_k1_MSFT_s117
                                 + 1.13302 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.980532 w_k1_ADM_s117
                                 + 1.28347 w_k1_AXP_s117
                                 + 1.18067 w_k1_DIS_s117
                                 + 1.20124 w_k1_FDX_s117
                                 + 0.883481 w_k1_IBM_s117
                                 + 1.08213 w_k1_INTC_s117
                                 + 0.91107 w_k1_KO_s117
                                 + 0.830458 w_k1_MAT_s117
                                 + 0.992147 w_k1_MSFT_s117
                                 + 1.06175 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00969574821855051 w_k1_ADM_s000
                                 + 0.0082524723832841 w_k1_ADM_s001
                                 + 0.00962096180703565 w_k1_ADM_s002
                                 + 0.0114604186305367 w_k1_ADM_s003
                                 + 0.0167272113642898 w_k1_ADM_s004
                                 + 0.0116158199992957 w_k1_ADM_s005
                                 + 0.00595448747974328 w_k1_ADM_s006
                                 + 0.0138907968430318 w_k1_ADM_s007
                                 + 0.0249653130814301 w_k1_ADM_s010
                                 + 0.0202255915795625 w_k1_ADM_s011
                                 + 0.0268846319652296 w_k1_ADM_s012
                                 + 0.0426911883262954 w_k1_ADM_s013
                                 + 0.0343257250509594 w_k1_ADM_s014
                                 + 0.02980873914681 w_k1_ADM_s015
                                 + 0.0197074604625394 w_k1_ADM_s016
                                 + 0.0287250492163038 w_k1_ADM_s017
                                 + 0.0272157049534937 w_k1_ADM_s100
                                 + 0.0291739899806724 w_k1_ADM_s101
                                 + 0.0520114090359646 w_k1_ADM_s102
                                 + 0.0391092120989465 w_k1_ADM_s103
                                 + 0.0375693389387268 w_k1_ADM_s104
                                 + 0.0877042684109645 w_k1_ADM_s105
                                 + 0.0188566307615459 w_k1_ADM_s106
                                 + 0.0277279852752936 w_k1_ADM_s107
                                 + 0.0460531706332033 w_k1_ADM_s110
                                 + 0.0253586069899235 w_k1_ADM_s111
                                 + 0.0272969928959414 w_k1_ADM_s112
                                 + 0.0956678841551906 w_k1_ADM_s113
                                 + 0.0192746814511553 w_k1_ADM_s114
                                 + 0.0399096371038134 w_k1_ADM_s115
                                 + 0.0589788770099966 w_k1_ADM_s116
                                 + 0.0796011746399064 w_k1_ADM_s117
                                 + 0.00972455433866212 w_k1_AXP_s000
                                 + 0.00772645344987859 w_k1_AXP_s001
                                 + 0.00894485247061253 w_k1_AXP_s002
                                 + 0.0101530060483324 w_k1_AXP_s003
                                 + 0.0147674517219081 w_k1_AXP_s004
                                 + 0.0103070196057406 w_k1_AXP_s005
                                 + 0.00662612515237769 w_k1_AXP_s006
                                 + 0.0143191347359032 w_k1_AXP_s007
                                 + 0.0234541996534741 w_k1_AXP_s010
                                 + 0.0182716702997255 w_k1_AXP_s011
                                 + 0.0326520633166892 w_k1_AXP_s012
                                 + 0.0419301353898301 w_k1_AXP_s013
                                 + 0.0325820029880889 w_k1_AXP_s014
                                 + 0.0284570571632681 w_k1_AXP_s015
                                 + 0.0212839804180433 w_k1_AXP_s016
                                 + 0.0237038783431416 w_k1_AXP_s017
                                 + 0.0276257310476961 w_k1_AXP_s100
                                 + 0.0269400667765962 w_k1_AXP_s101
                                 + 0.0505627671509485 w_k1_AXP_s102
                                 + 0.0393458782398147 w_k1_AXP_s103
                                 + 0.0449777600172155 w_k1_AXP_s104
                                 + 0.0844313162806477 w_k1_AXP_s105
                                 + 0.0184150708390569 w_k1_AXP_s106
                                 + 0.0305015018517649 w_k1_AXP_s107
                                 + 0.0550910769857814 w_k1_AXP_s110
                                 + 0.0285910085849728 w_k1_AXP_s111
                                 + 0.0251416639719371 w_k1_AXP_s112
                                 + 0.0824640267264012 w_k1_AXP_s113
                                 + 0.0186708096682828 w_k1_AXP_s114
                                 + 0.0432872279125977 w_k1_AXP_s115
                                 + 0.0595098642739384 w_k1_AXP_s116
                                 + 0.0905713839646938 w_k1_AXP_s117
                                 + 0.0108187301062162 w_k1_DIS_s000
                                 + 0.00857019030602255 w_k1_DIS_s001
                                 + 0.00909212279147326 w_k1_DIS_s002
                                 + 0.010770991397414 w_k1_DIS_s003
                                 + 0.0143315360078019 w_k1_DIS_s004
                                 + 0.0105951842610086 w_k1_DIS_s005
                                 + 0.00592983103282272 w_k1_DIS_s006
                                 + 0.0151526555420079 w_k1_DIS_s007
                                 + 0.0232180144000722 w_k1_DIS_s010
                                 + 0.0187555986301858 w_k1_DIS_s011
                                 + 0.0337843264040796 w_k1_DIS_s012
                                 + 0.0389814438333305 w_k1_DIS_s013
                                 + 0.0348842548777894 w_k1_DIS_s014
                                 + 0.0299046047055621 w_k1_DIS_s015
                                 + 0.0195637997077527 w_k1_DIS_s016
                                 + 0.0228728987431134 w_k1_DIS_s017
                                 + 0.0274181143223656 w_k1_DIS_s100
                                 + 0.0318966643948405 w_k1_DIS_s101
                                 + 0.0474491195526258 w_k1_DIS_s102
                                 + 0.0392910538289036 w_k1_DIS_s103
                                 + 0.0424765545441955 w_k1_DIS_s104
                                 + 0.0812702588917778 w_k1_DIS_s105
                                 + 0.019188039470674 w_k1_DIS_s106
                                 + 0.0331723399223922 w_k1_DIS_s107
                                 + 0.0570301663462933 w_k1_DIS_s110
                                 + 0.0283890281849272 w_k1_DIS_s111
                                 + 0.0254299495614816 w_k1_DIS_s112
                                 + 0.0907293398566378 w_k1_DIS_s113
                                 + 0.0195917054994014 w_k1_DIS_s114
                                 + 0.0434432245278976 w_k1_DIS_s115
                                 + 0.0597668529196038 w_k1_DIS_s116
                                 + 0.0876230854890947 w_k1_DIS_s117
                                 + 0.010478974277176 w_k1_FDX_s000
                                 + 0.00767311732028452 w_k1_FDX_s001
                                 + 0.00906484955321857 w_k1_FDX_s002
                                 + 0.0105264524408047 w_k1_FDX_s003
                                 + 0.0134034030950653 w_k1_FDX_s004
                                 + 0.0106955722142586 w_k1_FDX_s005
                                 + 0.0058254800224553 w_k1_FDX_s006
                                 + 0.0151909744120289 w_k1_FDX_s007
                                 + 0.0221402416889372 w_k1_FDX_s010
                                 + 0.0176668335202944 w_k1_FDX_s011
                                 + 0.0317565639650273 w_k1_FDX_s012
                                 + 0.040637979422231 w_k1_FDX_s013
                                 + 0.0329325146324825 w_k1_FDX_s014
                                 + 0.0286054894069209 w_k1_FDX_s015
                                 + 0.0203718942965101 w_k1_FDX_s016
                                 + 0.0241003380648195 w_k1_FDX_s017
                                 + 0.0279580597308631 w_k1_FDX_s100
                                 + 0.0290686994470193 w_k1_FDX_s101
                                 + 0.047726944179466 w_k1_FDX_s102
                                 + 0.040118521245957 w_k1_FDX_s103
                                 + 0.0430891410417978 w_k1_FDX_s104
                                 + 0.0760734725501005 w_k1_FDX_s105
                                 + 0.0190117135651497 w_k1_FDX_s106
                                 + 0.0323342849732962 w_k1_FDX_s107
                                 + 0.0541029376702848 w_k1_FDX_s110
                                 + 0.02842891246104 w_k1_FDX_s111
                                 + 0.0245480164316095 w_k1_FDX_s112
                                 + 0.0880627977137484 w_k1_FDX_s113
                                 + 0.0188016916336757 w_k1_FDX_s114
                                 + 0.0423392492757681 w_k1_FDX_s115
                                 + 0.0575761343546273 w_k1_FDX_s116
                                 + 0.0889020453474125 w_k1_FDX_s117
                                 + 0.0106532737161696 w_k1_IBM_s000
                                 + 0.00797093545394905 w_k1_IBM_s001
                                 + 0.00883785794179265 w_k1_IBM_s002
                                 + 0.0111048657816407 w_k1_IBM_s003
                                 + 0.0137322903301703 w_k1_IBM_s004
                                 + 0.0112789092297558 w_k1_IBM_s005
                                 + 0.00595359885062324 w_k1_IBM_s006
                                 + 0.0149009364334736 w_k1_IBM_s007
                                 + 0.0247359620571842 w_k1_IBM_s010
                                 + 0.0180863712536817 w_k1_IBM_s011
                                 + 0.0332374477181748 w_k1_IBM_s012
                                 + 0.0418484916859423 w_k1_IBM_s013
                                 + 0.0347388070957264 w_k1_IBM_s014
                                 + 0.0291054381978603 w_k1_IBM_s015
                                 + 0.0205627776605436 w_k1_IBM_s016
                                 + 0.0244013036500916 w_k1_IBM_s017
                                 + 0.028925521046562 w_k1_IBM_s100
                                 + 0.0305684891095251 w_k1_IBM_s101
                                 + 0.0489980784462647 w_k1_IBM_s102
                                 + 0.0393794479319428 w_k1_IBM_s103
                                 + 0.0439042136232148 w_k1_IBM_s104
                                 + 0.0770495112088488 w_k1_IBM_s105
                                 + 0.019236051386835 w_k1_IBM_s106
                                 + 0.0310143019051972 w_k1_IBM_s107
                                 + 0.0522400727451529 w_k1_IBM_s110
                                 + 0.0290051212788926 w_k1_IBM_s111
                                 + 0.0253277912023415 w_k1_IBM_s112
                                 + 0.0880266405928511 w_k1_IBM_s113
                                 + 0.0183344899947683 w_k1_IBM_s114
                                 + 0.0414771982013184 w_k1_IBM_s115
                                 + 0.0565126493323011 w_k1_IBM_s116
                                 + 0.0895408321310877 w_k1_IBM_s117
                                 + 0.0121204464049373 w_k1_INTC_s000
                                 + 0.00615165203370427 w_k1_INTC_s001
                                 + 0.00825958820003134 w_k1_INTC_s002
                                 + 0.0110021037693695 w_k1_INTC_s003
                                 + 0.0136792107020337 w_k1_INTC_s004
                                 + 0.00988421774791497 w_k1_INTC_s005
                                 + 0.00723958431024761 w_k1_INTC_s006
                                 + 0.0132255854455872 w_k1_INTC_s007
                                 + 0.0223792013908454 w_k1_INTC_s010
                                 + 0.0176556153730434 w_k1_INTC_s011
                                 + 0.0318202768783422 w_k1_INTC_s012
                                 + 0.03956733442104 w_k1_INTC_s013
                                 + 0.0348486290330614 w_k1_INTC_s014
                                 + 0.0260250622773087 w_k1_INTC_s015
                                 + 0.0207885210650651 w_k1_INTC_s016
                                 + 0.0221748456841179 w_k1_INTC_s017
                                 + 0.0253093385572583 w_k1_INTC_s100
                                 + 0.0255413574216266 w_k1_INTC_s101
                                 + 0.0444172152277159 w_k1_INTC_s102
                                 + 0.041351323374318 w_k1_INTC_s103
                                 + 0.0443769365795759 w_k1_INTC_s104
                                 + 0.0781169971992565 w_k1_INTC_s105
                                 + 0.017751319582384 w_k1_INTC_s106
                                 + 0.0296475413010229 w_k1_INTC_s107
                                 + 0.0517852977031934 w_k1_INTC_s110
                                 + 0.030181142565532 w_k1_INTC_s111
                                 + 0.0271115935114806 w_k1_INTC_s112
                                 + 0.0915824661067987 w_k1_INTC_s113
                                 + 0.0183212759188196 w_k1_INTC_s114
                                 + 0.042677998805773 w_k1_INTC_s115
                                 + 0.0526250836099422 w_k1_INTC_s116
                                 + 0.0801859363712105 w_k1_INTC_s117
                                 + 0.0109203374576536 w_k1_KO_s000
                                 + 0.00843754944408354 w_k1_KO_s001
                                 + 0.00908332052785604 w_k1_KO_s002
                                 + 0.00986011905256582 w_k1_KO_s003
                                 + 0.0150672148624148 w_k1_KO_s004
                                 + 0.0104331224418568 w_k1_KO_s005
                                 + 0.00635546324863451 w_k1_KO_s006
                                 + 0.0139612641289022 w_k1_KO_s007
                                 + 0.0242057966892067 w_k1_KO_s010
                                 + 0.0194329011750046 w_k1_KO_s011
                                 + 0.0355983873486349 w_k1_KO_s012
                                 + 0.0385531433844868 w_k1_KO_s013
                                 + 0.0356701945810666 w_k1_KO_s014
                                 + 0.0290921429096197 w_k1_KO_s015
                                 + 0.020219076499814 w_k1_KO_s016
                                 + 0.0213838751670449 w_k1_KO_s017
                                 + 0.0292937556027678 w_k1_KO_s100
                                 + 0.0290254463188176 w_k1_KO_s101
                                 + 0.051286308735596 w_k1_KO_s102
                                 + 0.0392772056120817 w_k1_KO_s103
                                 + 0.0414970725872266 w_k1_KO_s104
                                 + 0.0813276768536089 w_k1_KO_s105
                                 + 0.0184761439642321 w_k1_KO_s106
                                 + 0.0306077043256525 w_k1_KO_s107
                                 + 0.0555824422282015 w_k1_KO_s110
                                 + 0.0291286057202806 w_k1_KO_s111
                                 + 0.0281664042322303 w_k1_KO_s112
                                 + 0.088336984936728 w_k1_KO_s113
                                 + 0.0187674970751456 w_k1_KO_s114
                                 + 0.0466058904567918 w_k1_KO_s115
                                 + 0.0595125628146978 w_k1_KO_s116
                                 + 0.0822396683851913 w_k1_KO_s117
                                 + 0.0109032041445253 w_k1_MAT_s000
                                 + 0.00815263573387993 w_k1_MAT_s001
                                 + 0.00853984972495849 w_k1_MAT_s002
                                 + 0.00957710830184356 w_k1_MAT_s003
                                 + 0.014324486235976 w_k1_MAT_s004
                                 + 0.00984538743047503 w_k1_MAT_s005
                                 + 0.00546499986550755 w_k1_MAT_s006
                                 + 0.016464281294326 w_k1_MAT_s007
                                 + 0.0221654219489395 w_k1_MAT_s010
                                 + 0.018596018624365 w_k1_MAT_s011
                                 + 0.0343366668521224 w_k1_MAT_s012
                                 + 0.0370184578441216 w_k1_MAT_s013
                                 + 0.0424780815926323 w_k1_MAT_s014
                                 + 0.0415288645270534 w_k1_MAT_s015
                                 + 0.0202299574753034 w_k1_MAT_s016
                                 + 0.028129472936807 w_k1_MAT_s017
                                 + 0.0414891995342996 w_k1_MAT_s100
                                 + 0.0284615467076972 w_k1_MAT_s101
                                 + 0.0517481845851494 w_k1_MAT_s102
                                 + 0.0364873850819541 w_k1_MAT_s103
                                 + 0.0491652860898864 w_k1_MAT_s104
                                 + 0.0892007014384788 w_k1_MAT_s105
                                 + 0.0184956722084178 w_k1_MAT_s106
                                 + 0.0346942681885398 w_k1_MAT_s107
                                 + 0.0692852773207801 w_k1_MAT_s110
                                 + 0.026956661927499 w_k1_MAT_s111
                                 + 0.0213856410484622 w_k1_MAT_s112
                                 + 0.0746440869169767 w_k1_MAT_s113
                                 + 0.0176863510557592 w_k1_MAT_s114
                                 + 0.0422709833956365 w_k1_MAT_s115
                                 + 0.0420005680255552 w_k1_MAT_s116
                                 + 0.0788968301881188 w_k1_MAT_s117
                                 + 0.0127757925532559 w_k1_MSFT_s000
                                 + 0.00782986470440042 w_k1_MSFT_s001
                                 + 0.0077527660794453 w_k1_MSFT_s002
                                 + 0.0108049608151464 w_k1_MSFT_s003
                                 + 0.0130741751702457 w_k1_MSFT_s004
                                 + 0.0107976563055335 w_k1_MSFT_s005
                                 + 0.0062994101294779 w_k1_MSFT_s006
                                 + 0.0162581651018403 w_k1_MSFT_s007
                                 + 0.0215180836460916 w_k1_MSFT_s010
                                 + 0.0176123980456674 w_k1_MSFT_s011
                                 + 0.0307574061118207 w_k1_MSFT_s012
                                 + 0.0399765012315015 w_k1_MSFT_s013
                                 + 0.0345612547742104 w_k1_MSFT_s014
                                 + 0.0257459273905268 w_k1_MSFT_s015
                                 + 0.0185880912344112 w_k1_MSFT_s016
                                 + 0.0241035292743687 w_k1_MSFT_s017
                                 + 0.0303013239010346 w_k1_MSFT_s100
                                 + 0.0283963585967686 w_k1_MSFT_s101
                                 + 0.0432719402812221 w_k1_MSFT_s102
                                 + 0.0371708153515715 w_k1_MSFT_s103
                                 + 0.0436358158767957 w_k1_MSFT_s104
                                 + 0.0765503589716785 w_k1_MSFT_s105
                                 + 0.0179102805748612 w_k1_MSFT_s106
                                 + 0.0322166863513137 w_k1_MSFT_s107
                                 + 0.0521346267555382 w_k1_MSFT_s110
                                 + 0.0279838393353582 w_k1_MSFT_s111
                                 + 0.0257011150645149 w_k1_MSFT_s112
                                 + 0.0819553920366403 w_k1_MSFT_s113
                                 + 0.0188165568291209 w_k1_MSFT_s114
                                 + 0.0437154131599537 w_k1_MSFT_s115
                                 + 0.0556655136351808 w_k1_MSFT_s116
                                 + 0.0791509638483699 w_k1_MSFT_s117
                                 + 0.00997676726243312 w_k1_PG_s000
                                 + 0.00810180822281858 w_k1_PG_s001
                                 + 0.0091634414556741 w_k1_PG_s002
                                 + 0.0110506921154952 w_k1_PG_s003
                                 + 0.0126742659122725 w_k1_PG_s004
                                 + 0.00968398658423786 w_k1_PG_s005
                                 + 0.00550263829931194 w_k1_PG_s006
                                 + 0.0139404248103195 w_k1_PG_s007
                                 + 0.0239658814399885 w_k1_PG_s010
                                 + 0.0181706145843473 w_k1_PG_s011
                                 + 0.0335394942124062 w_k1_PG_s012
                                 + 0.0417295647211945 w_k1_PG_s013
                                 + 0.0335583422548541 w_k1_PG_s014
                                 + 0.0312184970701624 w_k1_PG_s015
                                 + 0.0208489539543943 w_k1_PG_s016
                                 + 0.0247574978017211 w_k1_PG_s017
                                 + 0.0282269829414519 w_k1_PG_s100
                                 + 0.0297935688210372 w_k1_PG_s101
                                 + 0.0449209301210245 w_k1_PG_s102
                                 + 0.0383906231117473 w_k1_PG_s103
                                 + 0.0419907441814008 w_k1_PG_s104
                                 + 0.0772040781428109 w_k1_PG_s105
                                 + 0.0189898179586368 w_k1_PG_s106
                                 + 0.0334292248775451 w_k1_PG_s107
                                 + 0.0543308700711091 w_k1_PG_s110
                                 + 0.0283704510444401 w_k1_PG_s111
                                 + 0.025938889261343 w_k1_PG_s112
                                 + 0.0825433371730056 w_k1_PG_s113
                                 + 0.0178054046498237 w_k1_PG_s114
                                 + 0.0429971253910978 w_k1_PG_s115
                                 + 0.0573537304920066 w_k1_PG_s116
                                 + 0.0856527955355918 w_k1_PG_s117 >= 1.229
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
