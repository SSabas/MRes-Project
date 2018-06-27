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
 w_k1_ADM_s0_return_constraint:  1.07306 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.890326 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.859558 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.12643 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.808464 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.02476 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.969816 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.59808 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.11576 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.21151 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.18246 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.11437 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.27759 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.895731 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.09832 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.788031 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.03108 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.858802 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.802264 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.39469 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.23716 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.19716 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.53957 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.02741 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.55314 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.15844 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.17835 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.874233 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.37184 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.951176 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.10154 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.935309 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.783301 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.28989 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.17058 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.40871 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.2912 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.19568 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.14922 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.863022 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.869382 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.21166 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.780643 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.13416 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.89909 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.51284 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.977354 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.14125 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.705113 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.39955 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.949203 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.703432 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.01171 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.661724 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.10705 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.790844 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.0954 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.73185 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.3157 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.03509 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.62309 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.830351 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.25743 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.2122 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.977129 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.668341 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.12778 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.962433 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.946525 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.803667 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.757274 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.32349 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.18652 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.60203 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.11485 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.11384 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.20121 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.806334 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.841936 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.159 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.769018 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.22852 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.783342 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.54226 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.942814 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.4071 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.859588 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.972502 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.985124 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.690272 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.924713 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.690535 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.4117 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.997912 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.887071 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.87015 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.57661 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.19382 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.66602 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.787215 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.31625 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.65583 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.02485 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.548641 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.13892 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.901092 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.00239 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.06878 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.647561 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.28445 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.792662 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.78369 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.30764 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.98128 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10531 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.870518 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.920937 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.09996 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.867165 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.12494 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.940061 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.58668 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.870009 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.2002 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.03669 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.13344 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.967679 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.73857 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.02431 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.735089 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.19625 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.902863 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.878519 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.5842 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.20542 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.24038 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.2712 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.895629 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.30711 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.3086 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.995579 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.673628 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.25257 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.903745 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.04407 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.980182 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.728607 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.985011 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.969503 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.43473 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.25168 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.95179 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14152 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.858148 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.742217 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.2406 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.774821 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.21429 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.620634 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.915144 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.47575 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.19261 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.602219 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.99539 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.02352 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.955122 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.881484 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.89159 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.08168 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.823642 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.37224 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.07739 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.80211 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.681202 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.5871 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.733395 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.899588 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.25072 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.929594 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.854571 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.735518 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.37554 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.6619 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.24229 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.985908 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.76312 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.81434 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.21408 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.94195 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.28289 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10539 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.877408 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.893533 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.08754 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.941819 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.17654 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.921664 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.22319 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.891343 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.14304 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.06673 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.1494 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.911084 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.80857 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.919668 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.692674 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.20597 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.799147 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.944567 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.43401 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.10888 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.07486 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.22882 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.998049 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.08068 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.3137 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.06496 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.731993 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.0672 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.958792 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.902557 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.915392 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.848719 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.930079 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.89555 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.17256 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.15832 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.957514 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.06708 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.863799 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.844019 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.17127 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.785559 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.17469 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.767523 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.19032 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.22957 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.26025 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.77836 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.938362 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.05121 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.860196 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.975539 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.845019 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.26474 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.89937 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.21351 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.26996 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.4848 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.974774 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.428 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.805499 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.12171 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.51807 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.12176 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.7869 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.888988 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.12947 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.889984 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.14917 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.826964 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.43365 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.855192 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.26962 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.09339 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.10666 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.11505 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.880895 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.827991 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.30148 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.662717 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.17984 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.653895 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.16844 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.22902 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.0128 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.407573 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.949579 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.792645 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.813018 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.833064 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.810723 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.16501 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.892292 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.41414 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.17649 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.80517 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.762989 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.54732 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.688145 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.94074 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.34771 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.972612 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.719467 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.937759 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.59565 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.60057 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.945554 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.911486 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.87187 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.973743 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.54172 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.761681 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.2487 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.0845 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.888927 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.918334 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.09699 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.917366 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.153 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.938063 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.38644 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.916676 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.25523 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.00459 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.99672 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.962676 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.807932 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.964755 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.674787 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.16658 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.917436 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.85655 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.31163 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.21565 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.16281 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.16163 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.874056 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.23229 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.29048 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.06659 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.691639 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.05649 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.882353 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.995735 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.03414 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.709104 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.07507 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.891366 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.1971 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.08181 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.934628 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09285 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.923107 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.910435 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10055 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.852182 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.07301 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.877303 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.19602 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.980345 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.18041 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.92162 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.987894 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.00257 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.892745 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.951712 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.828519 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00937 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.888734 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.00252 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.14875 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.17056 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.985051 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.27809 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.874985 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.17403 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.30761 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.997285 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.818892 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.980862 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.03715 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.951287 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.999857 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.833611 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.21994 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.92131 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.24216 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.13968 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.06031 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.269000464647836 z_k1_s0000
                                 - 0.162999891473027 z_k1_s0001
                                 - 0.0959998729491729 z_k1_s0002
                                 - 0.337000030224488 z_k1_s0003
                                 - 0.180999836019761 z_k1_s0004
                                 - 0.316000263619889 z_k1_s0005
                                 - 0.0919999977387372 z_k1_s0006
                                 - 0.23299990075467 z_k1_s0007
                                 - 0.135000077398522 z_k1_s0010
                                 - 0.417998785249087 z_k1_s0011
                                 - 0.311001049503065 z_k1_s0012
                                 - 0.495000607794508 z_k1_s0013
                                 - 0.352000000209526 z_k1_s0014
                                 - 0.285000055396905 z_k1_s0015
                                 - 0.172999922784813 z_k1_s0016
                                 - 0.260998950837415 z_k1_s0017
                                 - 0.369000087259589 z_k1_s0020
                                 - 0.470001218572626 z_k1_s0021
                                 - 0.417001567731805 z_k1_s0022
                                 - 0.402999042734247 z_k1_s0023
                                 - 0.482999010852821 z_k1_s0024
                                 - 0.490999382564946 z_k1_s0025
                                 - 0.88500081887506 z_k1_s0026
                                 - 0.231000048529983 z_k1_s0027
                                 - 0.296000310718131 z_k1_s0030
                                 - 0.544001153099226 z_k1_s0031
                                 - 1.19100365338029 z_k1_s0032
                                 - 1.0240030150824 z_k1_s0033
                                 - 0.785000890174891 z_k1_s0034
                                 - 0.386000537477258 z_k1_s0035
                                 - 1.1970010580929 z_k1_s0036
                                 - 1.10300038618814 z_k1_s0037
                                 - 0.143000123625952 z_k1_s0040
                                 - 0.124000075007803 z_k1_s0041
                                 - 0.0980000525303573 z_k1_s0042
                                 - 0.074999978992591 z_k1_s0043
                                 - 0.339000507686801 z_k1_s0044
                                 - 0.198999635500193 z_k1_s0045
                                 - 0.292000145131263 z_k1_s0046
                                 - 0.125000011487674 z_k1_s0047
                                 - 0.103999968531852 z_k1_s0050
                                 - 0.166999972380941 z_k1_s0051
                                 - 0.126000007698178 z_k1_s0052
                                 - 0.336000133995076 z_k1_s0053
                                 - 0.466000520159643 z_k1_s0054
                                 - 0.161999839697757 z_k1_s0055
                                 - 0.144999933214615 z_k1_s0056
                                 - 0.196000503663546 z_k1_s0057
                                 - 0.15600004439847 z_k1_s0060
                                 - 0.312999317395008 z_k1_s0061
                                 - 0.304000721412402 z_k1_s0062
                                 - 0.24600017652451 z_k1_s0063
                                 - 0.0749999743550337 z_k1_s0064
                                 - 0.088000045929906 z_k1_s0065
                                 - 0.216000512582496 z_k1_s0066
                                 - 0.231000344553503 z_k1_s0067
                                 - 1.42200036652387 z_k1_s0070
                                 - 0.838001424805333 z_k1_s0071
                                 - 1.25099835156423 z_k1_s0072
                                 - 0.405999776166798 z_k1_s0073
                                 - 0.417999251062917 z_k1_s0074
                                 - 0.47599930805808 z_k1_s0075
                                 - 0.843998982455701 z_k1_s0076
                                 - 0.572999682563487 z_k1_s0077
                                 - 0.655998920968254 z_k1_s0100
                                 - 0.643998742072577 z_k1_s0101
                                 - 0.674001225310902 z_k1_s0102
                                 - 0.15599988244483 z_k1_s0103
                                 - 0.49100155115054 z_k1_s0104
                                 - 0.239999913115084 z_k1_s0105
                                 - 0.495000253449676 z_k1_s0106
                                 - 0.715999815446635 z_k1_s0107
                                 - 1.14500432634199 z_k1_s0110
                                 - 0.452000343598466 z_k1_s0111
                                 - 0.944001007244831 z_k1_s0112
                                 - 1.46900582774905 z_k1_s0113
                                 - 1.98000146125873 z_k1_s0114
                                 - 2.17600114507182 z_k1_s0115
                                 - 0.935000414538608 z_k1_s0116
                                 - 0.817000908508535 z_k1_s0117
                                 - 0.270000291848558 z_k1_s0120
                                 - 0.470998725741912 z_k1_s0121
                                 - 0.238000060532247 z_k1_s0122
                                 - 0.569998948223159 z_k1_s0123
                                 - 0.223000108468499 z_k1_s0124
                                 - 0.95400249381962 z_k1_s0125
                                 - 0.304000465453321 z_k1_s0126
                                 - 0.819002540345522 z_k1_s0127
                                 - 0.706000915476527 z_k1_s0130
                                 - 0.255000779362281 z_k1_s0131
                                 - 0.793000854533363 z_k1_s0132
                                 - 0.314000758768697 z_k1_s0133
                                 - 0.222000479001917 z_k1_s0134
                                 - 0.268000799285795 z_k1_s0135
                                 - 0.642002038564067 z_k1_s0136
                                 - 0.40800095833887 z_k1_s0137
                                 - 0.200000144479195 z_k1_s0140
                                 - 0.424000428371882 z_k1_s0141
                                 - 0.48999945535911 z_k1_s0142
                                 - 0.527998997897201 z_k1_s0143
                                 - 0.507998881719291 z_k1_s0144
                                 - 0.495001179903433 z_k1_s0145
                                 - 0.522000319443704 z_k1_s0146
                                 - 0.223999944792718 z_k1_s0147
                                 - 0.591998592783746 z_k1_s0150
                                 - 0.475000628847467 z_k1_s0151
                                 - 0.396000040780473 z_k1_s0152
                                 - 0.28099976613326 z_k1_s0153
                                 - 0.333999579259541 z_k1_s0154
                                 - 0.128999797461941 z_k1_s0155
                                 - 0.431000301316483 z_k1_s0156
                                 - 0.268999967506466 z_k1_s0157
                                 - 0.349000507362536 z_k1_s0160
                                 - 0.646998576154463 z_k1_s0161
                                 - 0.421000441110753 z_k1_s0162
                                 - 0.324999866646742 z_k1_s0163
                                 - 0.574001128825365 z_k1_s0164
                                 - 0.351999676102393 z_k1_s0165
                                 - 0.45899874914523 z_k1_s0166
                                 - 0.186999993629327 z_k1_s0167
                                 - 0.35299945007426 z_k1_s0170
                                 - 0.461998901190569 z_k1_s0171
                                 - 0.682997929968419 z_k1_s0172
                                 - 0.219999778395142 z_k1_s0173
                                 - 0.775001018889784 z_k1_s0174
                                 - 0.507998466153664 z_k1_s0175
                                 - 0.35099976161029 z_k1_s0176
                                 - 0.607001058548198 z_k1_s0177
                                 - 0.268000805833351 z_k1_s1000
                                 - 0.24299988663183 z_k1_s1001
                                 - 0.206000208214872 z_k1_s1002
                                 - 0.106000011411462 z_k1_s1003
                                 - 0.25999969968824 z_k1_s1004
                                 - 0.0910000398909955 z_k1_s1005
                                 - 0.0830000937467762 z_k1_s1006
                                 - 0.192999498229903 z_k1_s1007
                                 - 0.160000513433514 z_k1_s1010
                                 - 0.187000723358964 z_k1_s1011
                                 - 0.197000636141902 z_k1_s1012
                                 - 0.107000360591846 z_k1_s1013
                                 - 0.452001487567302 z_k1_s1014
                                 - 0.151000443458363 z_k1_s1015
                                 - 0.475001689959433 z_k1_s1016
                                 - 0.392000314063891 z_k1_s1017
                                 - 0.137000293515223 z_k1_s1020
                                 - 0.297000811577107 z_k1_s1021
                                 - 0.301000058627304 z_k1_s1022
                                 - 0.0890002208968042 z_k1_s1023
                                 - 0.232999610750602 z_k1_s1024
                                 - 0.339000357617312 z_k1_s1025
                                 - 0.11500012913155 z_k1_s1026
                                 - 0.145000278018245 z_k1_s1027
                                 - 0.150000351168098 z_k1_s1030
                                 - 0.150999807828857 z_k1_s1031
                                 - 0.103000040095294 z_k1_s1032
                                 - 0.0500000811560088 z_k1_s1033
                                 - 0.168000264068184 z_k1_s1034
                                 - 0.14299984653993 z_k1_s1035
                                 - 0.101000049773062 z_k1_s1036
                                 - 0.211000594496524 z_k1_s1037
                                 - 0.358999312978955 z_k1_s1040
                                 - 0.114999707158401 z_k1_s1041
                                 - 0.191999375133603 z_k1_s1042
                                 - 0.500998032441918 z_k1_s1043
                                 - 0.189999404318595 z_k1_s1044
                                 - 0.348999458903914 z_k1_s1045
                                 - 0.110999567629036 z_k1_s1046
                                 - 0.161999417109786 z_k1_s1047
                                 - 0.377999002510607 z_k1_s1050
                                 - 0.155999763842255 z_k1_s1051
                                 - 0.120999832113525 z_k1_s1052
                                 - 0.150999711652513 z_k1_s1053
                                 - 0.431000032681276 z_k1_s1054
                                 - 0.360000288865706 z_k1_s1055
                                 - 0.432998926198572 z_k1_s1056
                                 - 0.137999749398969 z_k1_s1057
                                 - 0.149000326073483 z_k1_s1060
                                 - 0.252000676513601 z_k1_s1061
                                 - 0.0860000948450538 z_k1_s1062
                                 - 0.176999987241466 z_k1_s1063
                                 - 0.0760000029421025 z_k1_s1064
                                 - 0.0840001013044752 z_k1_s1065
                                 - 0.175000366301063 z_k1_s1066
                                 - 0.242999898280618 z_k1_s1067
                                 - 0.281000598203393 z_k1_s1070
                                 - 0.414000837883936 z_k1_s1071
                                 - 0.277000246704949 z_k1_s1072
                                 - 1.13100093645764 z_k1_s1073
                                 - 0.261000585113874 z_k1_s1074
                                 - 0.785003021342533 z_k1_s1075
                                 - 0.528000171123049 z_k1_s1076
                                 - 0.684002104844141 z_k1_s1077
                                 - 0.503999174184059 z_k1_s1100
                                 - 0.343000329903861 z_k1_s1101
                                 - 0.255000321855711 z_k1_s1102
                                 - 0.287000097327556 z_k1_s1103
                                 - 0.558000723919574 z_k1_s1104
                                 - 0.522001080763844 z_k1_s1105
                                 - 0.653001751415675 z_k1_s1106
                                 - 1.0480011952108 z_k1_s1107
                                 - 0.11599997474393 z_k1_s1110
                                 - 0.11299995452643 z_k1_s1111
                                 - 0.17599984109433 z_k1_s1112
                                 - 0.224000164574323 z_k1_s1113
                                 - 0.212000083704325 z_k1_s1114
                                 - 0.126999914374864 z_k1_s1115
                                 - 0.14700031815799 z_k1_s1116
                                 - 0.230000205009322 z_k1_s1117
                                 - 0.670002318696818 z_k1_s1120
                                 - 1.47899710977046 z_k1_s1121
                                 - 0.322999840032312 z_k1_s1122
                                 - 1.0240008442489 z_k1_s1123
                                 - 1.13799709946278 z_k1_s1124
                                 - 0.37699974795881 z_k1_s1125
                                 - 0.395999525760855 z_k1_s1126
                                 - 0.621999453030805 z_k1_s1127
                                 - 0.130999939357717 z_k1_s1130
                                 - 0.470999510838083 z_k1_s1131
                                 - 0.180999921201883 z_k1_s1132
                                 - 0.29399936156776 z_k1_s1133
                                 - 0.249000837837839 z_k1_s1134
                                 - 0.138999919020786 z_k1_s1135
                                 - 0.160000029061322 z_k1_s1136
                                 - 0.553999790117363 z_k1_s1137
                                 - 0.536999826865391 z_k1_s1140
                                 - 0.889997268464648 z_k1_s1141
                                 - 0.292999853835358 z_k1_s1142
                                 - 0.717999953145719 z_k1_s1143
                                 - 0.208999661569484 z_k1_s1144
                                 - 0.31099952676413 z_k1_s1145
                                 - 0.312999578556127 z_k1_s1146
                                 - 0.695001142235959 z_k1_s1147
                                 - 0.191000003478503 z_k1_s1150
                                 - 0.355999008944071 z_k1_s1151
                                 - 0.136999982544263 z_k1_s1152
                                 - 0.278000248905921 z_k1_s1153
                                 - 0.226000369880379 z_k1_s1154
                                 - 0.482999975545109 z_k1_s1155
                                 - 0.130000120963921 z_k1_s1156
                                 - 0.316000404770778 z_k1_s1157
                                 - 0.733003350465258 z_k1_s1160
                                 - 0.195000786874761 z_k1_s1161
                                 - 0.169000823738729 z_k1_s1162
                                 - 0.423002426722155 z_k1_s1163
                                 - 0.173000690828347 z_k1_s1164
                                 - 0.273000912583863 z_k1_s1165
                                 - 0.210000938288591 z_k1_s1166
                                 - 0.187000816394521 z_k1_s1167
                                 - 0.190999450622256 z_k1_s1170
                                 - 0.173999647477868 z_k1_s1171
                                 - 0.294000086497093 z_k1_s1172
                                 - 0.106000011000312 z_k1_s1173
                                 - 0.0970000487238711 z_k1_s1174
                                 - 0.278999678369687 z_k1_s1175
                                 - 0.121999975336208 z_k1_s1176
                                 - 0.149999842274026 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.17195 w_k1_ADM_s000
                                 + 1.24262 w_k1_AXP_s000
                                 + 1.15145 w_k1_DIS_s000
                                 + 1.13738 w_k1_FDX_s000
                                 + 1.32803 w_k1_IBM_s000
                                 + 1.03603 w_k1_INTC_s000
                                 + 1.21464 w_k1_KO_s000 + 1.10427 w_k1_MAT_s000
                                 + 1.04834 w_k1_MSFT_s000
                                 + 1.01208 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.55361 w_k1_ADM_s000
                                 + 1.6335 w_k1_AXP_s000 + 1.64246 w_k1_DIS_s000
                                 + 1.45259 w_k1_FDX_s000
                                 + 1.42768 w_k1_IBM_s000
                                 + 1.45528 w_k1_INTC_s000
                                 + 1.29358 w_k1_KO_s000 + 1.14988 w_k1_MAT_s000
                                 + 1.279 w_k1_MSFT_s000 + 1.28527 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.725817 w_k1_ADM_s000
                                 + 0.633798 w_k1_AXP_s000
                                 + 0.627859 w_k1_DIS_s000
                                 + 0.757532 w_k1_FDX_s000
                                 + 0.579778 w_k1_IBM_s000
                                 + 0.851822 w_k1_INTC_s000
                                 + 0.800328 w_k1_KO_s000
                                 + 0.653291 w_k1_MAT_s000
                                 + 0.871625 w_k1_MSFT_s000
                                 + 0.782717 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.978366 w_k1_ADM_s000
                                 + 1.09395 w_k1_AXP_s000
                                 + 0.891055 w_k1_DIS_s000
                                 + 1.03921 w_k1_FDX_s000
                                 + 0.934164 w_k1_IBM_s000
                                 + 1.18271 w_k1_INTC_s000
                                 + 1.05144 w_k1_KO_s000
                                 + 0.931413 w_k1_MAT_s000
                                 + 0.901466 w_k1_MSFT_s000
                                 + 0.93727 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.911917 w_k1_ADM_s000
                                 + 0.78212 w_k1_AXP_s000
                                 + 0.573216 w_k1_DIS_s000
                                 + 0.764554 w_k1_FDX_s000
                                 + 0.777372 w_k1_IBM_s000
                                 + 0.763784 w_k1_INTC_s000
                                 + 0.630828 w_k1_KO_s000
                                 + 0.787995 w_k1_MAT_s000
                                 + 0.716294 w_k1_MSFT_s000
                                 + 0.683552 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.02591 w_k1_ADM_s000
                                 + 1.02641 w_k1_AXP_s000
                                 + 0.903465 w_k1_DIS_s000
                                 + 0.9492 w_k1_FDX_s000 + 1.23776 w_k1_IBM_s000
                                 + 1.06691 w_k1_INTC_s000
                                 + 1.07352 w_k1_KO_s000 + 1.32287 w_k1_MAT_s000
                                 + 0.874236 w_k1_MSFT_s000
                                 + 0.94308 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.38916 w_k1_ADM_s000
                                 + 1.56236 w_k1_AXP_s000 + 1.1995 w_k1_DIS_s000
                                 + 1.36354 w_k1_FDX_s000
                                 + 0.804958 w_k1_IBM_s000
                                 + 1.26636 w_k1_INTC_s000
                                 + 0.767228 w_k1_KO_s000
                                 + 0.76358 w_k1_MAT_s000
                                 + 1.0557 w_k1_MSFT_s000
                                 + 0.950817 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.793743 w_k1_ADM_s000
                                 + 0.814081 w_k1_AXP_s000
                                 + 0.729747 w_k1_DIS_s000
                                 + 0.813907 w_k1_FDX_s000
                                 + 1.15212 w_k1_IBM_s000
                                 + 0.968557 w_k1_INTC_s000
                                 + 0.872092 w_k1_KO_s000
                                 + 0.986687 w_k1_MAT_s000
                                 + 0.861678 w_k1_MSFT_s000
                                 + 0.914079 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.26566 w_k1_ADM_s001
                                 + 1.1393 w_k1_AXP_s001 + 1.16031 w_k1_DIS_s001
                                 + 1.21621 w_k1_FDX_s001
                                 + 0.633223 w_k1_IBM_s001
                                 + 1.17979 w_k1_INTC_s001
                                 + 0.949682 w_k1_KO_s001
                                 + 0.566567 w_k1_MAT_s001
                                 + 1.1258 w_k1_MSFT_s001
                                 + 0.988957 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.28794 w_k1_ADM_s001
                                 + 1.34006 w_k1_AXP_s001
                                 + 1.69367 w_k1_DIS_s001
                                 + 1.33405 w_k1_FDX_s001
                                 + 1.26423 w_k1_IBM_s001
                                 + 1.22337 w_k1_INTC_s001
                                 + 1.37407 w_k1_KO_s001 + 1.61631 w_k1_MAT_s001
                                 + 1.32308 w_k1_MSFT_s001
                                 + 1.25988 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.847832 w_k1_ADM_s001
                                 + 0.879729 w_k1_AXP_s001
                                 + 1.0764 w_k1_DIS_s001
                                 + 0.966483 w_k1_FDX_s001
                                 + 0.742244 w_k1_IBM_s001
                                 + 1.07135 w_k1_INTC_s001
                                 + 0.947588 w_k1_KO_s001
                                 + 0.565175 w_k1_MAT_s001
                                 + 1.08163 w_k1_MSFT_s001
                                 + 0.997628 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.836537 w_k1_ADM_s001
                                 + 0.733008 w_k1_AXP_s001
                                 + 0.722128 w_k1_DIS_s001
                                 + 0.776458 w_k1_FDX_s001
                                 + 0.623174 w_k1_IBM_s001
                                 + 0.941913 w_k1_INTC_s001
                                 + 0.698492 w_k1_KO_s001
                                 + 0.505162 w_k1_MAT_s001
                                 + 0.891792 w_k1_MSFT_s001
                                 + 0.829675 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.04854 w_k1_ADM_s001
                                 + 1.02169 w_k1_AXP_s001
                                 + 1.30331 w_k1_DIS_s001
                                 + 1.08372 w_k1_FDX_s001
                                 + 1.20931 w_k1_IBM_s001
                                 + 1.20989 w_k1_INTC_s001
                                 + 1.24193 w_k1_KO_s001
                                 + 0.795764 w_k1_MAT_s001
                                 + 1.201 w_k1_MSFT_s001 + 1.07344 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.957817 w_k1_ADM_s001
                                 + 1.23054 w_k1_AXP_s001
                                 + 0.973792 w_k1_DIS_s001
                                 + 1.02047 w_k1_FDX_s001 + 1.3135 w_k1_IBM_s001
                                 + 1.10404 w_k1_INTC_s001
                                 + 1.14954 w_k1_KO_s001 + 1.53326 w_k1_MAT_s001
                                 + 1.07125 w_k1_MSFT_s001
                                 + 1.05852 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.07399 w_k1_ADM_s001
                                 + 1.1064 w_k1_AXP_s001
                                 + 0.846402 w_k1_DIS_s001
                                 + 1.01211 w_k1_FDX_s001
                                 + 0.797075 w_k1_IBM_s001
                                 + 0.993564 w_k1_INTC_s001
                                 + 0.745996 w_k1_KO_s001
                                 + 1.09249 w_k1_MAT_s001
                                 + 1.00741 w_k1_MSFT_s001
                                 + 0.927776 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.852584 w_k1_ADM_s001
                                 + 0.814917 w_k1_AXP_s001
                                 + 0.647193 w_k1_DIS_s001
                                 + 0.711332 w_k1_FDX_s001
                                 + 1.01022 w_k1_IBM_s001
                                 + 0.853799 w_k1_INTC_s001
                                 + 0.858448 w_k1_KO_s001
                                 + 0.875223 w_k1_MAT_s001
                                 + 0.715849 w_k1_MSFT_s001
                                 + 0.795272 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.63864 w_k1_ADM_s002
                                 + 0.733285 w_k1_AXP_s002
                                 + 0.662616 w_k1_DIS_s002
                                 + 0.815311 w_k1_FDX_s002
                                 + 0.690937 w_k1_IBM_s002
                                 + 0.88859 w_k1_INTC_s002
                                 + 0.770759 w_k1_KO_s002
                                 + 0.837305 w_k1_MAT_s002
                                 + 0.896806 w_k1_MSFT_s002
                                 + 0.845892 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.87082 w_k1_ADM_s002
                                 + 1.07403 w_k1_AXP_s002
                                 + 0.730482 w_k1_DIS_s002
                                 + 1.03231 w_k1_FDX_s002
                                 + 0.642229 w_k1_IBM_s002
                                 + 0.972601 w_k1_INTC_s002
                                 + 0.654582 w_k1_KO_s002
                                 + 0.748825 w_k1_MAT_s002
                                 + 0.916918 w_k1_MSFT_s002
                                 + 0.977757 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.971229 w_k1_ADM_s002
                                 + 1.16151 w_k1_AXP_s002
                                 + 1.15117 w_k1_DIS_s002
                                 + 1.01689 w_k1_FDX_s002
                                 + 1.48449 w_k1_IBM_s002
                                 + 0.92803 w_k1_INTC_s002
                                 + 1.23464 w_k1_KO_s002 + 2.18909 w_k1_MAT_s002
                                 + 1.13518 w_k1_MSFT_s002
                                 + 1.06462 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.794533 w_k1_ADM_s002
                                 + 1.19671 w_k1_AXP_s002
                                 + 0.845706 w_k1_DIS_s002
                                 + 0.950783 w_k1_FDX_s002
                                 + 1.00797 w_k1_IBM_s002
                                 + 1.27494 w_k1_INTC_s002
                                 + 0.987904 w_k1_KO_s002
                                 + 1.32561 w_k1_MAT_s002
                                 + 0.935936 w_k1_MSFT_s002
                                 + 1.00743 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.752935 w_k1_ADM_s002
                                 + 0.768019 w_k1_AXP_s002
                                 + 0.644544 w_k1_DIS_s002
                                 + 0.769423 w_k1_FDX_s002
                                 + 0.982509 w_k1_IBM_s002
                                 + 0.754004 w_k1_INTC_s002
                                 + 0.911639 w_k1_KO_s002
                                 + 1.19773 w_k1_MAT_s002
                                 + 0.84193 w_k1_MSFT_s002
                                 + 0.90453 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.831215 w_k1_ADM_s002
                                 + 0.823426 w_k1_AXP_s002
                                 + 0.959882 w_k1_DIS_s002
                                 + 1.01522 w_k1_FDX_s002
                                 + 0.861427 w_k1_IBM_s002
                                 + 1.21972 w_k1_INTC_s002
                                 + 0.958787 w_k1_KO_s002
                                 + 0.909421 w_k1_MAT_s002
                                 + 1.11175 w_k1_MSFT_s002
                                 + 0.90688 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.986824 w_k1_ADM_s002
                                 + 1.14202 w_k1_AXP_s002
                                 + 1.31679 w_k1_DIS_s002
                                 + 1.23658 w_k1_FDX_s002
                                 + 1.00158 w_k1_IBM_s002
                                 + 1.1353 w_k1_INTC_s002 + 1.13919 w_k1_KO_s002
                                 + 1.35336 w_k1_MAT_s002
                                 + 1.3011 w_k1_MSFT_s002 + 1.14304 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.998325 w_k1_ADM_s002
                                 + 1.56205 w_k1_AXP_s002
                                 + 1.74413 w_k1_DIS_s002
                                 + 1.42097 w_k1_FDX_s002 + 1.6607 w_k1_IBM_s002
                                 + 1.26099 w_k1_INTC_s002
                                 + 1.23191 w_k1_KO_s002 + 1.84546 w_k1_MAT_s002
                                 + 1.19973 w_k1_MSFT_s002
                                 + 1.18752 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.656187 w_k1_ADM_s003
                                 + 0.623698 w_k1_AXP_s003
                                 + 0.550551 w_k1_DIS_s003
                                 + 0.685973 w_k1_FDX_s003
                                 + 0.563972 w_k1_IBM_s003
                                 + 0.754362 w_k1_INTC_s003
                                 + 0.626862 w_k1_KO_s003
                                 + 0.693804 w_k1_MAT_s003
                                 + 0.681301 w_k1_MSFT_s003
                                 + 0.719412 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.919694 w_k1_ADM_s003
                                 + 0.773605 w_k1_AXP_s003
                                 + 0.664452 w_k1_DIS_s003
                                 + 0.809337 w_k1_FDX_s003
                                 + 0.709509 w_k1_IBM_s003
                                 + 0.873924 w_k1_INTC_s003
                                 + 0.770934 w_k1_KO_s003
                                 + 0.843145 w_k1_MAT_s003
                                 + 0.852998 w_k1_MSFT_s003
                                 + 0.846108 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.00959 w_k1_ADM_s003
                                 + 0.903837 w_k1_AXP_s003
                                 + 0.938968 w_k1_DIS_s003
                                 + 0.986053 w_k1_FDX_s003
                                 + 0.866997 w_k1_IBM_s003
                                 + 1.07732 w_k1_INTC_s003
                                 + 0.94792 w_k1_KO_s003
                                 + 0.805255 w_k1_MAT_s003
                                 + 1.03017 w_k1_MSFT_s003
                                 + 0.837011 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.19507 w_k1_ADM_s003
                                 + 1.30991 w_k1_AXP_s003
                                 + 1.22699 w_k1_DIS_s003
                                 + 1.33593 w_k1_FDX_s003
                                 + 0.959829 w_k1_IBM_s003
                                 + 1.34954 w_k1_INTC_s003
                                 + 1.18097 w_k1_KO_s003 + 1.06523 w_k1_MAT_s003
                                 + 1.23355 w_k1_MSFT_s003
                                 + 1.04838 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.999274 w_k1_ADM_s003
                                 + 0.928648 w_k1_AXP_s003
                                 + 0.770811 w_k1_DIS_s003
                                 + 0.976422 w_k1_FDX_s003
                                 + 0.602894 w_k1_IBM_s003
                                 + 1.10454 w_k1_INTC_s003
                                 + 0.803617 w_k1_KO_s003 + 0.7126 w_k1_MAT_s003
                                 + 0.901189 w_k1_MSFT_s003
                                 + 0.863579 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.80844 w_k1_ADM_s003
                                 + 0.733643 w_k1_AXP_s003
                                 + 0.646172 w_k1_DIS_s003
                                 + 0.717597 w_k1_FDX_s003
                                 + 1.04055 w_k1_IBM_s003
                                 + 0.698695 w_k1_INTC_s003
                                 + 0.771851 w_k1_KO_s003
                                 + 0.925317 w_k1_MAT_s003
                                 + 0.762325 w_k1_MSFT_s003
                                 + 0.735942 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.09692 w_k1_ADM_s003
                                 + 1.00159 w_k1_AXP_s003
                                 + 0.985039 w_k1_DIS_s003
                                 + 0.953688 w_k1_FDX_s003
                                 + 1.12942 w_k1_IBM_s003
                                 + 1.00121 w_k1_INTC_s003
                                 + 1.08259 w_k1_KO_s003 + 1.2524 w_k1_MAT_s003
                                 + 0.976192 w_k1_MSFT_s003
                                 + 0.901685 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.25923 w_k1_ADM_s003
                                 + 1.40342 w_k1_AXP_s003
                                 + 1.36686 w_k1_DIS_s003
                                 + 1.18231 w_k1_FDX_s003
                                 + 1.46171 w_k1_IBM_s003
                                 + 1.46691 w_k1_INTC_s003
                                 + 1.47903 w_k1_KO_s003 + 1.62538 w_k1_MAT_s003
                                 + 1.23747 w_k1_MSFT_s003 + 1.1957 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.715303 w_k1_ADM_s004
                                 + 0.754217 w_k1_AXP_s004
                                 + 0.647372 w_k1_DIS_s004
                                 + 0.760227 w_k1_FDX_s004
                                 + 0.982404 w_k1_IBM_s004
                                 + 0.741157 w_k1_INTC_s004
                                 + 0.90516 w_k1_KO_s004 + 1.20141 w_k1_MAT_s004
                                 + 0.869386 w_k1_MSFT_s004
                                 + 0.846473 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.931099 w_k1_ADM_s004
                                 + 0.80605 w_k1_AXP_s004
                                 + 0.734463 w_k1_DIS_s004
                                 + 0.837862 w_k1_FDX_s004
                                 + 0.890621 w_k1_IBM_s004
                                 + 0.866449 w_k1_INTC_s004
                                 + 0.85133 w_k1_KO_s004
                                 + 0.872378 w_k1_MAT_s004
                                 + 0.94034 w_k1_MSFT_s004
                                 + 0.834976 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.685654 w_k1_ADM_s004
                                 + 0.689494 w_k1_AXP_s004
                                 + 0.589339 w_k1_DIS_s004
                                 + 0.674808 w_k1_FDX_s004
                                 + 0.645652 w_k1_IBM_s004
                                 + 0.639662 w_k1_INTC_s004
                                 + 0.695003 w_k1_KO_s004
                                 + 0.781498 w_k1_MAT_s004
                                 + 0.683524 w_k1_MSFT_s004
                                 + 0.764109 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.7954 w_k1_ADM_s004
                                 + 0.656893 w_k1_AXP_s004
                                 + 0.991787 w_k1_DIS_s004
                                 + 0.816592 w_k1_FDX_s004
                                 + 0.718365 w_k1_IBM_s004
                                 + 0.773334 w_k1_INTC_s004
                                 + 0.920055 w_k1_KO_s004
                                 + 0.96133 w_k1_MAT_s004
                                 + 0.92415 w_k1_MSFT_s004
                                 + 0.878915 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.957831 w_k1_ADM_s004
                                 + 1.10037 w_k1_AXP_s004
                                 + 1.11609 w_k1_DIS_s004
                                 + 1.05009 w_k1_FDX_s004
                                 + 0.811978 w_k1_IBM_s004
                                 + 1.0162 w_k1_INTC_s004
                                 + 0.903858 w_k1_KO_s004
                                 + 0.856118 w_k1_MAT_s004
                                 + 1.03749 w_k1_MSFT_s004
                                 + 1.02038 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.10831 w_k1_ADM_s004
                                 + 1.37564 w_k1_AXP_s004
                                 + 1.47014 w_k1_DIS_s004
                                 + 1.23836 w_k1_FDX_s004
                                 + 1.51348 w_k1_IBM_s004
                                 + 1.11348 w_k1_INTC_s004
                                 + 1.45468 w_k1_KO_s004 + 1.51205 w_k1_MAT_s004
                                 + 1.17112 w_k1_MSFT_s004
                                 + 1.08342 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.911715 w_k1_ADM_s004
                                 + 1.17511 w_k1_AXP_s004
                                 + 1.03413 w_k1_DIS_s004
                                 + 0.969148 w_k1_FDX_s004
                                 + 1.29516 w_k1_IBM_s004
                                 + 1.03628 w_k1_INTC_s004
                                 + 1.13723 w_k1_KO_s004 + 1.41957 w_k1_MAT_s004
                                 + 0.975325 w_k1_MSFT_s004
                                 + 0.992336 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.19203 w_k1_ADM_s004
                                 + 1.58 w_k1_AXP_s004 + 1.60985 w_k1_DIS_s004
                                 + 1.38826 w_k1_FDX_s004
                                 + 0.945211 w_k1_IBM_s004
                                 + 1.37893 w_k1_INTC_s004
                                 + 1.16867 w_k1_KO_s004 + 1.41969 w_k1_MAT_s004
                                 + 1.40825 w_k1_MSFT_s004
                                 + 1.15782 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.25356 w_k1_ADM_s005
                                 + 1.35541 w_k1_AXP_s005
                                 + 1.23423 w_k1_DIS_s005
                                 + 1.16898 w_k1_FDX_s005 + 1.1996 w_k1_IBM_s005
                                 + 1.04586 w_k1_INTC_s005
                                 + 1.29845 w_k1_KO_s005 + 1.88725 w_k1_MAT_s005
                                 + 1.20458 w_k1_MSFT_s005
                                 + 1.17941 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.41385 w_k1_ADM_s005
                                 + 1.12109 w_k1_AXP_s005
                                 + 1.44156 w_k1_DIS_s005
                                 + 1.17258 w_k1_FDX_s005
                                 + 0.92324 w_k1_IBM_s005
                                 + 0.97667 w_k1_INTC_s005
                                 + 1.33101 w_k1_KO_s005 + 1.34227 w_k1_MAT_s005
                                 + 1.22461 w_k1_MSFT_s005
                                 + 1.10328 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.05768 w_k1_ADM_s005
                                 + 1.06027 w_k1_AXP_s005
                                 + 1.36103 w_k1_DIS_s005
                                 + 1.10084 w_k1_FDX_s005
                                 + 1.29564 w_k1_IBM_s005
                                 + 1.00077 w_k1_INTC_s005
                                 + 1.32136 w_k1_KO_s005 + 1.72793 w_k1_MAT_s005
                                 + 1.13773 w_k1_MSFT_s005
                                 + 1.18653 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.16427 w_k1_ADM_s005
                                 + 1.04984 w_k1_AXP_s005
                                 + 0.91344 w_k1_DIS_s005
                                 + 1.07579 w_k1_FDX_s005
                                 + 1.01885 w_k1_IBM_s005
                                 + 1.06146 w_k1_INTC_s005 + 0.9676 w_k1_KO_s005
                                 + 1.43819 w_k1_MAT_s005
                                 + 0.945682 w_k1_MSFT_s005
                                 + 1.02077 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.869171 w_k1_ADM_s005
                                 + 0.820837 w_k1_AXP_s005
                                 + 0.774984 w_k1_DIS_s005
                                 + 0.934742 w_k1_FDX_s005
                                 + 0.744713 w_k1_IBM_s005
                                 + 0.939685 w_k1_INTC_s005
                                 + 0.878312 w_k1_KO_s005
                                 + 1.18841 w_k1_MAT_s005
                                 + 0.973102 w_k1_MSFT_s005
                                 + 0.963754 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.945058 w_k1_ADM_s005
                                 + 0.656401 w_k1_AXP_s005
                                 + 0.632668 w_k1_DIS_s005
                                 + 0.697704 w_k1_FDX_s005
                                 + 0.961128 w_k1_IBM_s005
                                 + 0.629977 w_k1_INTC_s005
                                 + 0.888934 w_k1_KO_s005
                                 + 0.93914 w_k1_MAT_s005
                                 + 0.704127 w_k1_MSFT_s005
                                 + 0.956568 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.70143 w_k1_ADM_s005
                                 + 1.68975 w_k1_AXP_s005
                                 + 1.53997 w_k1_DIS_s005
                                 + 1.38998 w_k1_FDX_s005
                                 + 1.41616 w_k1_IBM_s005
                                 + 1.22149 w_k1_INTC_s005 + 1.3374 w_k1_KO_s005
                                 + 1.84353 w_k1_MAT_s005
                                 + 1.23968 w_k1_MSFT_s005
                                 + 1.34139 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.0316 w_k1_ADM_s005
                                 + 1.07964 w_k1_AXP_s005
                                 + 0.966086 w_k1_DIS_s005
                                 + 0.895769 w_k1_FDX_s005
                                 + 1.43268 w_k1_IBM_s005
                                 + 0.867827 w_k1_INTC_s005
                                 + 1.37021 w_k1_KO_s005 + 1.96697 w_k1_MAT_s005
                                 + 1.04665 w_k1_MSFT_s005
                                 + 1.09014 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.70581 w_k1_ADM_s006
                                 + 0.739445 w_k1_AXP_s006
                                 + 0.787796 w_k1_DIS_s006
                                 + 0.762917 w_k1_FDX_s006
                                 + 0.848457 w_k1_IBM_s006
                                 + 0.860356 w_k1_INTC_s006
                                 + 0.885355 w_k1_KO_s006
                                 + 0.74167 w_k1_MAT_s006
                                 + 0.852046 w_k1_MSFT_s006
                                 + 0.849865 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.892219 w_k1_ADM_s006
                                 + 0.764478 w_k1_AXP_s006
                                 + 0.791232 w_k1_DIS_s006
                                 + 0.872468 w_k1_FDX_s006
                                 + 0.716531 w_k1_IBM_s006
                                 + 0.848501 w_k1_INTC_s006
                                 + 0.730052 w_k1_KO_s006
                                 + 0.704568 w_k1_MAT_s006
                                 + 0.904194 w_k1_MSFT_s006
                                 + 0.895767 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.98783 w_k1_ADM_s006
                                 + 0.927965 w_k1_AXP_s006
                                 + 1.08549 w_k1_DIS_s006
                                 + 1.10105 w_k1_FDX_s006
                                 + 0.95455 w_k1_IBM_s006
                                 + 1.25882 w_k1_INTC_s006
                                 + 0.951287 w_k1_KO_s006
                                 + 0.822393 w_k1_MAT_s006
                                 + 1.09585 w_k1_MSFT_s006
                                 + 0.991443 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.04121 w_k1_ADM_s006
                                 + 1.28892 w_k1_AXP_s006
                                 + 1.26902 w_k1_DIS_s006
                                 + 1.10989 w_k1_FDX_s006
                                 + 1.29935 w_k1_IBM_s006
                                 + 1.15544 w_k1_INTC_s006
                                 + 1.05365 w_k1_KO_s006 + 1.08971 w_k1_MAT_s006
                                 + 1.11587 w_k1_MSFT_s006
                                 + 1.00609 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.04952 w_k1_ADM_s006
                                 + 1.35664 w_k1_AXP_s006
                                 + 1.83528 w_k1_DIS_s006
                                 + 1.35429 w_k1_FDX_s006
                                 + 1.13257 w_k1_IBM_s006
                                 + 1.26467 w_k1_INTC_s006
                                 + 1.14756 w_k1_KO_s006 + 1.04022 w_k1_MAT_s006
                                 + 1.28262 w_k1_MSFT_s006
                                 + 1.10309 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.944907 w_k1_ADM_s006
                                 + 1.62856 w_k1_AXP_s006
                                 + 1.34021 w_k1_DIS_s006
                                 + 1.25152 w_k1_FDX_s006
                                 + 1.54111 w_k1_IBM_s006
                                 + 1.09822 w_k1_INTC_s006
                                 + 1.38365 w_k1_KO_s006 + 2.50415 w_k1_MAT_s006
                                 + 1.18875 w_k1_MSFT_s006
                                 + 1.07784 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.784008 w_k1_ADM_s006
                                 + 0.933706 w_k1_AXP_s006
                                 + 0.876563 w_k1_DIS_s006
                                 + 0.871253 w_k1_FDX_s006
                                 + 1.16867 w_k1_IBM_s006
                                 + 0.903499 w_k1_INTC_s006
                                 + 1.04158 w_k1_KO_s006 + 1.17175 w_k1_MAT_s006
                                 + 0.835234 w_k1_MSFT_s006
                                 + 1.06926 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.814869 w_k1_ADM_s006
                                 + 1.00592 w_k1_AXP_s006
                                 + 0.942236 w_k1_DIS_s006
                                 + 1.07918 w_k1_FDX_s006
                                 + 0.831839 w_k1_IBM_s006
                                 + 1.22696 w_k1_INTC_s006
                                 + 0.845072 w_k1_KO_s006
                                 + 0.690785 w_k1_MAT_s006
                                 + 0.990531 w_k1_MSFT_s006
                                 + 0.953879 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.979747 w_k1_ADM_s007
                                 + 0.947318 w_k1_AXP_s007
                                 + 0.91429 w_k1_DIS_s007
                                 + 0.974054 w_k1_FDX_s007
                                 + 1.06603 w_k1_IBM_s007
                                 + 1.12958 w_k1_INTC_s007
                                 + 1.02308 w_k1_KO_s007
                                 + 0.792604 w_k1_MAT_s007
                                 + 0.937849 w_k1_MSFT_s007
                                 + 0.93949 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.05409 w_k1_ADM_s007
                                 + 0.948494 w_k1_AXP_s007
                                 + 1.23537 w_k1_DIS_s007
                                 + 1.18482 w_k1_FDX_s007
                                 + 0.774642 w_k1_IBM_s007
                                 + 1.11311 w_k1_INTC_s007
                                 + 0.936642 w_k1_KO_s007
                                 + 0.656013 w_k1_MAT_s007
                                 + 1.19211 w_k1_MSFT_s007
                                 + 0.955125 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.818141 w_k1_ADM_s007
                                 + 0.64313 w_k1_AXP_s007
                                 + 0.637171 w_k1_DIS_s007
                                 + 0.836656 w_k1_FDX_s007
                                 + 0.648426 w_k1_IBM_s007
                                 + 0.952416 w_k1_INTC_s007
                                 + 0.691834 w_k1_KO_s007
                                 + 0.609553 w_k1_MAT_s007
                                 + 0.786789 w_k1_MSFT_s007
                                 + 0.766653 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.15794 w_k1_ADM_s007
                                 + 1.58089 w_k1_AXP_s007
                                 + 1.67955 w_k1_DIS_s007
                                 + 1.46566 w_k1_FDX_s007
                                 + 1.49607 w_k1_IBM_s007
                                 + 1.46525 w_k1_INTC_s007
                                 + 1.56333 w_k1_KO_s007 + 1.57588 w_k1_MAT_s007
                                 + 1.47537 w_k1_MSFT_s007 + 1.2066 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.38743 w_k1_ADM_s007
                                 + 1.44418 w_k1_AXP_s007
                                 + 1.58438 w_k1_DIS_s007
                                 + 1.44802 w_k1_FDX_s007
                                 + 0.964702 w_k1_IBM_s007
                                 + 1.3737 w_k1_INTC_s007 + 1.22555 w_k1_KO_s007
                                 + 0.933801 w_k1_MAT_s007
                                 + 1.16681 w_k1_MSFT_s007
                                 + 1.14936 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.79455 w_k1_ADM_s007
                                 + 1.09254 w_k1_AXP_s007
                                 + 0.799403 w_k1_DIS_s007
                                 + 0.931221 w_k1_FDX_s007
                                 + 0.805943 w_k1_IBM_s007
                                 + 0.990109 w_k1_INTC_s007
                                 + 0.857111 w_k1_KO_s007
                                 + 0.987036 w_k1_MAT_s007
                                 + 0.823003 w_k1_MSFT_s007
                                 + 0.869476 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.15483 w_k1_ADM_s007
                                 + 1.30492 w_k1_AXP_s007
                                 + 1.16122 w_k1_DIS_s007
                                 + 1.14372 w_k1_FDX_s007
                                 + 1.27216 w_k1_IBM_s007
                                 + 1.2366 w_k1_INTC_s007 + 1.21075 w_k1_KO_s007
                                 + 1.22701 w_k1_MAT_s007
                                 + 1.11581 w_k1_MSFT_s007 + 1.1444 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.598646 w_k1_ADM_s007
                                 + 0.799145 w_k1_AXP_s007
                                 + 0.760888 w_k1_DIS_s007
                                 + 0.712661 w_k1_FDX_s007
                                 + 1.31697 w_k1_IBM_s007
                                 + 0.986228 w_k1_INTC_s007
                                 + 1.03847 w_k1_KO_s007 + 1.3285 w_k1_MAT_s007
                                 + 0.849216 w_k1_MSFT_s007
                                 + 0.964783 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.757606 w_k1_ADM_s010
                                 + 0.721269 w_k1_AXP_s010
                                 + 0.744613 w_k1_DIS_s010
                                 + 0.829991 w_k1_FDX_s010
                                 + 0.871995 w_k1_IBM_s010
                                 + 0.799942 w_k1_INTC_s010
                                 + 0.707473 w_k1_KO_s010
                                 + 0.903493 w_k1_MAT_s010
                                 + 0.773663 w_k1_MSFT_s010
                                 + 0.805892 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.83252 w_k1_ADM_s010
                                 + 0.755251 w_k1_AXP_s010
                                 + 0.783194 w_k1_DIS_s010
                                 + 0.932491 w_k1_FDX_s010
                                 + 0.65817 w_k1_IBM_s010
                                 + 1.02038 w_k1_INTC_s010
                                 + 0.72083 w_k1_KO_s010
                                 + 0.701921 w_k1_MAT_s010
                                 + 0.908207 w_k1_MSFT_s010
                                 + 0.870225 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.00531 w_k1_ADM_s010
                                 + 0.947511 w_k1_AXP_s010
                                 + 1.07282 w_k1_DIS_s010
                                 + 1.05816 w_k1_FDX_s010
                                 + 0.896875 w_k1_IBM_s010
                                 + 1.03153 w_k1_INTC_s010
                                 + 1.03122 w_k1_KO_s010
                                 + 0.806639 w_k1_MAT_s010
                                 + 1.00002 w_k1_MSFT_s010
                                 + 1.09712 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.00592 w_k1_ADM_s010
                                 + 1.17071 w_k1_AXP_s010
                                 + 1.07415 w_k1_DIS_s010
                                 + 1.18193 w_k1_FDX_s010
                                 + 0.756114 w_k1_IBM_s010
                                 + 1.07811 w_k1_INTC_s010
                                 + 0.764932 w_k1_KO_s010
                                 + 0.977796 w_k1_MAT_s010
                                 + 1.08896 w_k1_MSFT_s010
                                 + 0.939033 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.21294 w_k1_ADM_s010
                                 + 1.49782 w_k1_AXP_s010
                                 + 1.56122 w_k1_DIS_s010
                                 + 1.21006 w_k1_FDX_s010
                                 + 1.49821 w_k1_IBM_s010
                                 + 1.05336 w_k1_INTC_s010
                                 + 1.38088 w_k1_KO_s010 + 1.84495 w_k1_MAT_s010
                                 + 1.27178 w_k1_MSFT_s010
                                 + 1.25068 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.859068 w_k1_ADM_s010
                                 + 1.00378 w_k1_AXP_s010
                                 + 1.14349 w_k1_DIS_s010
                                 + 1.01331 w_k1_FDX_s010
                                 + 1.59406 w_k1_IBM_s010
                                 + 1.02467 w_k1_INTC_s010
                                 + 1.12783 w_k1_KO_s010 + 1.56274 w_k1_MAT_s010
                                 + 1.01159 w_k1_MSFT_s010
                                 + 0.990545 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.05406 w_k1_ADM_s010
                                 + 1.18184 w_k1_AXP_s010
                                 + 1.32739 w_k1_DIS_s010
                                 + 1.22499 w_k1_FDX_s010
                                 + 0.956256 w_k1_IBM_s010
                                 + 1.38221 w_k1_INTC_s010
                                 + 1.10614 w_k1_KO_s010 + 1.04714 w_k1_MAT_s010
                                 + 1.16081 w_k1_MSFT_s010
                                 + 1.15409 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.994257 w_k1_ADM_s010
                                 + 1.05525 w_k1_AXP_s010
                                 + 0.977882 w_k1_DIS_s010
                                 + 0.881866 w_k1_FDX_s010
                                 + 1.19718 w_k1_IBM_s010
                                 + 0.898606 w_k1_INTC_s010
                                 + 0.910754 w_k1_KO_s010
                                 + 1.27012 w_k1_MAT_s010
                                 + 0.947178 w_k1_MSFT_s010
                                 + 1.01358 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.06811 w_k1_ADM_s011
                                 + 1.42343 w_k1_AXP_s011
                                 + 1.25994 w_k1_DIS_s011 + 1.1356 w_k1_FDX_s011
                                 + 1.28763 w_k1_IBM_s011
                                 + 1.28741 w_k1_INTC_s011
                                 + 1.17535 w_k1_KO_s011 + 1.42827 w_k1_MAT_s011
                                 + 1.22956 w_k1_MSFT_s011
                                 + 1.13929 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.859572 w_k1_ADM_s011
                                 + 0.966746 w_k1_AXP_s011
                                 + 0.852579 w_k1_DIS_s011
                                 + 0.823809 w_k1_FDX_s011
                                 + 1.74635 w_k1_IBM_s011
                                 + 1.14445 w_k1_INTC_s011
                                 + 1.17071 w_k1_KO_s011 + 1.79059 w_k1_MAT_s011
                                 + 1.02741 w_k1_MSFT_s011
                                 + 1.15118 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.853553 w_k1_ADM_s011
                                 + 0.767364 w_k1_AXP_s011
                                 + 0.679365 w_k1_DIS_s011
                                 + 0.75304 w_k1_FDX_s011
                                 + 0.876216 w_k1_IBM_s011
                                 + 0.995326 w_k1_INTC_s011
                                 + 0.802007 w_k1_KO_s011
                                 + 0.751989 w_k1_MAT_s011
                                 + 0.9368 w_k1_MSFT_s011
                                 + 0.915183 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.892634 w_k1_ADM_s011
                                 + 0.868347 w_k1_AXP_s011
                                 + 0.800293 w_k1_DIS_s011
                                 + 0.872582 w_k1_FDX_s011
                                 + 1.05266 w_k1_IBM_s011
                                 + 0.971919 w_k1_INTC_s011
                                 + 0.957894 w_k1_KO_s011
                                 + 0.976559 w_k1_MAT_s011
                                 + 1.1305 w_k1_MSFT_s011
                                 + 0.885222 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.966678 w_k1_ADM_s011
                                 + 0.801516 w_k1_AXP_s011
                                 + 0.68087 w_k1_DIS_s011
                                 + 0.890068 w_k1_FDX_s011
                                 + 0.589153 w_k1_IBM_s011
                                 + 0.869132 w_k1_INTC_s011
                                 + 0.630754 w_k1_KO_s011
                                 + 0.612727 w_k1_MAT_s011
                                 + 0.982347 w_k1_MSFT_s011
                                 + 0.864328 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.31917 w_k1_ADM_s011
                                 + 1.38925 w_k1_AXP_s011
                                 + 1.11387 w_k1_DIS_s011
                                 + 1.24362 w_k1_FDX_s011
                                 + 0.817723 w_k1_IBM_s011
                                 + 1.19044 w_k1_INTC_s011
                                 + 0.903327 w_k1_KO_s011
                                 + 0.935287 w_k1_MAT_s011
                                 + 1.23959 w_k1_MSFT_s011
                                 + 0.985253 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.916123 w_k1_ADM_s011
                                 + 1.19262 w_k1_AXP_s011
                                 + 1.41454 w_k1_DIS_s011
                                 + 1.14187 w_k1_FDX_s011
                                 + 1.18052 w_k1_IBM_s011
                                 + 1.23695 w_k1_INTC_s011 + 1.2397 w_k1_KO_s011
                                 + 1.26006 w_k1_MAT_s011
                                 + 1.34999 w_k1_MSFT_s011
                                 + 1.14802 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.945757 w_k1_ADM_s011
                                 + 1.04063 w_k1_AXP_s011
                                 + 1.17069 w_k1_DIS_s011
                                 + 1.00938 w_k1_FDX_s011
                                 + 1.51743 w_k1_IBM_s011
                                 + 1.27006 w_k1_INTC_s011 + 1.3028 w_k1_KO_s011
                                 + 1.49661 w_k1_MAT_s011
                                 + 1.29353 w_k1_MSFT_s011
                                 + 1.02832 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.79 w_k1_ADM_s012
                                 + 0.823426 w_k1_AXP_s012
                                 + 0.840754 w_k1_DIS_s012
                                 + 0.815446 w_k1_FDX_s012
                                 + 0.881718 w_k1_IBM_s012
                                 + 0.804575 w_k1_INTC_s012
                                 + 0.929195 w_k1_KO_s012
                                 + 0.946619 w_k1_MAT_s012
                                 + 0.861154 w_k1_MSFT_s012
                                 + 1.07633 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.58646 w_k1_ADM_s012
                                 + 1.41441 w_k1_AXP_s012
                                 + 1.63765 w_k1_DIS_s012
                                 + 1.31676 w_k1_FDX_s012
                                 + 1.19049 w_k1_IBM_s012
                                 + 1.17957 w_k1_INTC_s012
                                 + 1.24849 w_k1_KO_s012
                                 + 0.949446 w_k1_MAT_s012
                                 + 1.34575 w_k1_MSFT_s012
                                 + 1.18861 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.06613 w_k1_ADM_s012
                                 + 0.751174 w_k1_AXP_s012
                                 + 0.724736 w_k1_DIS_s012
                                 + 0.818447 w_k1_FDX_s012
                                 + 0.955468 w_k1_IBM_s012
                                 + 0.817912 w_k1_INTC_s012
                                 + 0.830868 w_k1_KO_s012
                                 + 0.789691 w_k1_MAT_s012
                                 + 0.954111 w_k1_MSFT_s012
                                 + 0.865022 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.38599 w_k1_ADM_s012
                                 + 0.866876 w_k1_AXP_s012
                                 + 1.05586 w_k1_DIS_s012
                                 + 1.05475 w_k1_FDX_s012
                                 + 0.931903 w_k1_IBM_s012
                                 + 0.828427 w_k1_INTC_s012
                                 + 0.999635 w_k1_KO_s012
                                 + 1.04713 w_k1_MAT_s012
                                 + 1.11093 w_k1_MSFT_s012 + 1.2101 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.963612 w_k1_ADM_s012
                                 + 0.850249 w_k1_AXP_s012
                                 + 0.774358 w_k1_DIS_s012
                                 + 0.908183 w_k1_FDX_s012
                                 + 0.764619 w_k1_IBM_s012
                                 + 0.729911 w_k1_INTC_s012
                                 + 0.857871 w_k1_KO_s012
                                 + 0.904429 w_k1_MAT_s012
                                 + 0.929016 w_k1_MSFT_s012
                                 + 1.02199 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.11085 w_k1_ADM_s012
                                 + 1.0543 w_k1_AXP_s012
                                 + 0.932009 w_k1_DIS_s012
                                 + 0.964106 w_k1_FDX_s012
                                 + 1.3007 w_k1_IBM_s012
                                 + 1.07063 w_k1_INTC_s012
                                 + 1.01819 w_k1_KO_s012 + 1.2971 w_k1_MAT_s012
                                 + 0.965303 w_k1_MSFT_s012
                                 + 1.09594 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.767717 w_k1_ADM_s012
                                 + 0.719543 w_k1_AXP_s012
                                 + 0.645605 w_k1_DIS_s012
                                 + 0.720501 w_k1_FDX_s012
                                 + 0.715833 w_k1_IBM_s012
                                 + 0.667141 w_k1_INTC_s012
                                 + 0.683894 w_k1_KO_s012
                                 + 0.549656 w_k1_MAT_s012
                                 + 0.781026 w_k1_MSFT_s012
                                 + 0.885794 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.04144 w_k1_ADM_s012
                                 + 1.21107 w_k1_AXP_s012
                                 + 1.15464 w_k1_DIS_s012
                                 + 1.07379 w_k1_FDX_s012
                                 + 1.03467 w_k1_IBM_s012
                                 + 1.00749 w_k1_INTC_s012
                                 + 1.09848 w_k1_KO_s012 + 1.18748 w_k1_MAT_s012
                                 + 1.07219 w_k1_MSFT_s012
                                 + 1.16268 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.783383 w_k1_ADM_s013
                                 + 0.698291 w_k1_AXP_s013
                                 + 0.680641 w_k1_DIS_s013
                                 + 0.782286 w_k1_FDX_s013
                                 + 0.899878 w_k1_IBM_s013
                                 + 0.927746 w_k1_INTC_s013
                                 + 1.03727 w_k1_KO_s013
                                 + 0.685061 w_k1_MAT_s013
                                 + 0.879625 w_k1_MSFT_s013
                                 + 0.88558 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.849353 w_k1_ADM_s013
                                 + 0.641837 w_k1_AXP_s013
                                 + 0.756173 w_k1_DIS_s013
                                 + 0.876871 w_k1_FDX_s013
                                 + 0.562742 w_k1_IBM_s013
                                 + 0.962577 w_k1_INTC_s013
                                 + 0.715513 w_k1_KO_s013
                                 + 0.51905 w_k1_MAT_s013
                                 + 0.94543 w_k1_MSFT_s013
                                 + 0.916042 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.18809 w_k1_ADM_s013
                                 + 1.07014 w_k1_AXP_s013
                                 + 1.13169 w_k1_DIS_s013
                                 + 1.16373 w_k1_FDX_s013
                                 + 1.01091 w_k1_IBM_s013
                                 + 1.18667 w_k1_INTC_s013 + 1.2682 w_k1_KO_s013
                                 + 0.959754 w_k1_MAT_s013
                                 + 1.29027 w_k1_MSFT_s013
                                 + 1.17057 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.10046 w_k1_ADM_s013
                                 + 0.782068 w_k1_AXP_s013
                                 + 0.87386 w_k1_DIS_s013
                                 + 0.989224 w_k1_FDX_s013
                                 + 0.770583 w_k1_IBM_s013
                                 + 0.970188 w_k1_INTC_s013
                                 + 0.925616 w_k1_KO_s013
                                 + 0.722942 w_k1_MAT_s013
                                 + 1.08689 w_k1_MSFT_s013
                                 + 0.855721 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.22613 w_k1_ADM_s013
                                 + 1.51188 w_k1_AXP_s013
                                 + 1.92306 w_k1_DIS_s013 + 1.3776 w_k1_FDX_s013
                                 + 1.51153 w_k1_IBM_s013
                                 + 1.3269 w_k1_INTC_s013 + 1.67793 w_k1_KO_s013
                                 + 1.2701 w_k1_MAT_s013
                                 + 1.51921 w_k1_MSFT_s013
                                 + 1.22341 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.17304 w_k1_ADM_s013
                                 + 1.01738 w_k1_AXP_s013
                                 + 1.31314 w_k1_DIS_s013
                                 + 0.982009 w_k1_FDX_s013
                                 + 1.76257 w_k1_IBM_s013
                                 + 1.20995 w_k1_INTC_s013
                                 + 1.43568 w_k1_KO_s013 + 1.35121 w_k1_MAT_s013
                                 + 1.16415 w_k1_MSFT_s013
                                 + 1.18747 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.18759 w_k1_ADM_s013
                                 + 1.16502 w_k1_AXP_s013
                                 + 1.02637 w_k1_DIS_s013
                                 + 0.94974 w_k1_FDX_s013
                                 + 1.34836 w_k1_IBM_s013
                                 + 0.947403 w_k1_INTC_s013
                                 + 1.20102 w_k1_KO_s013 + 1.43672 w_k1_MAT_s013
                                 + 1.05475 w_k1_MSFT_s013
                                 + 1.03096 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.852699 w_k1_ADM_s013
                                 + 1.01121 w_k1_AXP_s013
                                 + 0.935387 w_k1_DIS_s013
                                 + 1.07434 w_k1_FDX_s013
                                 + 0.743746 w_k1_IBM_s013
                                 + 1.25642 w_k1_INTC_s013
                                 + 0.89909 w_k1_KO_s013
                                 + 0.786674 w_k1_MAT_s013
                                 + 1.135 w_k1_MSFT_s013 + 1.04532 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.01027 w_k1_ADM_s014
                                 + 0.808563 w_k1_AXP_s014
                                 + 0.686331 w_k1_DIS_s014
                                 + 0.902781 w_k1_FDX_s014
                                 + 0.88901 w_k1_IBM_s014
                                 + 1.0135 w_k1_INTC_s014
                                 + 0.847689 w_k1_KO_s014
                                 + 1.15732 w_k1_MAT_s014
                                 + 1.06685 w_k1_MSFT_s014
                                 + 0.909217 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.1684 w_k1_ADM_s014
                                 + 1.10546 w_k1_AXP_s014
                                 + 1.05896 w_k1_DIS_s014
                                 + 1.20639 w_k1_FDX_s014
                                 + 0.83142 w_k1_IBM_s014
                                 + 1.05918 w_k1_INTC_s014
                                 + 0.920042 w_k1_KO_s014
                                 + 1.03667 w_k1_MAT_s014
                                 + 1.28885 w_k1_MSFT_s014
                                 + 1.01513 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.16695 w_k1_ADM_s014
                                 + 1.00246 w_k1_AXP_s014
                                 + 1.17004 w_k1_DIS_s014
                                 + 1.20583 w_k1_FDX_s014 + 1.0763 w_k1_IBM_s014
                                 + 0.945378 w_k1_INTC_s014
                                 + 0.965418 w_k1_KO_s014 + 1.2132 w_k1_MAT_s014
                                 + 1.01453 w_k1_MSFT_s014
                                 + 1.14567 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.192 w_k1_ADM_s014
                                 + 1.34434 w_k1_AXP_s014
                                 + 1.44553 w_k1_DIS_s014
                                 + 1.17578 w_k1_FDX_s014
                                 + 1.47642 w_k1_IBM_s014
                                 + 1.0508 w_k1_INTC_s014 + 1.20094 w_k1_KO_s014
                                 + 1.44802 w_k1_MAT_s014 + 1.137 w_k1_MSFT_s014
                                 + 0.993614 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.11979 w_k1_ADM_s014
                                 + 0.796021 w_k1_AXP_s014
                                 + 0.89035 w_k1_DIS_s014
                                 + 0.870398 w_k1_FDX_s014
                                 + 1.29096 w_k1_IBM_s014
                                 + 0.869375 w_k1_INTC_s014
                                 + 1.08194 w_k1_KO_s014 + 1.25779 w_k1_MAT_s014
                                 + 1.00189 w_k1_MSFT_s014
                                 + 0.85814 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.758777 w_k1_ADM_s014
                                 + 0.705474 w_k1_AXP_s014
                                 + 0.655622 w_k1_DIS_s014
                                 + 0.755874 w_k1_FDX_s014
                                 + 0.783694 w_k1_IBM_s014
                                 + 0.809234 w_k1_INTC_s014
                                 + 0.658501 w_k1_KO_s014
                                 + 0.770387 w_k1_MAT_s014
                                 + 0.729986 w_k1_MSFT_s014
                                 + 0.788245 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.17277 w_k1_ADM_s014
                                 + 0.825627 w_k1_AXP_s014
                                 + 0.912722 w_k1_DIS_s014
                                 + 0.962009 w_k1_FDX_s014
                                 + 0.862348 w_k1_IBM_s014
                                 + 0.805146 w_k1_INTC_s014
                                 + 0.8522 w_k1_KO_s014 + 0.815918 w_k1_MAT_s014
                                 + 0.986926 w_k1_MSFT_s014
                                 + 0.942736 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.30123 w_k1_ADM_s014
                                 + 1.47086 w_k1_AXP_s014
                                 + 1.24395 w_k1_DIS_s014
                                 + 1.22208 w_k1_FDX_s014
                                 + 1.03767 w_k1_IBM_s014
                                 + 0.958243 w_k1_INTC_s014
                                 + 0.970074 w_k1_KO_s014 + 1.2627 w_k1_MAT_s014
                                 + 1.05873 w_k1_MSFT_s014
                                 + 0.991475 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.828735 w_k1_ADM_s015
                                 + 0.779793 w_k1_AXP_s015
                                 + 0.645995 w_k1_DIS_s015
                                 + 0.820166 w_k1_FDX_s015
                                 + 0.682764 w_k1_IBM_s015
                                 + 0.689264 w_k1_INTC_s015
                                 + 0.8538 w_k1_KO_s015 + 0.938915 w_k1_MAT_s015
                                 + 0.765671 w_k1_MSFT_s015
                                 + 0.998457 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.926046 w_k1_ADM_s015
                                 + 0.848331 w_k1_AXP_s015
                                 + 0.777155 w_k1_DIS_s015
                                 + 0.847434 w_k1_FDX_s015
                                 + 1.08182 w_k1_IBM_s015
                                 + 0.947073 w_k1_INTC_s015
                                 + 0.932865 w_k1_KO_s015
                                 + 1.12648 w_k1_MAT_s015
                                 + 0.954923 w_k1_MSFT_s015
                                 + 0.978544 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.40335 w_k1_ADM_s015
                                 + 1.13503 w_k1_AXP_s015
                                 + 0.920916 w_k1_DIS_s015
                                 + 1.13649 w_k1_FDX_s015
                                 + 0.845554 w_k1_IBM_s015
                                 + 0.989332 w_k1_INTC_s015
                                 + 1.01421 w_k1_KO_s015 + 1.07773 w_k1_MAT_s015
                                 + 1.15754 w_k1_MSFT_s015
                                 + 1.04217 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.16069 w_k1_ADM_s015
                                 + 0.800808 w_k1_AXP_s015
                                 + 1.26115 w_k1_DIS_s015
                                 + 1.06846 w_k1_FDX_s015
                                 + 0.844741 w_k1_IBM_s015
                                 + 1.1829 w_k1_INTC_s015 + 1.14506 w_k1_KO_s015
                                 + 0.720546 w_k1_MAT_s015
                                 + 1.35068 w_k1_MSFT_s015
                                 + 1.13186 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.18689 w_k1_ADM_s015
                                 + 1.36223 w_k1_AXP_s015
                                 + 1.14757 w_k1_DIS_s015
                                 + 1.08486 w_k1_FDX_s015
                                 + 1.37846 w_k1_IBM_s015
                                 + 1.15487 w_k1_INTC_s015
                                 + 1.33537 w_k1_KO_s015 + 2.09921 w_k1_MAT_s015
                                 + 1.20105 w_k1_MSFT_s015
                                 + 1.24839 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.15775 w_k1_ADM_s015
                                 + 1.22209 w_k1_AXP_s015
                                 + 1.80823 w_k1_DIS_s015
                                 + 1.23834 w_k1_FDX_s015
                                 + 1.58964 w_k1_IBM_s015
                                 + 1.0513 w_k1_INTC_s015 + 1.62739 w_k1_KO_s015
                                 + 1.96277 w_k1_MAT_s015
                                 + 1.42618 w_k1_MSFT_s015
                                 + 1.37677 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.920819 w_k1_ADM_s015
                                 + 0.857041 w_k1_AXP_s015
                                 + 1.0059 w_k1_DIS_s015
                                 + 0.926961 w_k1_FDX_s015
                                 + 1.36478 w_k1_IBM_s015
                                 + 0.905492 w_k1_INTC_s015
                                 + 1.27775 w_k1_KO_s015 + 1.41693 w_k1_MAT_s015
                                 + 1.03759 w_k1_MSFT_s015
                                 + 1.03348 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.3582 w_k1_ADM_s015
                                 + 1.33467 w_k1_AXP_s015
                                 + 1.81643 w_k1_DIS_s015
                                 + 1.37887 w_k1_FDX_s015
                                 + 1.25236 w_k1_IBM_s015
                                 + 1.13449 w_k1_INTC_s015
                                 + 1.37093 w_k1_KO_s015 + 1.12491 w_k1_MAT_s015
                                 + 1.50378 w_k1_MSFT_s015
                                 + 1.34166 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.843592 w_k1_ADM_s016
                                 + 0.764819 w_k1_AXP_s016
                                 + 0.873873 w_k1_DIS_s016
                                 + 0.863765 w_k1_FDX_s016
                                 + 0.718743 w_k1_IBM_s016
                                 + 0.929515 w_k1_INTC_s016
                                 + 0.82665 w_k1_KO_s016
                                 + 0.610994 w_k1_MAT_s016
                                 + 0.964532 w_k1_MSFT_s016
                                 + 0.918001 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.878253 w_k1_ADM_s016
                                 + 0.804696 w_k1_AXP_s016
                                 + 0.842907 w_k1_DIS_s016
                                 + 0.768965 w_k1_FDX_s016
                                 + 1.00663 w_k1_IBM_s016
                                 + 0.846273 w_k1_INTC_s016
                                 + 0.990556 w_k1_KO_s016
                                 + 0.905148 w_k1_MAT_s016
                                 + 0.798773 w_k1_MSFT_s016
                                 + 0.908673 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.46873 w_k1_ADM_s016
                                 + 1.43038 w_k1_AXP_s016
                                 + 1.84228 w_k1_DIS_s016
                                 + 1.40173 w_k1_FDX_s016
                                 + 1.41931 w_k1_IBM_s016
                                 + 1.50916 w_k1_INTC_s016
                                 + 1.79726 w_k1_KO_s016 + 1.72256 w_k1_MAT_s016
                                 + 1.46721 w_k1_MSFT_s016
                                 + 1.29837 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.766048 w_k1_ADM_s016
                                 + 0.95775 w_k1_AXP_s016
                                 + 1.06524 w_k1_DIS_s016
                                 + 0.852318 w_k1_FDX_s016
                                 + 1.43096 w_k1_IBM_s016
                                 + 0.953785 w_k1_INTC_s016
                                 + 1.30185 w_k1_KO_s016 + 1.34092 w_k1_MAT_s016
                                 + 0.988479 w_k1_MSFT_s016
                                 + 0.997162 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.0324 w_k1_ADM_s016
                                 + 1.4149 w_k1_AXP_s016 + 1.06487 w_k1_DIS_s016
                                 + 1.16042 w_k1_FDX_s016
                                 + 1.05075 w_k1_IBM_s016
                                 + 1.14479 w_k1_INTC_s016
                                 + 1.08821 w_k1_KO_s016 + 1.35553 w_k1_MAT_s016
                                 + 1.00866 w_k1_MSFT_s016
                                 + 1.15407 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.796724 w_k1_ADM_s016
                                 + 0.61949 w_k1_AXP_s016
                                 + 0.605768 w_k1_DIS_s016
                                 + 0.698895 w_k1_FDX_s016
                                 + 0.745422 w_k1_IBM_s016
                                 + 0.867196 w_k1_INTC_s016
                                 + 0.768094 w_k1_KO_s016
                                 + 0.662724 w_k1_MAT_s016
                                 + 0.879048 w_k1_MSFT_s016
                                 + 0.868341 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.10171 w_k1_ADM_s016
                                 + 1.11249 w_k1_AXP_s016 + 1.1937 w_k1_DIS_s016
                                 + 1.11454 w_k1_FDX_s016
                                 + 0.698349 w_k1_IBM_s016
                                 + 0.972631 w_k1_INTC_s016
                                 + 0.918489 w_k1_KO_s016
                                 + 0.693396 w_k1_MAT_s016
                                 + 1.0277 w_k1_MSFT_s016
                                 + 0.960224 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.561978 w_k1_ADM_s016
                                 + 0.60749 w_k1_AXP_s016
                                 + 0.64868 w_k1_DIS_s016
                                 + 0.604257 w_k1_FDX_s016
                                 + 1.17998 w_k1_IBM_s016
                                 + 0.719144 w_k1_INTC_s016
                                 + 0.912917 w_k1_KO_s016
                                 + 0.987601 w_k1_MAT_s016
                                 + 0.645362 w_k1_MSFT_s016
                                 + 0.855465 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.997149 w_k1_ADM_s017
                                 + 1.43031 w_k1_AXP_s017
                                 + 0.804007 w_k1_DIS_s017
                                 + 0.95692 w_k1_FDX_s017
                                 + 1.18893 w_k1_IBM_s017
                                 + 1.14053 w_k1_INTC_s017
                                 + 1.01762 w_k1_KO_s017 + 1.15367 w_k1_MAT_s017
                                 + 0.887441 w_k1_MSFT_s017
                                 + 1.04312 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.995704 w_k1_ADM_s017
                                 + 1.16235 w_k1_AXP_s017
                                 + 1.27158 w_k1_DIS_s017
                                 + 1.20052 w_k1_FDX_s017
                                 + 0.938114 w_k1_IBM_s017
                                 + 1.41045 w_k1_INTC_s017
                                 + 1.17504 w_k1_KO_s017
                                 + 0.822495 w_k1_MAT_s017
                                 + 1.17155 w_k1_MSFT_s017
                                 + 1.07619 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.969289 w_k1_ADM_s017
                                 + 1.04104 w_k1_AXP_s017
                                 + 1.08589 w_k1_DIS_s017
                                 + 0.889965 w_k1_FDX_s017
                                 + 1.27135 w_k1_IBM_s017
                                 + 1.00738 w_k1_INTC_s017
                                 + 1.13718 w_k1_KO_s017 + 1.35383 w_k1_MAT_s017
                                 + 1.06484 w_k1_MSFT_s017
                                 + 1.07754 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.901796 w_k1_ADM_s017
                                 + 1.02241 w_k1_AXP_s017
                                 + 0.826605 w_k1_DIS_s017
                                 + 1.02775 w_k1_FDX_s017
                                 + 0.737597 w_k1_IBM_s017
                                 + 1.03855 w_k1_INTC_s017
                                 + 0.882668 w_k1_KO_s017 + 0.8209 w_k1_MAT_s017
                                 + 1.07008 w_k1_MSFT_s017
                                 + 0.897472 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.07376 w_k1_ADM_s017
                                 + 0.914606 w_k1_AXP_s017
                                 + 0.930981 w_k1_DIS_s017
                                 + 0.981147 w_k1_FDX_s017
                                 + 0.8046 w_k1_IBM_s017
                                 + 0.909891 w_k1_INTC_s017
                                 + 0.807622 w_k1_KO_s017
                                 + 0.830779 w_k1_MAT_s017
                                 + 0.956948 w_k1_MSFT_s017
                                 + 0.914726 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.711043 w_k1_ADM_s017
                                 + 0.826801 w_k1_AXP_s017
                                 + 0.650937 w_k1_DIS_s017
                                 + 0.764028 w_k1_FDX_s017
                                 + 0.781208 w_k1_IBM_s017
                                 + 0.844984 w_k1_INTC_s017
                                 + 0.708773 w_k1_KO_s017
                                 + 0.573907 w_k1_MAT_s017
                                 + 0.725467 w_k1_MSFT_s017
                                 + 0.836881 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.34819 w_k1_ADM_s017
                                 + 1.6407 w_k1_AXP_s017 + 1.60033 w_k1_DIS_s017
                                 + 1.40161 w_k1_FDX_s017
                                 + 1.16309 w_k1_IBM_s017
                                 + 1.41578 w_k1_INTC_s017
                                 + 1.33182 w_k1_KO_s017 + 1.16145 w_k1_MAT_s017
                                 + 1.23948 w_k1_MSFT_s017
                                 + 1.19899 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.01835 w_k1_ADM_s017
                                 + 1.50054 w_k1_AXP_s017
                                 + 1.46814 w_k1_DIS_s017
                                 + 1.17712 w_k1_FDX_s017
                                 + 1.45812 w_k1_IBM_s017
                                 + 1.15565 w_k1_INTC_s017
                                 + 1.33609 w_k1_KO_s017 + 1.36178 w_k1_MAT_s017
                                 + 1.27713 w_k1_MSFT_s017
                                 + 1.21519 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.89243 w_k1_ADM_s100
                                 + 1.12448 w_k1_AXP_s100
                                 + 1.22134 w_k1_DIS_s100 + 1.1108 w_k1_FDX_s100
                                 + 1.52648 w_k1_IBM_s100
                                 + 0.978161 w_k1_INTC_s100
                                 + 1.32823 w_k1_KO_s100 + 2.14774 w_k1_MAT_s100
                                 + 1.0844 w_k1_MSFT_s100 + 1.089 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.581084 w_k1_ADM_s100
                                 + 0.773367 w_k1_AXP_s100
                                 + 0.631806 w_k1_DIS_s100
                                 + 0.790458 w_k1_FDX_s100
                                 + 1.05637 w_k1_IBM_s100
                                 + 0.847813 w_k1_INTC_s100
                                 + 0.935216 w_k1_KO_s100
                                 + 1.13771 w_k1_MAT_s100
                                 + 0.802836 w_k1_MSFT_s100
                                 + 0.901897 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.899843 w_k1_ADM_s100
                                 + 0.835096 w_k1_AXP_s100
                                 + 0.765339 w_k1_DIS_s100
                                 + 1.01121 w_k1_FDX_s100
                                 + 0.816088 w_k1_IBM_s100
                                 + 0.962781 w_k1_INTC_s100
                                 + 0.818104 w_k1_KO_s100
                                 + 0.984678 w_k1_MAT_s100
                                 + 0.961645 w_k1_MSFT_s100
                                 + 0.989602 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.868076 w_k1_ADM_s100
                                 + 0.758517 w_k1_AXP_s100
                                 + 0.580438 w_k1_DIS_s100
                                 + 0.844652 w_k1_FDX_s100
                                 + 0.646207 w_k1_IBM_s100
                                 + 0.707916 w_k1_INTC_s100
                                 + 0.741923 w_k1_KO_s100
                                 + 1.13879 w_k1_MAT_s100
                                 + 0.823631 w_k1_MSFT_s100
                                 + 0.89238 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.648813 w_k1_ADM_s100
                                 + 0.950396 w_k1_AXP_s100
                                 + 0.883768 w_k1_DIS_s100
                                 + 0.953978 w_k1_FDX_s100
                                 + 1.27783 w_k1_IBM_s100
                                 + 0.953911 w_k1_INTC_s100
                                 + 1.05387 w_k1_KO_s100 + 1.62542 w_k1_MAT_s100
                                 + 0.981299 w_k1_MSFT_s100
                                 + 0.931548 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.33667 w_k1_ADM_s100
                                 + 1.52082 w_k1_AXP_s100
                                 + 1.71953 w_k1_DIS_s100
                                 + 1.54497 w_k1_FDX_s100
                                 + 0.935232 w_k1_IBM_s100
                                 + 1.20311 w_k1_INTC_s100 + 1.2926 w_k1_KO_s100
                                 + 1.40887 w_k1_MAT_s100
                                 + 1.4051 w_k1_MSFT_s100 + 1.37276 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.958025 w_k1_ADM_s100
                                 + 0.999106 w_k1_AXP_s100
                                 + 0.893343 w_k1_DIS_s100
                                 + 1.23105 w_k1_FDX_s100
                                 + 0.619711 w_k1_IBM_s100
                                 + 1.00777 w_k1_INTC_s100
                                 + 0.760255 w_k1_KO_s100
                                 + 0.706135 w_k1_MAT_s100
                                 + 0.978353 w_k1_MSFT_s100
                                 + 0.967357 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.26752 w_k1_ADM_s100
                                 + 1.16832 w_k1_AXP_s100
                                 + 1.15663 w_k1_DIS_s100
                                 + 1.25761 w_k1_FDX_s100
                                 + 0.894561 w_k1_IBM_s100
                                 + 1.23883 w_k1_INTC_s100
                                 + 1.00882 w_k1_KO_s100 + 1.11468 w_k1_MAT_s100
                                 + 1.3609 w_k1_MSFT_s100 + 1.09468 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.981391 w_k1_ADM_s101
                                 + 1.28026 w_k1_AXP_s101
                                 + 1.23471 w_k1_DIS_s101
                                 + 1.07618 w_k1_FDX_s101
                                 + 1.03339 w_k1_IBM_s101
                                 + 0.828663 w_k1_INTC_s101
                                 + 1.0932 w_k1_KO_s101 + 1.18894 w_k1_MAT_s101
                                 + 0.975775 w_k1_MSFT_s101
                                 + 1.03459 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.987151 w_k1_ADM_s101
                                 + 0.71145 w_k1_AXP_s101
                                 + 0.932515 w_k1_DIS_s101
                                 + 0.857787 w_k1_FDX_s101
                                 + 1.14275 w_k1_IBM_s101
                                 + 0.737848 w_k1_INTC_s101
                                 + 1.1437 w_k1_KO_s101 + 1.23124 w_k1_MAT_s101
                                 + 0.882775 w_k1_MSFT_s101
                                 + 0.987106 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.830429 w_k1_ADM_s101
                                 + 0.775969 w_k1_AXP_s101
                                 + 0.787715 w_k1_DIS_s101
                                 + 0.755864 w_k1_FDX_s101
                                 + 0.962472 w_k1_IBM_s101
                                 + 0.772209 w_k1_INTC_s101
                                 + 0.991981 w_k1_KO_s101
                                 + 0.957651 w_k1_MAT_s101
                                 + 0.778554 w_k1_MSFT_s101
                                 + 0.897305 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.802962 w_k1_ADM_s101
                                 + 0.986566 w_k1_AXP_s101
                                 + 1.05044 w_k1_DIS_s101
                                 + 0.840303 w_k1_FDX_s101
                                 + 1.49752 w_k1_IBM_s101
                                 + 0.890497 w_k1_INTC_s101
                                 + 1.39348 w_k1_KO_s101 + 1.98099 w_k1_MAT_s101
                                 + 0.970559 w_k1_MSFT_s101
                                 + 1.04317 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.891162 w_k1_ADM_s101
                                 + 0.840705 w_k1_AXP_s101
                                 + 1.0294 w_k1_DIS_s101 + 1.07365 w_k1_FDX_s101
                                 + 0.933535 w_k1_IBM_s101
                                 + 1.11435 w_k1_INTC_s101
                                 + 0.945548 w_k1_KO_s101
                                 + 0.824559 w_k1_MAT_s101
                                 + 1.10326 w_k1_MSFT_s101
                                 + 0.995975 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.14925 w_k1_ADM_s101
                                 + 1.00788 w_k1_AXP_s101
                                 + 1.21965 w_k1_DIS_s101
                                 + 1.00196 w_k1_FDX_s101
                                 + 1.40202 w_k1_IBM_s101
                                 + 0.977887 w_k1_INTC_s101
                                 + 1.24503 w_k1_KO_s101 + 1.4357 w_k1_MAT_s101
                                 + 1.003 w_k1_MSFT_s101 + 1.0383 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.30393 w_k1_ADM_s101
                                 + 1.38526 w_k1_AXP_s101
                                 + 1.43206 w_k1_DIS_s101
                                 + 1.32559 w_k1_FDX_s101
                                 + 1.30055 w_k1_IBM_s101
                                 + 1.03746 w_k1_INTC_s101
                                 + 1.24646 w_k1_KO_s101 + 1.33028 w_k1_MAT_s101
                                 + 1.31044 w_k1_MSFT_s101
                                 + 1.30755 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.839665 w_k1_ADM_s101
                                 + 0.890338 w_k1_AXP_s101
                                 + 0.853322 w_k1_DIS_s101
                                 + 0.909292 w_k1_FDX_s101
                                 + 0.657349 w_k1_IBM_s101
                                 + 0.740801 w_k1_INTC_s101
                                 + 0.773761 w_k1_KO_s101
                                 + 0.890731 w_k1_MAT_s101
                                 + 0.861332 w_k1_MSFT_s101
                                 + 0.910854 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.974111 w_k1_ADM_s102
                                 + 1.20785 w_k1_AXP_s102 + 1.1152 w_k1_DIS_s102
                                 + 0.979498 w_k1_FDX_s102
                                 + 1.47695 w_k1_IBM_s102
                                 + 1.30525 w_k1_INTC_s102
                                 + 1.32454 w_k1_KO_s102 + 1.9333 w_k1_MAT_s102
                                 + 1.06713 w_k1_MSFT_s102
                                 + 1.08527 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.849092 w_k1_ADM_s102
                                 + 0.978819 w_k1_AXP_s102
                                 + 1.06532 w_k1_DIS_s102
                                 + 0.98496 w_k1_FDX_s102
                                 + 0.995546 w_k1_IBM_s102
                                 + 1.02913 w_k1_INTC_s102
                                 + 1.00287 w_k1_KO_s102 + 1.5173 w_k1_MAT_s102
                                 + 1.00633 w_k1_MSFT_s102 + 1.0243 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.625957 w_k1_ADM_s102
                                 + 0.881079 w_k1_AXP_s102
                                 + 0.853193 w_k1_DIS_s102
                                 + 0.862241 w_k1_FDX_s102
                                 + 0.908596 w_k1_IBM_s102
                                 + 0.955241 w_k1_INTC_s102
                                 + 0.955155 w_k1_KO_s102
                                 + 1.08735 w_k1_MAT_s102
                                 + 0.867809 w_k1_MSFT_s102
                                 + 0.839746 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.837646 w_k1_ADM_s102
                                 + 1.52733 w_k1_AXP_s102
                                 + 1.86474 w_k1_DIS_s102
                                 + 1.23536 w_k1_FDX_s102
                                 + 1.43054 w_k1_IBM_s102
                                 + 1.33919 w_k1_INTC_s102
                                 + 1.46914 w_k1_KO_s102 + 2.44266 w_k1_MAT_s102
                                 + 1.24911 w_k1_MSFT_s102
                                 + 1.25812 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.698508 w_k1_ADM_s102
                                 + 0.565916 w_k1_AXP_s102
                                 + 0.67648 w_k1_DIS_s102
                                 + 0.766641 w_k1_FDX_s102
                                 + 0.899251 w_k1_IBM_s102
                                 + 0.858615 w_k1_INTC_s102
                                 + 0.873178 w_k1_KO_s102
                                 + 0.928788 w_k1_MAT_s102
                                 + 0.780077 w_k1_MSFT_s102
                                 + 0.774982 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.957972 w_k1_ADM_s102
                                 + 0.915419 w_k1_AXP_s102
                                 + 1.17627 w_k1_DIS_s102
                                 + 1.10999 w_k1_FDX_s102
                                 + 0.950115 w_k1_IBM_s102
                                 + 1.04985 w_k1_INTC_s102
                                 + 1.00256 w_k1_KO_s102 + 1.04704 w_k1_MAT_s102
                                 + 1.03226 w_k1_MSFT_s102
                                 + 1.04796 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.888809 w_k1_ADM_s102
                                 + 1.22883 w_k1_AXP_s102
                                 + 1.61565 w_k1_DIS_s102
                                 + 1.17748 w_k1_FDX_s102
                                 + 1.14878 w_k1_IBM_s102
                                 + 1.4061 w_k1_INTC_s102 + 1.19882 w_k1_KO_s102
                                 + 1.24108 w_k1_MAT_s102
                                 + 1.13484 w_k1_MSFT_s102
                                 + 1.21619 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.1186 w_k1_ADM_s102
                                 + 1.43194 w_k1_AXP_s102
                                 + 1.59247 w_k1_DIS_s102 + 1.4187 w_k1_FDX_s102
                                 + 0.90734 w_k1_IBM_s102
                                 + 1.41232 w_k1_INTC_s102
                                 + 1.16684 w_k1_KO_s102
                                 + 0.990142 w_k1_MAT_s102
                                 + 1.21282 w_k1_MSFT_s102
                                 + 1.13656 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.880009 w_k1_ADM_s103
                                 + 0.710188 w_k1_AXP_s103
                                 + 0.699637 w_k1_DIS_s103
                                 + 0.714885 w_k1_FDX_s103
                                 + 0.833871 w_k1_IBM_s103
                                 + 0.732855 w_k1_INTC_s103
                                 + 0.665418 w_k1_KO_s103
                                 + 0.509634 w_k1_MAT_s103
                                 + 0.729793 w_k1_MSFT_s103
                                 + 0.65507 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.21926 w_k1_ADM_s103
                                 + 1.04704 w_k1_AXP_s103
                                 + 0.991279 w_k1_DIS_s103
                                 + 1.15719 w_k1_FDX_s103
                                 + 0.696763 w_k1_IBM_s103
                                 + 1.02961 w_k1_INTC_s103
                                 + 0.834416 w_k1_KO_s103
                                 + 0.845794 w_k1_MAT_s103
                                 + 1.23152 w_k1_MSFT_s103
                                 + 0.92804 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.970761 w_k1_ADM_s103
                                 + 1.15666 w_k1_AXP_s103
                                 + 1.15078 w_k1_DIS_s103
                                 + 0.970153 w_k1_FDX_s103
                                 + 1.5112 w_k1_IBM_s103
                                 + 1.06164 w_k1_INTC_s103
                                 + 0.973868 w_k1_KO_s103
                                 + 1.14676 w_k1_MAT_s103
                                 + 1.05902 w_k1_MSFT_s103
                                 + 0.867646 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.989144 w_k1_ADM_s103
                                 + 1.39664 w_k1_AXP_s103
                                 + 0.999407 w_k1_DIS_s103
                                 + 1.05449 w_k1_FDX_s103
                                 + 1.00611 w_k1_IBM_s103
                                 + 0.908621 w_k1_INTC_s103
                                 + 0.898905 w_k1_KO_s103
                                 + 1.41247 w_k1_MAT_s103
                                 + 0.942369 w_k1_MSFT_s103
                                 + 0.959721 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.59919 w_k1_ADM_s103
                                 + 1.16066 w_k1_AXP_s103
                                 + 1.66587 w_k1_DIS_s103
                                 + 1.32219 w_k1_FDX_s103
                                 + 1.13752 w_k1_IBM_s103
                                 + 1.08373 w_k1_INTC_s103
                                 + 1.08578 w_k1_KO_s103 + 1.26811 w_k1_MAT_s103
                                 + 1.24015 w_k1_MSFT_s103
                                 + 1.00253 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.17041 w_k1_ADM_s103
                                 + 1.01402 w_k1_AXP_s103
                                 + 0.996963 w_k1_DIS_s103
                                 + 1.05895 w_k1_FDX_s103
                                 + 1.12254 w_k1_IBM_s103
                                 + 0.969893 w_k1_INTC_s103
                                 + 1.00309 w_k1_KO_s103 + 1.31141 w_k1_MAT_s103
                                 + 0.978814 w_k1_MSFT_s103
                                 + 1.03629 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.05788 w_k1_ADM_s103
                                 + 0.805284 w_k1_AXP_s103
                                 + 0.745513 w_k1_DIS_s103
                                 + 0.770145 w_k1_FDX_s103
                                 + 1.15317 w_k1_IBM_s103
                                 + 0.742769 w_k1_INTC_s103
                                 + 0.926978 w_k1_KO_s103
                                 + 1.15533 w_k1_MAT_s103
                                 + 0.798107 w_k1_MSFT_s103
                                 + 0.988644 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.07199 w_k1_ADM_s103
                                 + 0.910411 w_k1_AXP_s103
                                 + 1.0074 w_k1_DIS_s103 + 0.96394 w_k1_FDX_s103
                                 + 0.837894 w_k1_IBM_s103
                                 + 0.902034 w_k1_INTC_s103
                                 + 0.760759 w_k1_KO_s103
                                 + 0.619231 w_k1_MAT_s103
                                 + 0.895766 w_k1_MSFT_s103
                                 + 0.776245 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.1555 w_k1_ADM_s104
                                 + 1.24895 w_k1_AXP_s104
                                 + 1.27664 w_k1_DIS_s104
                                 + 1.26089 w_k1_FDX_s104
                                 + 0.935459 w_k1_IBM_s104
                                 + 1.16518 w_k1_INTC_s104
                                 + 0.948101 w_k1_KO_s104
                                 + 1.13258 w_k1_MAT_s104
                                 + 1.10414 w_k1_MSFT_s104
                                 + 1.01017 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.821833 w_k1_ADM_s104
                                 + 0.858734 w_k1_AXP_s104
                                 + 0.985498 w_k1_DIS_s104
                                 + 0.781617 w_k1_FDX_s104
                                 + 1.32337 w_k1_IBM_s104
                                 + 0.813699 w_k1_INTC_s104
                                 + 0.830518 w_k1_KO_s104
                                 + 1.14565 w_k1_MAT_s104
                                 + 0.754471 w_k1_MSFT_s104
                                 + 0.888989 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.945343 w_k1_ADM_s104
                                 + 0.84938 w_k1_AXP_s104
                                 + 0.983751 w_k1_DIS_s104
                                 + 0.931401 w_k1_FDX_s104
                                 + 1.0522 w_k1_IBM_s104
                                 + 0.905005 w_k1_INTC_s104
                                 + 1.00919 w_k1_KO_s104
                                 + 0.629535 w_k1_MAT_s104
                                 + 0.921552 w_k1_MSFT_s104
                                 + 1.0225 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.10833 w_k1_ADM_s104
                                 + 1.47753 w_k1_AXP_s104
                                 + 1.29546 w_k1_DIS_s104
                                 + 1.16192 w_k1_FDX_s104
                                 + 1.36476 w_k1_IBM_s104
                                 + 0.977149 w_k1_INTC_s104
                                 + 1.11915 w_k1_KO_s104 + 1.64075 w_k1_MAT_s104
                                 + 0.983178 w_k1_MSFT_s104
                                 + 1.12093 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.851773 w_k1_ADM_s104
                                 + 1.0655 w_k1_AXP_s104
                                 + 0.790317 w_k1_DIS_s104
                                 + 0.988854 w_k1_FDX_s104
                                 + 1.13344 w_k1_IBM_s104
                                 + 0.753925 w_k1_INTC_s104
                                 + 0.803796 w_k1_KO_s104
                                 + 1.44329 w_k1_MAT_s104
                                 + 0.769842 w_k1_MSFT_s104
                                 + 0.993868 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.714033 w_k1_ADM_s104
                                 + 0.759923 w_k1_AXP_s104
                                 + 0.669746 w_k1_DIS_s104
                                 + 0.844856 w_k1_FDX_s104
                                 + 0.78862 w_k1_IBM_s104
                                 + 0.754579 w_k1_INTC_s104
                                 + 0.648655 w_k1_KO_s104
                                 + 0.690819 w_k1_MAT_s104
                                 + 0.790981 w_k1_MSFT_s104
                                 + 0.875145 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.37389 w_k1_ADM_s104
                                 + 1.14789 w_k1_AXP_s104
                                 + 0.809279 w_k1_DIS_s104
                                 + 1.12931 w_k1_FDX_s104
                                 + 0.827027 w_k1_IBM_s104
                                 + 0.930055 w_k1_INTC_s104
                                 + 0.698435 w_k1_KO_s104
                                 + 0.83713 w_k1_MAT_s104
                                 + 0.824448 w_k1_MSFT_s104
                                 + 0.903312 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.742148 w_k1_ADM_s104
                                 + 0.763645 w_k1_AXP_s104
                                 + 0.805987 w_k1_DIS_s104
                                 + 0.91464 w_k1_FDX_s104
                                 + 0.642093 w_k1_IBM_s104
                                 + 0.818304 w_k1_INTC_s104
                                 + 0.682491 w_k1_KO_s104
                                 + 0.604934 w_k1_MAT_s104
                                 + 0.830806 w_k1_MSFT_s104
                                 + 0.864573 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.03155 w_k1_ADM_s105
                                 + 1.12338 w_k1_AXP_s105
                                 + 1.19569 w_k1_DIS_s105
                                 + 1.29354 w_k1_FDX_s105
                                 + 0.727058 w_k1_IBM_s105
                                 + 1.10331 w_k1_INTC_s105
                                 + 0.81075 w_k1_KO_s105
                                 + 0.976716 w_k1_MAT_s105
                                 + 1.16388 w_k1_MSFT_s105
                                 + 0.968958 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.783576 w_k1_ADM_s105
                                 + 0.939423 w_k1_AXP_s105
                                 + 0.908076 w_k1_DIS_s105
                                 + 0.926827 w_k1_FDX_s105
                                 + 0.858843 w_k1_IBM_s105
                                 + 0.908336 w_k1_INTC_s105
                                 + 0.879666 w_k1_KO_s105
                                 + 0.904335 w_k1_MAT_s105
                                 + 0.836113 w_k1_MSFT_s105
                                 + 0.919895 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.725736 w_k1_ADM_s105
                                 + 0.684552 w_k1_AXP_s105
                                 + 0.6593 w_k1_DIS_s105
                                 + 0.720644 w_k1_FDX_s105
                                 + 0.909449 w_k1_IBM_s105
                                 + 0.854389 w_k1_INTC_s105
                                 + 0.799864 w_k1_KO_s105
                                 + 0.863702 w_k1_MAT_s105
                                 + 0.815467 w_k1_MSFT_s105
                                 + 0.842857 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.04435 w_k1_ADM_s105
                                 + 0.834968 w_k1_AXP_s105
                                 + 1.01281 w_k1_DIS_s105
                                 + 0.93446 w_k1_FDX_s105
                                 + 1.14132 w_k1_IBM_s105
                                 + 1.07852 w_k1_INTC_s105
                                 + 1.01166 w_k1_KO_s105 + 1.05779 w_k1_MAT_s105
                                 + 1.04138 w_k1_MSFT_s105
                                 + 0.885049 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.17517 w_k1_ADM_s105
                                 + 1.48309 w_k1_AXP_s105
                                 + 1.37635 w_k1_DIS_s105
                                 + 1.15715 w_k1_FDX_s105
                                 + 1.52246 w_k1_IBM_s105
                                 + 1.25283 w_k1_INTC_s105
                                 + 1.25478 w_k1_KO_s105 + 1.96826 w_k1_MAT_s105
                                 + 1.21551 w_k1_MSFT_s105
                                 + 1.27332 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.822689 w_k1_ADM_s105
                                 + 0.750484 w_k1_AXP_s105
                                 + 0.796714 w_k1_DIS_s105
                                 + 0.818508 w_k1_FDX_s105
                                 + 0.645718 w_k1_IBM_s105
                                 + 0.73576 w_k1_INTC_s105
                                 + 0.675571 w_k1_KO_s105
                                 + 0.620272 w_k1_MAT_s105
                                 + 0.78446 w_k1_MSFT_s105
                                 + 0.839397 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.980303 w_k1_ADM_s105
                                 + 1.16229 w_k1_AXP_s105
                                 + 1.17468 w_k1_DIS_s105
                                 + 1.13304 w_k1_FDX_s105
                                 + 1.02994 w_k1_IBM_s105
                                 + 1.14311 w_k1_INTC_s105
                                 + 1.14303 w_k1_KO_s105 + 1.49067 w_k1_MAT_s105
                                 + 1.30013 w_k1_MSFT_s105
                                 + 1.08102 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.6704 w_k1_ADM_s105
                                 + 1.00838 w_k1_AXP_s105
                                 + 0.823278 w_k1_DIS_s105
                                 + 0.798107 w_k1_FDX_s105
                                 + 1.28549 w_k1_IBM_s105
                                 + 0.993001 w_k1_INTC_s105
                                 + 0.984799 w_k1_KO_s105
                                 + 1.23255 w_k1_MAT_s105
                                 + 0.856886 w_k1_MSFT_s105
                                 + 0.991916 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.886547 w_k1_ADM_s106
                                 + 1.24562 w_k1_AXP_s106
                                 + 0.988891 w_k1_DIS_s106
                                 + 0.903944 w_k1_FDX_s106
                                 + 1.18195 w_k1_IBM_s106
                                 + 0.922888 w_k1_INTC_s106
                                 + 1.13013 w_k1_KO_s106 + 1.36799 w_k1_MAT_s106
                                 + 1.02736 w_k1_MSFT_s106
                                 + 1.11594 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.912926 w_k1_ADM_s106
                                 + 1.22735 w_k1_AXP_s106
                                 + 1.26822 w_k1_DIS_s106
                                 + 1.15075 w_k1_FDX_s106
                                 + 1.11067 w_k1_IBM_s106 + 1.33 w_k1_INTC_s106
                                 + 1.20352 w_k1_KO_s106 + 1.12241 w_k1_MAT_s106
                                 + 1.16167 w_k1_MSFT_s106
                                 + 1.21725 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.25417 w_k1_ADM_s106
                                 + 1.37367 w_k1_AXP_s106
                                 + 1.33477 w_k1_DIS_s106
                                 + 1.45616 w_k1_FDX_s106
                                 + 0.674651 w_k1_IBM_s106
                                 + 1.52697 w_k1_INTC_s106
                                 + 1.08513 w_k1_KO_s106
                                 + 0.623839 w_k1_MAT_s106
                                 + 1.43702 w_k1_MSFT_s106
                                 + 1.18398 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.737298 w_k1_ADM_s106
                                 + 0.655496 w_k1_AXP_s106
                                 + 0.585712 w_k1_DIS_s106
                                 + 0.736728 w_k1_FDX_s106
                                 + 0.693953 w_k1_IBM_s106
                                 + 0.767934 w_k1_INTC_s106
                                 + 0.707934 w_k1_KO_s106
                                 + 0.756749 w_k1_MAT_s106
                                 + 0.856225 w_k1_MSFT_s106
                                 + 0.90585 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.922783 w_k1_ADM_s106
                                 + 0.779813 w_k1_AXP_s106
                                 + 0.727647 w_k1_DIS_s106
                                 + 0.75748 w_k1_FDX_s106
                                 + 1.14661 w_k1_IBM_s106
                                 + 0.841074 w_k1_INTC_s106
                                 + 1.08245 w_k1_KO_s106
                                 + 0.914827 w_k1_MAT_s106
                                 + 0.799189 w_k1_MSFT_s106
                                 + 1.04315 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.05331 w_k1_ADM_s106
                                 + 1.37987 w_k1_AXP_s106
                                 + 0.79496 w_k1_DIS_s106
                                 + 0.990792 w_k1_FDX_s106
                                 + 1.07718 w_k1_IBM_s106
                                 + 0.97125 w_k1_INTC_s106
                                 + 0.833834 w_k1_KO_s106
                                 + 1.03798 w_k1_MAT_s106
                                 + 0.823511 w_k1_MSFT_s106
                                 + 1.07417 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.01398 w_k1_ADM_s106
                                 + 0.772855 w_k1_AXP_s106
                                 + 0.832591 w_k1_DIS_s106
                                 + 0.910532 w_k1_FDX_s106
                                 + 0.732416 w_k1_IBM_s106
                                 + 0.979296 w_k1_INTC_s106
                                 + 0.844162 w_k1_KO_s106
                                 + 0.584633 w_k1_MAT_s106
                                 + 1.01427 w_k1_MSFT_s106 + 1.0264 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.13374 w_k1_ADM_s106
                                 + 1.15319 w_k1_AXP_s106
                                 + 0.990946 w_k1_DIS_s106
                                 + 1.17206 w_k1_FDX_s106
                                 + 0.773867 w_k1_IBM_s106
                                 + 0.968127 w_k1_INTC_s106
                                 + 1.12002 w_k1_KO_s106
                                 + 0.828827 w_k1_MAT_s106
                                 + 1.16971 w_k1_MSFT_s106
                                 + 1.18561 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.05825 w_k1_ADM_s107
                                 + 1.26917 w_k1_AXP_s107
                                 + 1.22379 w_k1_DIS_s107
                                 + 1.02248 w_k1_FDX_s107
                                 + 1.73207 w_k1_IBM_s107
                                 + 0.942119 w_k1_INTC_s107
                                 + 1.38194 w_k1_KO_s107 + 1.83919 w_k1_MAT_s107
                                 + 1.03384 w_k1_MSFT_s107
                                 + 1.05562 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.26144 w_k1_ADM_s107
                                 + 1.22158 w_k1_AXP_s107
                                 + 1.29454 w_k1_DIS_s107
                                 + 1.10153 w_k1_FDX_s107
                                 + 1.27948 w_k1_IBM_s107
                                 + 1.10955 w_k1_INTC_s107
                                 + 1.11314 w_k1_KO_s107
                                 + 0.879234 w_k1_MAT_s107
                                 + 1.07742 w_k1_MSFT_s107 + 1.1635 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.42734 w_k1_ADM_s107
                                 + 1.60164 w_k1_AXP_s107
                                 + 1.77423 w_k1_DIS_s107
                                 + 1.51593 w_k1_FDX_s107
                                 + 0.954026 w_k1_IBM_s107
                                 + 1.22359 w_k1_INTC_s107
                                 + 1.20152 w_k1_KO_s107 + 1.16204 w_k1_MAT_s107
                                 + 1.34513 w_k1_MSFT_s107
                                 + 1.21051 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.854388 w_k1_ADM_s107
                                 + 0.837661 w_k1_AXP_s107
                                 + 0.728652 w_k1_DIS_s107
                                 + 0.805918 w_k1_FDX_s107
                                 + 0.927789 w_k1_IBM_s107
                                 + 0.744872 w_k1_INTC_s107
                                 + 0.797348 w_k1_KO_s107
                                 + 0.701357 w_k1_MAT_s107
                                 + 0.778833 w_k1_MSFT_s107
                                 + 0.765845 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.35786 w_k1_ADM_s107
                                 + 1.54428 w_k1_AXP_s107 + 1.5284 w_k1_DIS_s107
                                 + 1.32237 w_k1_FDX_s107
                                 + 1.26276 w_k1_IBM_s107
                                 + 1.08224 w_k1_INTC_s107
                                 + 1.26051 w_k1_KO_s107 + 1.18071 w_k1_MAT_s107
                                 + 1.29202 w_k1_MSFT_s107
                                 + 1.16813 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.09188 w_k1_ADM_s107
                                 + 0.816409 w_k1_AXP_s107
                                 + 0.788586 w_k1_DIS_s107
                                 + 0.942571 w_k1_FDX_s107
                                 + 0.636742 w_k1_IBM_s107
                                 + 0.868467 w_k1_INTC_s107
                                 + 0.754052 w_k1_KO_s107
                                 + 0.486251 w_k1_MAT_s107
                                 + 0.970562 w_k1_MSFT_s107
                                 + 0.776861 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.06897 w_k1_ADM_s107
                                 + 1.15164 w_k1_AXP_s107
                                 + 1.38526 w_k1_DIS_s107
                                 + 1.10756 w_k1_FDX_s107
                                 + 0.948319 w_k1_IBM_s107
                                 + 1.02359 w_k1_INTC_s107
                                 + 1.05888 w_k1_KO_s107 + 1.0094 w_k1_MAT_s107
                                 + 1.06927 w_k1_MSFT_s107
                                 + 1.03145 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.12553 w_k1_ADM_s107
                                 + 1.05381 w_k1_AXP_s107
                                 + 1.02337 w_k1_DIS_s107
                                 + 0.961093 w_k1_FDX_s107
                                 + 1.06505 w_k1_IBM_s107
                                 + 0.666539 w_k1_INTC_s107
                                 + 0.933624 w_k1_KO_s107
                                 + 0.955577 w_k1_MAT_s107
                                 + 0.957299 w_k1_MSFT_s107
                                 + 1.05627 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.004 w_k1_ADM_s110
                                 + 1.06302 w_k1_AXP_s110
                                 + 1.21187 w_k1_DIS_s110
                                 + 1.06922 w_k1_FDX_s110
                                 + 1.03981 w_k1_IBM_s110
                                 + 1.20144 w_k1_INTC_s110
                                 + 1.02326 w_k1_KO_s110 + 1.18453 w_k1_MAT_s110
                                 + 1.08691 w_k1_MSFT_s110
                                 + 1.07546 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.05774 w_k1_ADM_s110
                                 + 0.961614 w_k1_AXP_s110
                                 + 1.3595 w_k1_DIS_s110 + 1.08029 w_k1_FDX_s110
                                 + 1.63168 w_k1_IBM_s110
                                 + 1.11593 w_k1_INTC_s110
                                 + 1.44827 w_k1_KO_s110 + 1.57655 w_k1_MAT_s110
                                 + 1.14733 w_k1_MSFT_s110 + 1.1503 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.818069 w_k1_ADM_s110
                                 + 0.582059 w_k1_AXP_s110
                                 + 0.69956 w_k1_DIS_s110
                                 + 0.839785 w_k1_FDX_s110
                                 + 0.587186 w_k1_IBM_s110
                                 + 0.905193 w_k1_INTC_s110
                                 + 0.611666 w_k1_KO_s110
                                 + 0.583186 w_k1_MAT_s110
                                 + 0.806561 w_k1_MSFT_s110
                                 + 0.852474 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.002 w_k1_ADM_s110
                                 + 0.961838 w_k1_AXP_s110
                                 + 0.953605 w_k1_DIS_s110
                                 + 1.06218 w_k1_FDX_s110
                                 + 0.747323 w_k1_IBM_s110
                                 + 1.17755 w_k1_INTC_s110
                                 + 0.881281 w_k1_KO_s110
                                 + 0.884081 w_k1_MAT_s110
                                 + 0.997309 w_k1_MSFT_s110
                                 + 0.981333 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.838123 w_k1_ADM_s110
                                 + 0.628891 w_k1_AXP_s110
                                 + 0.679695 w_k1_DIS_s110
                                 + 0.693367 w_k1_FDX_s110
                                 + 0.858988 w_k1_IBM_s110
                                 + 0.85957 w_k1_INTC_s110
                                 + 0.733819 w_k1_KO_s110
                                 + 0.697912 w_k1_MAT_s110
                                 + 0.78211 w_k1_MSFT_s110
                                 + 0.794298 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.00258 w_k1_ADM_s110
                                 + 0.83851 w_k1_AXP_s110
                                 + 0.991139 w_k1_DIS_s110
                                 + 0.950633 w_k1_FDX_s110
                                 + 1.2478 w_k1_IBM_s110
                                 + 1.15101 w_k1_INTC_s110
                                 + 0.996996 w_k1_KO_s110
                                 + 1.24119 w_k1_MAT_s110
                                 + 1.00751 w_k1_MSFT_s110
                                 + 0.931337 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.892897 w_k1_ADM_s110
                                 + 0.820411 w_k1_AXP_s110
                                 + 0.803864 w_k1_DIS_s110
                                 + 0.990995 w_k1_FDX_s110
                                 + 0.805723 w_k1_IBM_s110
                                 + 1.13984 w_k1_INTC_s110
                                 + 0.745862 w_k1_KO_s110
                                 + 0.861734 w_k1_MAT_s110
                                 + 0.85635 w_k1_MSFT_s110
                                 + 0.934526 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.11352 w_k1_ADM_s110
                                 + 1.31163 w_k1_AXP_s110
                                 + 1.47062 w_k1_DIS_s110
                                 + 1.43236 w_k1_FDX_s110
                                 + 1.10603 w_k1_IBM_s110
                                 + 1.40613 w_k1_INTC_s110
                                 + 1.35024 w_k1_KO_s110 + 1.60565 w_k1_MAT_s110
                                 + 1.2223 w_k1_MSFT_s110 + 1.30091 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.861707 w_k1_ADM_s111
                                 + 0.8392 w_k1_AXP_s111
                                 + 0.701019 w_k1_DIS_s111
                                 + 0.784343 w_k1_FDX_s111
                                 + 0.765382 w_k1_IBM_s111
                                 + 0.910994 w_k1_INTC_s111
                                 + 0.817379 w_k1_KO_s111
                                 + 1.02046 w_k1_MAT_s111
                                 + 0.776364 w_k1_MSFT_s111
                                 + 0.923199 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.892101 w_k1_ADM_s111
                                 + 0.940697 w_k1_AXP_s111
                                 + 0.771738 w_k1_DIS_s111
                                 + 0.909494 w_k1_FDX_s111
                                 + 0.646544 w_k1_IBM_s111
                                 + 1.13701 w_k1_INTC_s111
                                 + 0.818471 w_k1_KO_s111
                                 + 0.880242 w_k1_MAT_s111
                                 + 0.917137 w_k1_MSFT_s111
                                 + 0.932196 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.09363 w_k1_ADM_s111
                                 + 1.62641 w_k1_AXP_s111 + 1.5271 w_k1_DIS_s111
                                 + 1.28794 w_k1_FDX_s111
                                 + 1.44664 w_k1_IBM_s111
                                 + 1.19595 w_k1_INTC_s111
                                 + 1.25201 w_k1_KO_s111 + 1.69087 w_k1_MAT_s111
                                 + 1.07208 w_k1_MSFT_s111
                                 + 1.15981 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.11646 w_k1_ADM_s111
                                 + 1.32494 w_k1_AXP_s111
                                 + 1.09384 w_k1_DIS_s111
                                 + 0.98129 w_k1_FDX_s111
                                 + 1.24347 w_k1_IBM_s111
                                 + 1.30963 w_k1_INTC_s111
                                 + 1.15099 w_k1_KO_s111 + 1.46134 w_k1_MAT_s111
                                 + 0.872735 w_k1_MSFT_s111
                                 + 1.0818 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.873854 w_k1_ADM_s111
                                 + 1.06723 w_k1_AXP_s111
                                 + 0.787744 w_k1_DIS_s111
                                 + 0.799002 w_k1_FDX_s111
                                 + 1.08382 w_k1_IBM_s111
                                 + 0.952684 w_k1_INTC_s111
                                 + 0.880131 w_k1_KO_s111
                                 + 1.03741 w_k1_MAT_s111
                                 + 0.827364 w_k1_MSFT_s111
                                 + 0.870416 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.09784 w_k1_ADM_s111
                                 + 1.29747 w_k1_AXP_s111 + 1.362 w_k1_DIS_s111
                                 + 1.22568 w_k1_FDX_s111
                                 + 1.12514 w_k1_IBM_s111
                                 + 1.31044 w_k1_INTC_s111 + 1.167 w_k1_KO_s111
                                 + 1.44452 w_k1_MAT_s111
                                 + 1.05002 w_k1_MSFT_s111 + 1.2743 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.31495 w_k1_ADM_s111
                                 + 1.41367 w_k1_AXP_s111 + 1.2534 w_k1_DIS_s111
                                 + 1.17656 w_k1_FDX_s111
                                 + 0.878946 w_k1_IBM_s111
                                 + 1.3432 w_k1_INTC_s111 + 1.01458 w_k1_KO_s111
                                 + 0.761854 w_k1_MAT_s111
                                 + 1.16665 w_k1_MSFT_s111 + 1.1242 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05776 w_k1_ADM_s111
                                 + 1.1841 w_k1_AXP_s111
                                 + 0.973288 w_k1_DIS_s111
                                 + 0.97181 w_k1_FDX_s111
                                 + 0.907795 w_k1_IBM_s111
                                 + 1.0414 w_k1_INTC_s111
                                 + 0.868355 w_k1_KO_s111 + 1.0365 w_k1_MAT_s111
                                 + 0.970339 w_k1_MSFT_s111
                                 + 0.951897 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.35886 w_k1_ADM_s112
                                 + 1.27101 w_k1_AXP_s112
                                 + 1.53028 w_k1_DIS_s112
                                 + 1.37429 w_k1_FDX_s112
                                 + 1.33322 w_k1_IBM_s112
                                 + 1.24745 w_k1_INTC_s112
                                 + 1.32638 w_k1_KO_s112 + 1.49294 w_k1_MAT_s112
                                 + 1.29289 w_k1_MSFT_s112
                                 + 1.24695 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.903176 w_k1_ADM_s112
                                 + 0.809258 w_k1_AXP_s112
                                 + 0.854461 w_k1_DIS_s112
                                 + 0.902469 w_k1_FDX_s112
                                 + 0.740763 w_k1_IBM_s112
                                 + 0.928298 w_k1_INTC_s112
                                 + 0.840105 w_k1_KO_s112
                                 + 0.605694 w_k1_MAT_s112
                                 + 0.98328 w_k1_MSFT_s112
                                 + 0.937958 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.04163 w_k1_ADM_s112
                                 + 0.747277 w_k1_AXP_s112
                                 + 0.764219 w_k1_DIS_s112
                                 + 0.788427 w_k1_FDX_s112
                                 + 1.31127 w_k1_IBM_s112
                                 + 0.786891 w_k1_INTC_s112
                                 + 1.02606 w_k1_KO_s112 + 1.26061 w_k1_MAT_s112
                                 + 0.917396 w_k1_MSFT_s112
                                 + 1.00019 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.26888 w_k1_ADM_s112
                                 + 1.10829 w_k1_AXP_s112
                                 + 1.22614 w_k1_DIS_s112
                                 + 1.18625 w_k1_FDX_s112
                                 + 0.890476 w_k1_IBM_s112
                                 + 0.795717 w_k1_INTC_s112
                                 + 1.03631 w_k1_KO_s112 + 1.10483 w_k1_MAT_s112
                                 + 1.16408 w_k1_MSFT_s112
                                 + 1.09213 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.10754 w_k1_ADM_s112
                                 + 0.983266 w_k1_AXP_s112
                                 + 1.19227 w_k1_DIS_s112
                                 + 1.03589 w_k1_FDX_s112
                                 + 1.27263 w_k1_IBM_s112
                                 + 1.06629 w_k1_INTC_s112
                                 + 1.26171 w_k1_KO_s112 + 1.04971 w_k1_MAT_s112
                                 + 1.23467 w_k1_MSFT_s112 + 1.0674 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.704364 w_k1_ADM_s112
                                 + 0.548741 w_k1_AXP_s112
                                 + 0.584959 w_k1_DIS_s112
                                 + 0.694233 w_k1_FDX_s112
                                 + 0.604687 w_k1_IBM_s112
                                 + 0.723337 w_k1_INTC_s112
                                 + 0.631899 w_k1_KO_s112
                                 + 0.481513 w_k1_MAT_s112
                                 + 0.761162 w_k1_MSFT_s112
                                 + 0.731486 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.80186 w_k1_ADM_s112
                                 + 1.00945 w_k1_AXP_s112
                                 + 1.09438 w_k1_DIS_s112
                                 + 0.903755 w_k1_FDX_s112
                                 + 1.82129 w_k1_IBM_s112
                                 + 0.894857 w_k1_INTC_s112
                                 + 1.4355 w_k1_KO_s112 + 1.75777 w_k1_MAT_s112
                                 + 0.946381 w_k1_MSFT_s112
                                 + 1.09509 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.754628 w_k1_ADM_s112
                                 + 0.879095 w_k1_AXP_s112
                                 + 0.838949 w_k1_DIS_s112
                                 + 0.782731 w_k1_FDX_s112
                                 + 1.03623 w_k1_IBM_s112
                                 + 0.809725 w_k1_INTC_s112
                                 + 0.908716 w_k1_KO_s112
                                 + 1.16003 w_k1_MAT_s112
                                 + 0.780487 w_k1_MSFT_s112
                                 + 0.922857 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.657518 w_k1_ADM_s113
                                 + 0.599087 w_k1_AXP_s113
                                 + 0.72351 w_k1_DIS_s113
                                 + 0.847227 w_k1_FDX_s113
                                 + 0.752098 w_k1_IBM_s113
                                 + 1.15636 w_k1_INTC_s113
                                 + 0.797333 w_k1_KO_s113
                                 + 0.547801 w_k1_MAT_s113
                                 + 0.893936 w_k1_MSFT_s113
                                 + 0.763259 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.83836 w_k1_ADM_s113
                                 + 0.82318 w_k1_AXP_s113
                                 + 0.879829 w_k1_DIS_s113
                                 + 0.911186 w_k1_FDX_s113
                                 + 1.20214 w_k1_IBM_s113
                                 + 0.911889 w_k1_INTC_s113
                                 + 0.941533 w_k1_KO_s113
                                 + 1.13182 w_k1_MAT_s113
                                 + 0.848516 w_k1_MSFT_s113
                                 + 0.857207 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.720535 w_k1_ADM_s113
                                 + 0.65479 w_k1_AXP_s113
                                 + 0.545641 w_k1_DIS_s113
                                 + 0.74508 w_k1_FDX_s113
                                 + 0.81748 w_k1_IBM_s113
                                 + 0.933775 w_k1_INTC_s113
                                 + 0.73257 w_k1_KO_s113
                                 + 0.644482 w_k1_MAT_s113
                                 + 0.68305 w_k1_MSFT_s113
                                 + 0.790859 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11177 w_k1_ADM_s113
                                 + 1.1073 w_k1_AXP_s113 + 1.20481 w_k1_DIS_s113
                                 + 1.11006 w_k1_FDX_s113
                                 + 1.65792 w_k1_IBM_s113
                                 + 1.0441 w_k1_INTC_s113 + 1.32518 w_k1_KO_s113
                                 + 1.63674 w_k1_MAT_s113
                                 + 1.09172 w_k1_MSFT_s113
                                 + 1.11799 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.34447 w_k1_ADM_s113
                                 + 1.27814 w_k1_AXP_s113
                                 + 1.49349 w_k1_DIS_s113
                                 + 1.45708 w_k1_FDX_s113
                                 + 1.05793 w_k1_IBM_s113
                                 + 1.56564 w_k1_INTC_s113
                                 + 1.23187 w_k1_KO_s113 + 1.0393 w_k1_MAT_s113
                                 + 1.26881 w_k1_MSFT_s113
                                 + 1.13647 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.789713 w_k1_ADM_s113
                                 + 0.814229 w_k1_AXP_s113
                                 + 0.640837 w_k1_DIS_s113
                                 + 0.985058 w_k1_FDX_s113
                                 + 0.613868 w_k1_IBM_s113
                                 + 1.02716 w_k1_INTC_s113
                                 + 0.687019 w_k1_KO_s113
                                 + 0.687663 w_k1_MAT_s113
                                 + 0.873051 w_k1_MSFT_s113
                                 + 0.842531 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.837459 w_k1_ADM_s113
                                 + 0.855828 w_k1_AXP_s113
                                 + 1.11774 w_k1_DIS_s113
                                 + 1.14645 w_k1_FDX_s113
                                 + 0.711023 w_k1_IBM_s113
                                 + 1.21947 w_k1_INTC_s113
                                 + 0.968064 w_k1_KO_s113
                                 + 0.915251 w_k1_MAT_s113
                                 + 1.14542 w_k1_MSFT_s113
                                 + 0.906499 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.08166 w_k1_ADM_s113
                                 + 1.02173 w_k1_AXP_s113
                                 + 0.998389 w_k1_DIS_s113
                                 + 1.14231 w_k1_FDX_s113
                                 + 0.995101 w_k1_IBM_s113
                                 + 1.10439 w_k1_INTC_s113 + 0.8926 w_k1_KO_s113
                                 + 0.868518 w_k1_MAT_s113
                                 + 0.963336 w_k1_MSFT_s113
                                 + 0.985126 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09597 w_k1_ADM_s114
                                 + 1.10084 w_k1_AXP_s114
                                 + 1.54797 w_k1_DIS_s114
                                 + 1.19137 w_k1_FDX_s114
                                 + 1.09202 w_k1_IBM_s114
                                 + 1.2046 w_k1_INTC_s114 + 1.21994 w_k1_KO_s114
                                 + 1.20466 w_k1_MAT_s114
                                 + 1.33137 w_k1_MSFT_s114
                                 + 1.22911 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.753644 w_k1_ADM_s114
                                 + 0.666953 w_k1_AXP_s114
                                 + 0.770491 w_k1_DIS_s114
                                 + 0.809596 w_k1_FDX_s114
                                 + 0.686338 w_k1_IBM_s114
                                 + 0.91181 w_k1_INTC_s114
                                 + 0.747298 w_k1_KO_s114
                                 + 0.64206 w_k1_MAT_s114
                                 + 0.886025 w_k1_MSFT_s114
                                 + 0.865592 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.11706 w_k1_ADM_s114
                                 + 1.09116 w_k1_AXP_s114
                                 + 1.17073 w_k1_DIS_s114
                                 + 1.26358 w_k1_FDX_s114
                                 + 0.835546 w_k1_IBM_s114
                                 + 1.22334 w_k1_INTC_s114
                                 + 0.862532 w_k1_KO_s114
                                 + 0.863487 w_k1_MAT_s114
                                 + 1.07399 w_k1_MSFT_s114
                                 + 1.19722 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.871535 w_k1_ADM_s114
                                 + 0.712858 w_k1_AXP_s114
                                 + 0.978152 w_k1_DIS_s114
                                 + 0.81213 w_k1_FDX_s114
                                 + 1.17613 w_k1_IBM_s114
                                 + 0.820575 w_k1_INTC_s114
                                 + 1.1167 w_k1_KO_s114 + 0.999581 w_k1_MAT_s114
                                 + 0.798313 w_k1_MSFT_s114
                                 + 0.936416 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.877845 w_k1_ADM_s114
                                 + 1.18508 w_k1_AXP_s114
                                 + 1.06754 w_k1_DIS_s114
                                 + 1.03721 w_k1_FDX_s114
                                 + 1.04332 w_k1_IBM_s114
                                 + 0.993021 w_k1_INTC_s114
                                 + 1.07564 w_k1_KO_s114 + 1.03385 w_k1_MAT_s114
                                 + 0.928788 w_k1_MSFT_s114
                                 + 1.04896 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.01603 w_k1_ADM_s114
                                 + 1.25535 w_k1_AXP_s114
                                 + 1.65667 w_k1_DIS_s114 + 1.2053 w_k1_FDX_s114
                                 + 1.64714 w_k1_IBM_s114
                                 + 1.19713 w_k1_INTC_s114
                                 + 1.52283 w_k1_KO_s114 + 1.53289 w_k1_MAT_s114
                                 + 1.25885 w_k1_MSFT_s114
                                 + 1.29485 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.09062 w_k1_ADM_s114
                                 + 1.34784 w_k1_AXP_s114
                                 + 1.17391 w_k1_DIS_s114
                                 + 1.04389 w_k1_FDX_s114
                                 + 1.40437 w_k1_IBM_s114
                                 + 0.993368 w_k1_INTC_s114
                                 + 1.21863 w_k1_KO_s114 + 1.45268 w_k1_MAT_s114
                                 + 0.941925 w_k1_MSFT_s114
                                 + 1.12944 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.934705 w_k1_ADM_s114
                                 + 0.830977 w_k1_AXP_s114
                                 + 0.9155 w_k1_DIS_s114
                                 + 0.975523 w_k1_FDX_s114
                                 + 0.879186 w_k1_IBM_s114
                                 + 0.996746 w_k1_INTC_s114
                                 + 0.902601 w_k1_KO_s114
                                 + 0.839096 w_k1_MAT_s114
                                 + 0.885641 w_k1_MSFT_s114
                                 + 1.06012 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.26064 w_k1_ADM_s115
                                 + 1.45732 w_k1_AXP_s115
                                 + 1.63538 w_k1_DIS_s115
                                 + 1.19734 w_k1_FDX_s115
                                 + 1.72139 w_k1_IBM_s115
                                 + 1.0758 w_k1_INTC_s115 + 1.46509 w_k1_KO_s115
                                 + 1.64944 w_k1_MAT_s115
                                 + 1.05493 w_k1_MSFT_s115
                                 + 1.19454 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.903245 w_k1_ADM_s115
                                 + 1.06349 w_k1_AXP_s115
                                 + 0.722395 w_k1_DIS_s115
                                 + 0.895993 w_k1_FDX_s115
                                 + 1.16333 w_k1_IBM_s115
                                 + 0.926641 w_k1_INTC_s115
                                 + 0.932706 w_k1_KO_s115
                                 + 0.95774 w_k1_MAT_s115
                                 + 0.934878 w_k1_MSFT_s115
                                 + 0.952854 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.18978 w_k1_ADM_s115
                                 + 1.04202 w_k1_AXP_s115
                                 + 1.20271 w_k1_DIS_s115
                                 + 1.20058 w_k1_FDX_s115
                                 + 1.23366 w_k1_IBM_s115
                                 + 1.30905 w_k1_INTC_s115
                                 + 1.26945 w_k1_KO_s115 + 1.11712 w_k1_MAT_s115
                                 + 1.34316 w_k1_MSFT_s115
                                 + 1.07555 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06626 w_k1_ADM_s115
                                 + 0.849243 w_k1_AXP_s115
                                 + 1.0737 w_k1_DIS_s115 + 1.12388 w_k1_FDX_s115
                                 + 0.654018 w_k1_IBM_s115
                                 + 1.13949 w_k1_INTC_s115
                                 + 1.00368 w_k1_KO_s115
                                 + 0.678825 w_k1_MAT_s115
                                 + 1.32103 w_k1_MSFT_s115
                                 + 1.07193 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.940502 w_k1_ADM_s115
                                 + 1.05327 w_k1_AXP_s115
                                 + 1.25668 w_k1_DIS_s115
                                 + 1.15814 w_k1_FDX_s115
                                 + 0.941329 w_k1_IBM_s115
                                 + 1.26774 w_k1_INTC_s115
                                 + 1.14751 w_k1_KO_s115
                                 + 0.937169 w_k1_MAT_s115
                                 + 1.21403 w_k1_MSFT_s115 + 1.1572 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.692875 w_k1_ADM_s115
                                 + 0.775343 w_k1_AXP_s115
                                 + 0.664959 w_k1_DIS_s115
                                 + 0.813079 w_k1_FDX_s115
                                 + 0.755378 w_k1_IBM_s115
                                 + 0.923207 w_k1_INTC_s115
                                 + 0.772029 w_k1_KO_s115
                                 + 0.697135 w_k1_MAT_s115
                                 + 0.822282 w_k1_MSFT_s115
                                 + 0.802515 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.23892 w_k1_ADM_s115
                                 + 1.63352 w_k1_AXP_s115
                                 + 1.59123 w_k1_DIS_s115
                                 + 1.63945 w_k1_FDX_s115
                                 + 1.10044 w_k1_IBM_s115
                                 + 1.49652 w_k1_INTC_s115
                                 + 1.17903 w_k1_KO_s115
                                 + 0.948704 w_k1_MAT_s115
                                 + 1.2871 w_k1_MSFT_s115 + 1.13545 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.08843 w_k1_ADM_s115
                                 + 0.827045 w_k1_AXP_s115
                                 + 0.977223 w_k1_DIS_s115
                                 + 0.931553 w_k1_FDX_s115
                                 + 1.20808 w_k1_IBM_s115
                                 + 0.830064 w_k1_INTC_s115
                                 + 1.12671 w_k1_KO_s115
                                 + 0.779995 w_k1_MAT_s115
                                 + 1.06879 w_k1_MSFT_s115
                                 + 0.935059 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.797746 w_k1_ADM_s116
                                 + 0.958332 w_k1_AXP_s116
                                 + 0.936514 w_k1_DIS_s116
                                 + 0.944445 w_k1_FDX_s116
                                 + 1.22074 w_k1_IBM_s116
                                 + 1.10536 w_k1_INTC_s116
                                 + 1.03558 w_k1_KO_s116 + 1.4492 w_k1_MAT_s116
                                 + 1.08335 w_k1_MSFT_s116
                                 + 0.985778 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.29627 w_k1_ADM_s116
                                 + 1.52476 w_k1_AXP_s116
                                 + 1.37714 w_k1_DIS_s116
                                 + 1.46854 w_k1_FDX_s116
                                 + 0.873661 w_k1_IBM_s116
                                 + 1.26219 w_k1_INTC_s116
                                 + 1.04785 w_k1_KO_s116 + 1.50856 w_k1_MAT_s116
                                 + 1.42102 w_k1_MSFT_s116
                                 + 1.04446 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.01523 w_k1_ADM_s116
                                 + 1.12709 w_k1_AXP_s116
                                 + 0.739691 w_k1_DIS_s116
                                 + 0.84796 w_k1_FDX_s116
                                 + 0.800421 w_k1_IBM_s116
                                 + 0.969812 w_k1_INTC_s116
                                 + 0.852854 w_k1_KO_s116
                                 + 1.04062 w_k1_MAT_s116
                                 + 0.900544 w_k1_MSFT_s116
                                 + 0.912764 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.929649 w_k1_ADM_s116
                                 + 1.18305 w_k1_AXP_s116
                                 + 0.913316 w_k1_DIS_s116
                                 + 1.08907 w_k1_FDX_s116
                                 + 0.887833 w_k1_IBM_s116
                                 + 1.02245 w_k1_INTC_s116
                                 + 0.918663 w_k1_KO_s116
                                 + 1.33809 w_k1_MAT_s116 + 1.167 w_k1_MSFT_s116
                                 + 0.957834 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.594115 w_k1_ADM_s116
                                 + 0.71326 w_k1_AXP_s116
                                 + 0.643624 w_k1_DIS_s116
                                 + 0.682736 w_k1_FDX_s116
                                 + 0.855236 w_k1_IBM_s116
                                 + 0.823648 w_k1_INTC_s116
                                 + 0.792376 w_k1_KO_s116
                                 + 1.20312 w_k1_MAT_s116
                                 + 0.812386 w_k1_MSFT_s116
                                 + 0.739218 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.816882 w_k1_ADM_s116
                                 + 0.757587 w_k1_AXP_s116
                                 + 0.679013 w_k1_DIS_s116
                                 + 0.801515 w_k1_FDX_s116
                                 + 0.790274 w_k1_IBM_s116
                                 + 0.729224 w_k1_INTC_s116
                                 + 0.660283 w_k1_KO_s116
                                 + 0.858852 w_k1_MAT_s116
                                 + 0.90105 w_k1_MSFT_s116
                                 + 0.807973 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.09524 w_k1_ADM_s116
                                 + 1.35214 w_k1_AXP_s116
                                 + 1.49574 w_k1_DIS_s116
                                 + 1.20497 w_k1_FDX_s116
                                 + 1.58023 w_k1_IBM_s116
                                 + 1.17014 w_k1_INTC_s116
                                 + 1.65069 w_k1_KO_s116 + 2.12962 w_k1_MAT_s116
                                 + 1.3331 w_k1_MSFT_s116 + 1.34437 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.21867 w_k1_ADM_s116
                                 + 1.10607 w_k1_AXP_s116
                                 + 1.26749 w_k1_DIS_s116
                                 + 1.14701 w_k1_FDX_s116
                                 + 1.03254 w_k1_IBM_s116
                                 + 0.875604 w_k1_INTC_s116
                                 + 1.28154 w_k1_KO_s116 + 1.55264 w_k1_MAT_s116
                                 + 1.14531 w_k1_MSFT_s116
                                 + 1.07823 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.952017 w_k1_ADM_s117
                                 + 1.03455 w_k1_AXP_s117
                                 + 0.855979 w_k1_DIS_s117
                                 + 1.00286 w_k1_FDX_s117
                                 + 0.943904 w_k1_IBM_s117
                                 + 1.10944 w_k1_INTC_s117
                                 + 0.829264 w_k1_KO_s117
                                 + 1.31858 w_k1_MAT_s117
                                 + 0.961065 w_k1_MSFT_s117
                                 + 0.89656 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.842555 w_k1_ADM_s117
                                 + 1.02103 w_k1_AXP_s117
                                 + 0.789014 w_k1_DIS_s117
                                 + 0.863825 w_k1_FDX_s117
                                 + 1.32114 w_k1_IBM_s117
                                 + 0.910058 w_k1_INTC_s117
                                 + 1.06108 w_k1_KO_s117 + 1.40081 w_k1_MAT_s117
                                 + 0.832826 w_k1_MSFT_s117
                                 + 1.02975 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.899633 w_k1_ADM_s117
                                 + 0.766067 w_k1_AXP_s117
                                 + 0.662323 w_k1_DIS_s117
                                 + 0.826356 w_k1_FDX_s117
                                 + 0.843211 w_k1_IBM_s117
                                 + 0.861086 w_k1_INTC_s117
                                 + 0.80394 w_k1_KO_s117
                                 + 0.742497 w_k1_MAT_s117
                                 + 0.845259 w_k1_MSFT_s117
                                 + 0.85523 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.05214 w_k1_ADM_s117
                                 + 1.16702 w_k1_AXP_s117
                                 + 1.17823 w_k1_DIS_s117
                                 + 1.04496 w_k1_FDX_s117
                                 + 1.24892 w_k1_IBM_s117
                                 + 0.790433 w_k1_INTC_s117
                                 + 0.995121 w_k1_KO_s117
                                 + 1.41734 w_k1_MAT_s117
                                 + 0.837798 w_k1_MSFT_s117
                                 + 0.985815 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.994501 w_k1_ADM_s117
                                 + 1.46629 w_k1_AXP_s117
                                 + 1.44856 w_k1_DIS_s117
                                 + 1.35638 w_k1_FDX_s117 + 1.0727 w_k1_IBM_s117
                                 + 1.30546 w_k1_INTC_s117
                                 + 1.33814 w_k1_KO_s117 + 1.27785 w_k1_MAT_s117
                                 + 1.25218 w_k1_MSFT_s117
                                 + 1.18753 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.979872 w_k1_ADM_s117
                                 + 1.05662 w_k1_AXP_s117 + 1.0669 w_k1_DIS_s117
                                 + 1.09977 w_k1_FDX_s117
                                 + 0.954602 w_k1_IBM_s117
                                 + 1.14141 w_k1_INTC_s117
                                 + 1.05657 w_k1_KO_s117 + 1.0201 w_k1_MAT_s117
                                 + 1.20069 w_k1_MSFT_s117
                                 + 1.01871 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.968864 w_k1_ADM_s117
                                 + 1.56483 w_k1_AXP_s117
                                 + 1.41048 w_k1_DIS_s117
                                 + 1.20798 w_k1_FDX_s117
                                 + 1.56124 w_k1_IBM_s117
                                 + 1.21681 w_k1_INTC_s117 + 1.2027 w_k1_KO_s117
                                 + 1.62259 w_k1_MAT_s117
                                 + 1.13162 w_k1_MSFT_s117
                                 + 1.08034 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.04007 w_k1_ADM_s117
                                 + 1.02968 w_k1_AXP_s117
                                 + 1.00168 w_k1_DIS_s117
                                 + 1.01673 w_k1_FDX_s117
                                 + 0.634321 w_k1_IBM_s117
                                 + 0.95049 w_k1_INTC_s117
                                 + 0.884692 w_k1_KO_s117
                                 + 0.811967 w_k1_MAT_s117
                                 + 1.02134 w_k1_MSFT_s117
                                 + 0.864152 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0176987048228777 w_k1_ADM_s000
                                 + 0.0243737221353102 w_k1_ADM_s001
                                 + 0.0324589081448169 w_k1_ADM_s002
                                 + 0.0691916164502807 w_k1_ADM_s003
                                 + 0.0130507703188944 w_k1_ADM_s004
                                 + 0.0189798041421366 w_k1_ADM_s005
                                 + 0.0146525618785497 w_k1_ADM_s006
                                 + 0.060459952021324 w_k1_ADM_s007
                                 + 0.0390301358730572 w_k1_ADM_s010
                                 + 0.101423558462483 w_k1_ADM_s011
                                 + 0.0436524029751888 w_k1_ADM_s012
                                 + 0.0375427076744531 w_k1_ADM_s013
                                 + 0.0374674525723814 w_k1_ADM_s014
                                 + 0.0312036196037896 w_k1_ADM_s015
                                 + 0.0321376270191089 w_k1_ADM_s016
                                 + 0.0395715224724227 w_k1_ADM_s017
                                 + 0.0127223431382319 w_k1_ADM_s100
                                 + 0.0211599498636103 w_k1_ADM_s101
                                 + 0.0140051368220958 w_k1_ADM_s102
                                 + 0.0123462463955468 w_k1_ADM_s103
                                 + 0.0192987290769682 w_k1_ADM_s104
                                 + 0.0207732478941432 w_k1_ADM_s105
                                 + 0.0121206947074986 w_k1_ADM_s106
                                 + 0.0472710584565591 w_k1_ADM_s107
                                 + 0.041060569756665 w_k1_ADM_s110
                                 + 0.0140459708498146 w_k1_ADM_s111
                                 + 0.061948513000004 w_k1_ADM_s112
                                 + 0.0211605549327973 w_k1_ADM_s113
                                 + 0.0370277610764179 w_k1_ADM_s114
                                 + 0.020739734322071 w_k1_ADM_s115
                                 + 0.0218602819957802 w_k1_ADM_s116
                                 + 0.0134852036671076 w_k1_ADM_s117
                                 + 0.0182935872164462 w_k1_AXP_s000
                                 + 0.0246482353310842 w_k1_AXP_s001
                                 + 0.0388877844901843 w_k1_AXP_s002
                                 + 0.0678233496311947 w_k1_AXP_s003
                                 + 0.0151205095003912 w_k1_AXP_s004
                                 + 0.0175999595188828 w_k1_AXP_s005
                                 + 0.0163292128316839 w_k1_AXP_s006
                                 + 0.0627129684855391 w_k1_AXP_s007
                                 + 0.0409769553796744 w_k1_AXP_s010
                                 + 0.106421105166708 w_k1_AXP_s011
                                 + 0.0396742969561165 w_k1_AXP_s012
                                 + 0.0351967150179218 w_k1_AXP_s013
                                 + 0.0334548161397636 w_k1_AXP_s014
                                 + 0.0288013762934555 w_k1_AXP_s015
                                 + 0.0335546523960476 w_k1_AXP_s016
                                 + 0.0459341010364282 w_k1_AXP_s017
                                 + 0.0143563082069034 w_k1_AXP_s100
                                 + 0.0213551708055048 w_k1_AXP_s101
                                 + 0.0164845626282045 w_k1_AXP_s102
                                 + 0.0106702630510627 w_k1_AXP_s103
                                 + 0.0216922882227152 w_k1_AXP_s104
                                 + 0.0233190969438904 w_k1_AXP_s105
                                 + 0.0131969932128239 w_k1_AXP_s106
                                 + 0.0462623861373946 w_k1_AXP_s107
                                 + 0.039890123403865 w_k1_AXP_s110
                                 + 0.0165786518739125 w_k1_AXP_s111
                                 + 0.0569709754241435 w_k1_AXP_s112
                                 + 0.0204466657583599 w_k1_AXP_s113
                                 + 0.0365377685970036 w_k1_AXP_s114
                                 + 0.0212203041026722 w_k1_AXP_s115
                                 + 0.0251170520280348 w_k1_AXP_s116
                                 + 0.0150657313940612 w_k1_AXP_s117
                                 + 0.016076534889681 w_k1_DIS_s000
                                 + 0.02608449339862 w_k1_DIS_s001
                                 + 0.0375956112184082 w_k1_DIS_s002
                                 + 0.0644043159271566 w_k1_DIS_s003
                                 + 0.0148989677958219 w_k1_DIS_s004
                                 + 0.0172379006565256 w_k1_DIS_s005
                                 + 0.016752995664287 w_k1_DIS_s006
                                 + 0.0627320133912855 w_k1_DIS_s007
                                 + 0.0428171120389425 w_k1_DIS_s010
                                 + 0.0969591534742151 w_k1_DIS_s011
                                 + 0.0397639933186422 w_k1_DIS_s012
                                 + 0.0366459494903035 w_k1_DIS_s013
                                 + 0.0345473952176557 w_k1_DIS_s014
                                 + 0.0300935488827914 w_k1_DIS_s015
                                 + 0.0346582198436494 w_k1_DIS_s016
                                 + 0.0429986033434993 w_k1_DIS_s017
                                 + 0.0138366859412194 w_k1_DIS_s100
                                 + 0.023037044626499 w_k1_DIS_s101
                                 + 0.018650411255552 w_k1_DIS_s102
                                 + 0.0113342055635524 w_k1_DIS_s103
                                 + 0.0201384714683615 w_k1_DIS_s104
                                 + 0.0232860997757081 w_k1_DIS_s105
                                 + 0.0119398014439709 w_k1_DIS_s106
                                 + 0.0464475504677339 w_k1_DIS_s107
                                 + 0.044919452295248 w_k1_DIS_s110
                                 + 0.0143039655818294 w_k1_DIS_s111
                                 + 0.0632397648504958 w_k1_DIS_s112
                                 + 0.0215505542490728 w_k1_DIS_s113
                                 + 0.0430437728211554 w_k1_DIS_s114
                                 + 0.0215363788760343 w_k1_DIS_s115
                                 + 0.0231420158956831 w_k1_DIS_s116
                                 + 0.0138090060350254 w_k1_DIS_s117
                                 + 0.0171908206399694 w_k1_FDX_s000
                                 + 0.0243979943715295 w_k1_FDX_s001
                                 + 0.0388596754380637 w_k1_FDX_s002
                                 + 0.0667485645431877 w_k1_FDX_s003
                                 + 0.013989212124598 w_k1_FDX_s004
                                 + 0.0174330252016123 w_k1_FDX_s005
                                 + 0.0164903768373101 w_k1_FDX_s006
                                 + 0.0644188377207774 w_k1_FDX_s007
                                 + 0.0411770021170622 w_k1_FDX_s010
                                 + 0.100260952107715 w_k1_FDX_s011
                                 + 0.0385711357924838 w_k1_FDX_s012
                                 + 0.0362642721406947 w_k1_FDX_s013
                                 + 0.0349596750256892 w_k1_FDX_s014
                                 + 0.0293088074547394 w_k1_FDX_s015
                                 + 0.0320276759257733 w_k1_FDX_s016
                                 + 0.0408137301830028 w_k1_FDX_s017
                                 + 0.0152114078121004 w_k1_FDX_s100
                                 + 0.0219410259610125 w_k1_FDX_s101
                                 + 0.016922436453179 w_k1_FDX_s102
                                 + 0.0108935340517628 w_k1_FDX_s103
                                 + 0.0205975244010002 w_k1_FDX_s104
                                 + 0.0225598108998425 w_k1_FDX_s105
                                 + 0.0126525747886807 w_k1_FDX_s106
                                 + 0.044020007623477 w_k1_FDX_s107
                                 + 0.0445978423604995 w_k1_FDX_s110
                                 + 0.0136176384515979 w_k1_FDX_s111
                                 + 0.0601021961224815 w_k1_FDX_s112
                                 + 0.0231737985508959 w_k1_FDX_s113
                                 + 0.0390999276747174 w_k1_FDX_s114
                                 + 0.0218653995183839 w_k1_FDX_s115
                                 + 0.0238709775938033 w_k1_FDX_s116
                                 + 0.014338527337165 w_k1_FDX_s117
                                 + 0.0183476095335959 w_k1_IBM_s000
                                 + 0.0235482689662096 w_k1_IBM_s001
                                 + 0.0374958172932198 w_k1_IBM_s002
                                 + 0.064074828012833 w_k1_IBM_s003
                                 + 0.0144085347047333 w_k1_IBM_s004
                                 + 0.0177341213755531 w_k1_IBM_s005
                                 + 0.0163160614794503 w_k1_IBM_s006
                                 + 0.0619882825398038 w_k1_IBM_s007
                                 + 0.0416706229209389 w_k1_IBM_s010
                                 + 0.0992662354519903 w_k1_IBM_s011
                                 + 0.0403376064927365 w_k1_IBM_s012
                                 + 0.0379945998324954 w_k1_IBM_s013
                                 + 0.0356357890409107 w_k1_IBM_s014
                                 + 0.0308084563802905 w_k1_IBM_s015
                                 + 0.0337143935224977 w_k1_IBM_s016
                                 + 0.0419744284994586 w_k1_IBM_s017
                                 + 0.0154383536841749 w_k1_IBM_s100
                                 + 0.0223799009621775 w_k1_IBM_s101
                                 + 0.0169411523708538 w_k1_IBM_s102
                                 + 0.0108114433181468 w_k1_IBM_s103
                                 + 0.0206017932731269 w_k1_IBM_s104
                                 + 0.0220318785934955 w_k1_IBM_s105
                                 + 0.0113069851198349 w_k1_IBM_s106
                                 + 0.043886486921015 w_k1_IBM_s107
                                 + 0.042638727188404 w_k1_IBM_s110
                                 + 0.0140565062216302 w_k1_IBM_s111
                                 + 0.0636621556815081 w_k1_IBM_s112
                                 + 0.0231392588593657 w_k1_IBM_s113
                                 + 0.0406744541674071 w_k1_IBM_s114
                                 + 0.0219615774813273 w_k1_IBM_s115
                                 + 0.0246463596205489 w_k1_IBM_s116
                                 + 0.0144645757506503 w_k1_IBM_s117
                                 + 0.0181381856499896 w_k1_INTC_s000
                                 + 0.0260533727702054 w_k1_INTC_s001
                                 + 0.0394489814111457 w_k1_INTC_s002
                                 + 0.0731694538959189 w_k1_INTC_s003
                                 + 0.0137514683325801 w_k1_INTC_s004
                                 + 0.0164178136936317 w_k1_INTC_s005
                                 + 0.0173679449797917 w_k1_INTC_s006
                                 + 0.0697970720074205 w_k1_INTC_s007
                                 + 0.0413604017785224 w_k1_INTC_s010
                                 + 0.108641956044341 w_k1_INTC_s011
                                 + 0.0365177904195332 w_k1_INTC_s012
                                 + 0.0388581272682461 w_k1_INTC_s013
                                 + 0.0314699513316097 w_k1_INTC_s014
                                 + 0.027988624034727 w_k1_INTC_s015
                                 + 0.0336055545533915 w_k1_INTC_s016
                                 + 0.0430358828132817 w_k1_INTC_s017
                                 + 0.0142177679405333 w_k1_INTC_s100
                                 + 0.0195251248577973 w_k1_INTC_s101
                                 + 0.0181363284160685 w_k1_INTC_s102
                                 + 0.0100629039836552 w_k1_INTC_s103
                                 + 0.0181757752459826 w_k1_INTC_s104
                                 + 0.022618302906802 w_k1_INTC_s105
                                 + 0.0129205403182662 w_k1_INTC_s106
                                 + 0.0386606055995017 w_k1_INTC_s107
                                 + 0.0485548318011587 w_k1_INTC_s110
                                 + 0.0154336960115989 w_k1_INTC_s111
                                 + 0.0562186924697472 w_k1_INTC_s112
                                 + 0.0239652818986846 w_k1_INTC_s113
                                 + 0.039895009438211 w_k1_INTC_s114
                                 + 0.0222074642051029 w_k1_INTC_s115
                                 + 0.0240379813597427 w_k1_INTC_s116
                                 + 0.0144330431342612 w_k1_INTC_s117
                                 + 0.0169595351336036 w_k1_KO_s000
                                 + 0.0246091116307474 w_k1_KO_s001
                                 + 0.0370887557071456 w_k1_KO_s002
                                 + 0.0679924035345157 w_k1_KO_s003
                                 + 0.0144616161105757 w_k1_KO_s004
                                 + 0.0186470826737504 w_k1_KO_s005
                                 + 0.0154303318842653 w_k1_KO_s006
                                 + 0.0627710026257576 w_k1_KO_s007
                                 + 0.038910205837352 w_k1_KO_s010
                                 + 0.0947723266698099 w_k1_KO_s011
                                 + 0.0387668078326077 w_k1_KO_s012
                                 + 0.0410624927352727 w_k1_KO_s013
                                 + 0.0320451381205183 w_k1_KO_s014
                                 + 0.0324738541272525 w_k1_KO_s015
                                 + 0.0359644130310028 w_k1_KO_s016
                                 + 0.0413740485618992 w_k1_KO_s017
                                 + 0.0147983305826544 w_k1_KO_s100
                                 + 0.0224408338527185 w_k1_KO_s101
                                 + 0.0174794598006258 w_k1_KO_s102
                                 + 0.0095106182351477 w_k1_KO_s103
                                 + 0.0175752286423154 w_k1_KO_s104
                                 + 0.0210808291152779 w_k1_KO_s105
                                 + 0.0126250493353558 w_k1_KO_s106
                                 + 0.0420240607858993 w_k1_KO_s107
                                 + 0.0425338889128557 w_k1_KO_s110
                                 + 0.0134914079617317 w_k1_KO_s111
                                 + 0.0633151633143259 w_k1_KO_s112
                                 + 0.0212173207158807 w_k1_KO_s113
                                 + 0.040818602528334 w_k1_KO_s114
                                 + 0.0220635505563614 w_k1_KO_s115
                                 + 0.023997798230955 w_k1_KO_s116
                                 + 0.0138887314702412 w_k1_KO_s117
                                 + 0.0172188308632249 w_k1_MAT_s000
                                 + 0.0231244880412153 w_k1_MAT_s001
                                 + 0.0475704141605874 w_k1_MAT_s002
                                 + 0.069223818169998 w_k1_MAT_s003
                                 + 0.0161176103864873 w_k1_MAT_s004
                                 + 0.0248016326473869 w_k1_MAT_s005
                                 + 0.0156535873560231 w_k1_MAT_s006
                                 + 0.0573616157407465 w_k1_MAT_s007
                                 + 0.0444960943658432 w_k1_MAT_s010
                                 + 0.102384461503562 w_k1_MAT_s011
                                 + 0.0406635398316903 w_k1_MAT_s012
                                 + 0.0349153100950216 w_k1_MAT_s013
                                 + 0.0375908571381533 w_k1_MAT_s014
                                 + 0.035877996482001 w_k1_MAT_s015
                                 + 0.0347416768510077 w_k1_MAT_s016
                                 + 0.0406216658923475 w_k1_MAT_s017
                                 + 0.0200017296771356 w_k1_MAT_s100
                                 + 0.0239164359272676 w_k1_MAT_s101
                                 + 0.021178410946327 w_k1_MAT_s102
                                 + 0.0104082069129353 w_k1_MAT_s103
                                 + 0.0218746742565225 w_k1_MAT_s104
                                 + 0.0266167582609415 w_k1_MAT_s105
                                 + 0.0113470695759138 w_k1_MAT_s106
                                 + 0.0387239245043077 w_k1_MAT_s107
                                 + 0.0482297811566346 w_k1_MAT_s110
                                 + 0.0159654680804024 w_k1_MAT_s111
                                 + 0.0622832794498713 w_k1_MAT_s112
                                 + 0.0218467091879532 w_k1_MAT_s113
                                 + 0.0391969780449378 w_k1_MAT_s114
                                 + 0.0191608357509729 w_k1_MAT_s115
                                 + 0.032784949136025 w_k1_MAT_s116
                                 + 0.0159243068567919 w_k1_MAT_s117
                                 + 0.0158175395591179 w_k1_MSFT_s000
                                 + 0.0257203056688914 w_k1_MSFT_s001
                                 + 0.0399355939573952 w_k1_MSFT_s002
                                 + 0.0669091712788781 w_k1_MSFT_s003
                                 + 0.0142280955207791 w_k1_MSFT_s004
                                 + 0.017433213294089 w_k1_MSFT_s005
                                 + 0.0163360522228407 w_k1_MSFT_s006
                                 + 0.0622370069976499 w_k1_MSFT_s007
                                 + 0.0405630657828946 w_k1_MSFT_s010
                                 + 0.113787465740596 w_k1_MSFT_s011
                                 + 0.0397029651312251 w_k1_MSFT_s012
                                 + 0.0401606215546852 w_k1_MSFT_s013
                                 + 0.0347993119403575 w_k1_MSFT_s014
                                 + 0.0318163733095592 w_k1_MSFT_s015
                                 + 0.0327317104568786 w_k1_MSFT_s016
                                 + 0.0413767012096983 w_k1_MSFT_s017
                                 + 0.0149797110567099 w_k1_MSFT_s100
                                 + 0.0218866209587124 w_k1_MSFT_s101
                                 + 0.0165551970597428 w_k1_MSFT_s102
                                 + 0.0106955790308734 w_k1_MSFT_s103
                                 + 0.0180108201887204 w_k1_MSFT_s104
                                 + 0.0231379398291234 w_k1_MSFT_s105
                                 + 0.0131260467757312 w_k1_MSFT_s106
                                 + 0.0430850598771486 w_k1_MSFT_s107
                                 + 0.0423574657137901 w_k1_MSFT_s110
                                 + 0.0128130311441018 w_k1_MSFT_s111
                                 + 0.0636108291517085 w_k1_MSFT_s112
                                 + 0.0211559710837363 w_k1_MSFT_s113
                                 + 0.0388733466548072 w_k1_MSFT_s114
                                 + 0.0226216148376422 w_k1_MSFT_s115
                                 + 0.0259769267504933 w_k1_MSFT_s116
                                 + 0.0141349922958204 w_k1_MSFT_s117
                                 + 0.0159494141831549 w_k1_PG_s000
                                 + 0.024286900382446 w_k1_PG_s001
                                 + 0.0378969586692498 w_k1_PG_s002
                                 + 0.061038170951725 w_k1_PG_s003
                                 + 0.0136138345139765 w_k1_PG_s004
                                 + 0.0181163115577134 w_k1_PG_s005
                                 + 0.015907399190792 w_k1_PG_s006
                                 + 0.0599830658619707 w_k1_PG_s007
                                 + 0.0412385181903565 w_k1_PG_s010
                                 + 0.0975797559449527 w_k1_PG_s011
                                 + 0.0424102726097111 w_k1_PG_s012
                                 + 0.0373398226683415 w_k1_PG_s013
                                 + 0.0323857916236001 w_k1_PG_s014
                                 + 0.0318754738581856 w_k1_PG_s015
                                 + 0.0334779160963875 w_k1_PG_s016
                                 + 0.0409133528127619 w_k1_PG_s017
                                 + 0.0146815108195418 w_k1_PG_s100
                                 + 0.0222362314499253 w_k1_PG_s101
                                 + 0.0165812963512871 w_k1_PG_s102
                                 + 0.00956004157762761 w_k1_PG_s103
                                 + 0.0195737391915498 w_k1_PG_s104
                                 + 0.0220134504223914 w_k1_PG_s105
                                 + 0.0137241435536113 w_k1_PG_s106
                                 + 0.0416162033261448 w_k1_PG_s107
                                 + 0.0433858817140917 w_k1_PG_s110
                                 + 0.013894374385959 w_k1_PG_s111
                                 + 0.0616223962590962 w_k1_PG_s112
                                 + 0.0206645790985544 w_k1_PG_s113
                                 + 0.0416576717907912 w_k1_PG_s114
                                 + 0.0210494937509201 w_k1_PG_s115
                                 + 0.0231808451563417 w_k1_PG_s116
                                 + 0.0136718704652795 w_k1_PG_s117 >= 1.11
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
