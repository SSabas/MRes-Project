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
 w_k1_ADM_s0_return_constraint:  0.943664 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.14966 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.859714 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.28367 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09502 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.924865 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.937504 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.916137 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.78433 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.33282 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.09717 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.938804 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.908043 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.63889 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.37625 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.936406 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.766616 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.97082 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.01704 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.801432 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.19381 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.12945 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.39606 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.955757 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.0836 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.724908 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.883492 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.759927 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.06064 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.23936 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.80698 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.669109 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.944307 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.936279 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.952045 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.989556 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.49834 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.20249 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.92378 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.22698 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.02104 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.1541 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.13622 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.958931 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.64655 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.719467 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.77747 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.04045 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01436 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.700561 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.802699 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.42212 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.61086 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.2207 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.646513 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.15413 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.992518 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.766415 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.57503 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.994305 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.67269 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.98101 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.30825 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.899503 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.625918 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.609168 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.945807 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.09918 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.764464 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.662959 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.808334 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.32282 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.935316 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.21593 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.60703 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.12357 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.940831 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.29197 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.978843 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.23852 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.22831 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.91396 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.601108 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.657363 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.75933 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.11836 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.05164 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.722056 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.712823 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.75443 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.61992 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.14142 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.577906 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.01331 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.02728 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.772986 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.93335 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.42766 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.67469 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.964688 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.01289 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.576896 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.576229 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.601275 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.776344 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.48439 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.884783 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.559869 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.847097 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.1004 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.8618 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.32141 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.89246 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.08618 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.966327 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.15808 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.940325 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.24375 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.19686 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.868994 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.769031 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.700927 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.52816 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.15732 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.930612 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.857878 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.774478 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.44853 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.41694 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.06888 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.69654 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.17021 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.980461 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.864629 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.39532 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12702 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.38297 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.947359 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.12288 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.825923 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.731833 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.670063 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.885468 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.17039 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.865958 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.703991 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.936884 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.0531 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.883786 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.22885 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.50309 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.15929 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.89782 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.32858 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.08972 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.833265 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.887791 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.09882 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.583455 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.29829 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.66193 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.91622 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.28598 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.779268 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.835571 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.2687 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.08801 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.31746 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.724551 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.853094 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.03954 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.787161 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.77359 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.4826 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.53622 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.20365 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.876243 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.646355 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.548394 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.815722 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.862825 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.923149 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.973646 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.675712 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.624644 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.79049 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.27624 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.1768 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.63525 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.823685 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.919719 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.09205 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.00391 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.20568 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.14898 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.983313 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.806257 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.768098 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.35885 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.03098 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.961219 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.891892 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.875622 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.3407 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.36973 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05384 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.748754 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.11171 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.888432 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.80513 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.34197 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.15511 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.34153 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07601 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.926242 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.848365 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.796539 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.815332 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.836669 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.05269 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.829355 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.652899 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.858871 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.20903 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.88968 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.01408 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.37085 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.04789 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.94291 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.21742 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.05346 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.00683 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.04898 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.994518 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.640761 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.962239 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.43267 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.98564 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.07136 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.749355 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.776682 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.26638 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.38168 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.20873 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.686927 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.903574 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.00667 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.887866 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.63044 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.33108 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.55397 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.06268 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.890867 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.677364 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.705358 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.829731 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.882864 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.21375 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.946291 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.61938 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.75006 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.25133 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.959853 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.28785 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.73725 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.9185 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.929406 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.42349 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.12003 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.873031 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.855395 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.114 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.499813 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.19082 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 2.11242 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.960595 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.988138 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.678685 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.774429 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.27678 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.20035 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.73926 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.556798 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.946095 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.96993 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.86464 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.01402 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.17061 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.6097 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.30913 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.28279 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.760494 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.47756 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.609163 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.00971 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.17829 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.766132 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.511733 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.566296 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.79944 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.1252 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.10166 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.85058 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.698282 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.969281 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.11051 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.946033 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.13148 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.22772 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03507 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.799422 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.749656 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.37584 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.08083 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.06379 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.851684 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.81839 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.30584 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.53204 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.1187 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.750433 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.08572 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.965783 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.930866 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.44256 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.14982 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.28071 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.901928 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.04541 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.795704 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.806386 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.695965 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.877037 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.23954 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.789539 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.6264 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.824988 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.98325 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.849336 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.03202 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.34842 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.877991 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.942693 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.11603 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.01795 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.05357 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10463 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.986328 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.77196 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.886714 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.38788 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06621 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.057 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.849371 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.838054 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.28902 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.28381 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.21319 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.854623 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.0123 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02058 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.938613 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.40629 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.12576 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.28926 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.992203 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.985921 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.820222 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.799102 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.786344 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.933647 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.11726 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.949733 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.786527 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.901557 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.06778 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.97757 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.1967 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.35872 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.01603 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 1.30500720572339 z_k1_s0000
                                 - 0.784002859439835 z_k1_s0001
                                 - 0.434001110559821 z_k1_s0002
                                 - 1.61600818819152 z_k1_s0003
                                 - 1.0640029359796 z_k1_s0004
                                 - 0.51300200599206 z_k1_s0005
                                 - 1.40000038269883 z_k1_s0006
                                 - 1.15000267608162 z_k1_s0007
                                 - 0.249000097262187 z_k1_s0010
                                 - 0.259000118855056 z_k1_s0011
                                 - 0.284000172837228 z_k1_s0012
                                 - 0.704001813738129 z_k1_s0013
                                 - 1.03299914774162 z_k1_s0014
                                 - 0.284000172837228 z_k1_s0015
                                 - 0.421998856017915 z_k1_s0016
                                 - 0.435001673290204 z_k1_s0017
                                 - 0.278999999407482 z_k1_s0020
                                 - 0.444000030885027 z_k1_s0021
                                 - 0.600998441805973 z_k1_s0022
                                 - 0.316000001981234 z_k1_s0023
                                 - 0.506999935267356 z_k1_s0024
                                 - 0.420001829215692 z_k1_s0025
                                 - 0.706999749179516 z_k1_s0026
                                 - 0.425999529632897 z_k1_s0027
                                 - 2.38100758042558 z_k1_s0030
                                 - 1.01000263486375 z_k1_s0031
                                 - 0.976001926112308 z_k1_s0032
                                 - 3.35800607903577 z_k1_s0033
                                 - 2.94500756251537 z_k1_s0034
                                 - 1.12500166786959 z_k1_s0035
                                 - 1.10400215527781 z_k1_s0036
                                 - 1.39900334248238 z_k1_s0037
                                 - 2.38200253029234 z_k1_s0040
                                 - 0.549999672460196 z_k1_s0041
                                 - 1.21100448569107 z_k1_s0042
                                 - 1.04799962050604 z_k1_s0043
                                 - 0.614000368201222 z_k1_s0044
                                 - 1.02400344385429 z_k1_s0045
                                 - 1.60399593462851 z_k1_s0046
                                 - 1.1770042962662 z_k1_s0047
                                 - 0.598999004643544 z_k1_s0050
                                 - 0.436001170819889 z_k1_s0051
                                 - 0.377000046773629 z_k1_s0052
                                 - 0.799000409680594 z_k1_s0053
                                 - 0.570001639154492 z_k1_s0054
                                 - 0.164999976555016 z_k1_s0055
                                 - 0.39300023801663 z_k1_s0056
                                 - 0.603000136504799 z_k1_s0057
                                 - 0.469998675413284 z_k1_s0060
                                 - 0.70099901375455 z_k1_s0061
                                 - 0.367000029258715 z_k1_s0062
                                 - 0.486998297715519 z_k1_s0063
                                 - 0.241999911595682 z_k1_s0064
                                 - 0.325000073423922 z_k1_s0065
                                 - 0.884999484554487 z_k1_s0066
                                 - 0.398001186838968 z_k1_s0067
                                 - 0.71200278036947 z_k1_s0070
                                 - 0.603000921851412 z_k1_s0071
                                 - 0.65000034680795 z_k1_s0072
                                 - 0.592998916114758 z_k1_s0073
                                 - 0.346000384648655 z_k1_s0074
                                 - 1.02300014574213 z_k1_s0075
                                 - 0.273999946152775 z_k1_s0076
                                 - 0.800000426840554 z_k1_s0077
                                 - 0.270000139242566 z_k1_s0100
                                 - 0.109999775662241 z_k1_s0101
                                 - 0.145999794227916 z_k1_s0102
                                 - 0.149999085358586 z_k1_s0103
                                 - 0.315000162449661 z_k1_s0104
                                 - 0.137999723968984 z_k1_s0105
                                 - 0.294000052419844 z_k1_s0106
                                 - 0.0649999012549058 z_k1_s0107
                                 - 0.0559999253051367 z_k1_s0110
                                 - 0.164999606345818 z_k1_s0111
                                 - 0.103999916823721 z_k1_s0112
                                 - 0.101999544577813 z_k1_s0113
                                 - 0.072999881801378 z_k1_s0114
                                 - 0.116999420427598 z_k1_s0115
                                 - 0.169999564962413 z_k1_s0116
                                 - 0.184999440812199 z_k1_s0117
                                 - 0.346999280243584 z_k1_s0120
                                 - 0.743997376290256 z_k1_s0121
                                 - 0.449998918005002 z_k1_s0122
                                 - 0.691997223050341 z_k1_s0123
                                 - 0.312999378363167 z_k1_s0124
                                 - 0.486998247946796 z_k1_s0125
                                 - 0.480998655225794 z_k1_s0126
                                 - 0.168999577281917 z_k1_s0127
                                 - 0.191000199147865 z_k1_s0130
                                 - 0.177000270046041 z_k1_s0131
                                 - 0.217001233881381 z_k1_s0132
                                 - 0.480001457210571 z_k1_s0133
                                 - 0.211000097864755 z_k1_s0134
                                 - 0.310999591449209 z_k1_s0135
                                 - 0.218000451216186 z_k1_s0136
                                 - 0.139000268083689 z_k1_s0137
                                 - 0.144999904798928 z_k1_s0140
                                 - 0.366998960767604 z_k1_s0141
                                 - 0.240000481045973 z_k1_s0142
                                 - 0.424998922687176 z_k1_s0143
                                 - 0.424998922687176 z_k1_s0144
                                 - 0.129999874733555 z_k1_s0145
                                 - 0.2029998667185 z_k1_s0146
                                 - 0.379999989990096 z_k1_s0147
                                 - 0.453001586364231 z_k1_s0150
                                 - 0.183000172423836 z_k1_s0151
                                 - 0.193000054680843 z_k1_s0152
                                 - 0.400999626547114 z_k1_s0153
                                 - 0.4659997798179 z_k1_s0154
                                 - 0.570998849716549 z_k1_s0155
                                 - 0.250000118425975 z_k1_s0156
                                 - 0.544999400808398 z_k1_s0157
                                 - 0.148999916002449 z_k1_s0160
                                 - 0.191999299816285 z_k1_s0161
                                 - 0.0659998450616959 z_k1_s0162
                                 - 0.105999552190349 z_k1_s0163
                                 - 0.170999897022956 z_k1_s0164
                                 - 0.085999845625762 z_k1_s0165
                                 - 0.134999660808054 z_k1_s0166
                                 - 0.0749998551155083 z_k1_s0167
                                 - 0.0739999111229238 z_k1_s0170
                                 - 0.0739999111229238 z_k1_s0171
                                 - 0.110999866684386 z_k1_s0172
                                 - 0.31899968005682 z_k1_s0173
                                 - 0.138000213390748 z_k1_s0174
                                 - 0.164999391099115 z_k1_s0175
                                 - 0.0989998684590684 z_k1_s0176
                                 - 0.189000088948547 z_k1_s0177
                                 - 0.593999917095136 z_k1_s1000
                                 - 0.320999655965236 z_k1_s1001
                                 - 0.129000172221872 z_k1_s1002
                                 - 0.197000231839571 z_k1_s1003
                                 - 0.347001091502911 z_k1_s1004
                                 - 0.120000179890135 z_k1_s1005
                                 - 0.225000866294563 z_k1_s1006
                                 - 0.183000126212299 z_k1_s1007
                                 - 0.205999981956141 z_k1_s1010
                                 - 0.404999691709881 z_k1_s1011
                                 - 0.488999024138579 z_k1_s1012
                                 - 0.414000121970113 z_k1_s1013
                                 - 0.254000122343984 z_k1_s1014
                                 - 0.317999279194411 z_k1_s1015
                                 - 0.19099992052244 z_k1_s1016
                                 - 0.533000815410817 z_k1_s1017
                                 - 0.183000087879173 z_k1_s1020
                                 - 0.0739998661808548 z_k1_s1021
                                 - 0.347999507607097 z_k1_s1022
                                 - 0.09499990069155 z_k1_s1023
                                 - 0.366999964544828 z_k1_s1024
                                 - 0.138999845285542 z_k1_s1025
                                 - 0.0789998531119985 z_k1_s1026
                                 - 0.20499991117635 z_k1_s1027
                                 - 0.0839997620146506 z_k1_s1030
                                 - 0.0850000148960027 z_k1_s1031
                                 - 0.132999981190179 z_k1_s1032
                                 - 0.069000049997965 z_k1_s1033
                                 - 0.0770001398470785 z_k1_s1034
                                 - 0.0570000226243894 z_k1_s1035
                                 - 0.0880000575394281 z_k1_s1036
                                 - 0.123000316379181 z_k1_s1037
                                 - 0.177000046292818 z_k1_s1040
                                 - 0.0899996838775387 z_k1_s1041
                                 - 0.122999734966051 z_k1_s1042
                                 - 0.0450000089388508 z_k1_s1043
                                 - 0.104000135703178 z_k1_s1044
                                 - 0.0640000257019274 z_k1_s1045
                                 - 0.13700018679169 z_k1_s1046
                                 - 0.0949996663151797 z_k1_s1047
                                 - 0.131999934119527 z_k1_s1050
                                 - 0.0649999983588736 z_k1_s1051
                                 - 0.0929997995841353 z_k1_s1052
                                 - 0.0649999983588736 z_k1_s1053
                                 - 0.0869996485786251 z_k1_s1054
                                 - 0.082000059174302 z_k1_s1055
                                 - 0.265000175740115 z_k1_s1056
                                 - 0.0580000480525582 z_k1_s1057
                                 - 0.312999658102971 z_k1_s1060
                                 - 0.282001393724602 z_k1_s1061
                                 - 0.172000167242397 z_k1_s1062
                                 - 0.275000553701916 z_k1_s1063
                                 - 0.162000201926042 z_k1_s1064
                                 - 0.306999799513331 z_k1_s1065
                                 - 0.104000161890839 z_k1_s1066
                                 - 0.394999735497599 z_k1_s1067
                                 - 0.109000004127648 z_k1_s1070
                                 - 0.106000004014043 z_k1_s1071
                                 - 0.140000005301566 z_k1_s1072
                                 - 0.0830001032431754 z_k1_s1073
                                 - 0.13200000499862 z_k1_s1074
                                 - 0.0460000017419433 z_k1_s1075
                                 - 0.195000007384325 z_k1_s1076
                                 - 0.190000007194983 z_k1_s1077
                                 - 0.414998971191586 z_k1_s1100
                                 - 0.227000169935924 z_k1_s1101
                                 - 0.903000842862078 z_k1_s1102
                                 - 0.358001942742661 z_k1_s1103
                                 - 0.618001800598323 z_k1_s1104
                                 - 0.304000368416586 z_k1_s1105
                                 - 0.22300043943082 z_k1_s1106
                                 - 0.427001637711681 z_k1_s1107
                                 - 0.606997940383199 z_k1_s1110
                                 - 0.334000656973055 z_k1_s1111
                                 - 0.290999770348175 z_k1_s1112
                                 - 0.290999770348175 z_k1_s1113
                                 - 0.672998845766137 z_k1_s1114
                                 - 0.203999754332846 z_k1_s1115
                                 - 0.250999614112575 z_k1_s1116
                                 - 0.507998161806838 z_k1_s1117
                                 - 0.129999697338952 z_k1_s1120
                                 - 0.631998377069655 z_k1_s1121
                                 - 0.217999959684427 z_k1_s1122
                                 - 0.230999354859419 z_k1_s1123
                                 - 0.239000087580381 z_k1_s1124
                                 - 0.279998969284619 z_k1_s1125
                                 - 0.188999714666334 z_k1_s1126
                                 - 0.132999715609802 z_k1_s1127
                                 - 0.0619999603191217 z_k1_s1130
                                 - 0.0639999141359166 z_k1_s1131
                                 - 0.295999196879057 z_k1_s1132
                                 - 0.185999996957238 z_k1_s1133
                                 - 0.249999447093662 z_k1_s1134
                                 - 0.0989998599290039 z_k1_s1135
                                 - 0.0649999490442507 z_k1_s1136
                                 - 0.138000409354921 z_k1_s1137
                                 - 0.34499909083984 z_k1_s1140
                                 - 0.450999297671437 z_k1_s1141
                                 - 0.335998965391953 z_k1_s1142
                                 - 0.151999958543668 z_k1_s1143
                                 - 0.121999921583766 z_k1_s1144
                                 - 0.13999998187968 z_k1_s1145
                                 - 0.138999904407738 z_k1_s1146
                                 - 0.221000539111081 z_k1_s1147
                                 - 0.242999443969496 z_k1_s1150
                                 - 0.621998549847071 z_k1_s1151
                                 - 0.483998814266953 z_k1_s1152
                                 - 0.381998622186681 z_k1_s1153
                                 - 0.531998205600079 z_k1_s1154
                                 - 0.311999014660079 z_k1_s1155
                                 - 0.167999652262797 z_k1_s1156
                                 - 0.227999485628156 z_k1_s1157
                                 - 0.283000830077625 z_k1_s1160
                                 - 0.266999164191171 z_k1_s1161
                                 - 0.169999959483114 z_k1_s1162
                                 - 0.34400065508758 z_k1_s1163
                                 - 0.319999221757025 z_k1_s1164
                                 - 0.215000675804519 z_k1_s1165
                                 - 0.378000732224132 z_k1_s1166
                                 - 0.153999785295437 z_k1_s1167
                                 - 0.575001372860721 z_k1_s1170
                                 - 0.718003046869121 z_k1_s1171
                                 - 0.329000884440534 z_k1_s1172
                                 - 0.178000244596473 z_k1_s1173
                                 - 0.40999969869777 z_k1_s1174
                                 - 0.343999698234115 z_k1_s1175
                                 - 0.501001068158354 z_k1_s1176
                                 - 0.291000610409321 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.932704 w_k1_ADM_s000
                                 + 1.25304 w_k1_AXP_s000
                                 + 1.02786 w_k1_DIS_s000
                                 + 1.01894 w_k1_FDX_s000
                                 + 1.41269 w_k1_IBM_s000
                                 + 0.907472 w_k1_INTC_s000
                                 + 1.06364 w_k1_KO_s000 + 1.56702 w_k1_MAT_s000
                                 + 1.03297 w_k1_MSFT_s000
                                 + 1.06636 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.897018 w_k1_ADM_s000
                                 + 0.829337 w_k1_AXP_s000
                                 + 0.825357 w_k1_DIS_s000
                                 + 0.959968 w_k1_FDX_s000
                                 + 0.734718 w_k1_IBM_s000
                                 + 0.820749 w_k1_INTC_s000
                                 + 0.733388 w_k1_KO_s000
                                 + 0.578057 w_k1_MAT_s000
                                 + 0.901566 w_k1_MSFT_s000
                                 + 0.878337 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.931308 w_k1_ADM_s000
                                 + 0.819501 w_k1_AXP_s000
                                 + 1.08745 w_k1_DIS_s000
                                 + 1.03319 w_k1_FDX_s000
                                 + 1.10694 w_k1_IBM_s000
                                 + 1.12871 w_k1_INTC_s000
                                 + 1.07247 w_k1_KO_s000 + 1.07543 w_k1_MAT_s000
                                 + 1.09706 w_k1_MSFT_s000
                                 + 1.02572 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.40949 w_k1_ADM_s000
                                 + 1.3645 w_k1_AXP_s000 + 1.38363 w_k1_DIS_s000
                                 + 1.40045 w_k1_FDX_s000 + 1.1596 w_k1_IBM_s000
                                 + 1.17572 w_k1_INTC_s000 + 1.2635 w_k1_KO_s000
                                 + 1.35012 w_k1_MAT_s000
                                 + 1.28314 w_k1_MSFT_s000
                                 + 1.22216 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.982811 w_k1_ADM_s000
                                 + 1.05816 w_k1_AXP_s000
                                 + 0.957234 w_k1_DIS_s000
                                 + 1.16157 w_k1_FDX_s000
                                 + 0.683939 w_k1_IBM_s000
                                 + 1.03423 w_k1_INTC_s000
                                 + 0.810789 w_k1_KO_s000
                                 + 0.755812 w_k1_MAT_s000
                                 + 1.10191 w_k1_MSFT_s000
                                 + 1.00528 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.688867 w_k1_ADM_s000
                                 + 0.494048 w_k1_AXP_s000
                                 + 0.469668 w_k1_DIS_s000
                                 + 0.67473 w_k1_FDX_s000
                                 + 0.670152 w_k1_IBM_s000
                                 + 0.65434 w_k1_INTC_s000
                                 + 0.642511 w_k1_KO_s000
                                 + 0.683351 w_k1_MAT_s000
                                 + 0.669948 w_k1_MSFT_s000
                                 + 0.704045 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.654587 w_k1_ADM_s000
                                 + 0.792703 w_k1_AXP_s000
                                 + 0.567611 w_k1_DIS_s000
                                 + 0.712544 w_k1_FDX_s000
                                 + 0.955934 w_k1_IBM_s000
                                 + 0.839998 w_k1_INTC_s000
                                 + 0.721985 w_k1_KO_s000
                                 + 0.795727 w_k1_MAT_s000
                                 + 0.767575 w_k1_MSFT_s000
                                 + 0.866914 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.985391 w_k1_ADM_s000
                                 + 0.79072 w_k1_AXP_s000
                                 + 0.860379 w_k1_DIS_s000
                                 + 0.839411 w_k1_FDX_s000
                                 + 1.08555 w_k1_IBM_s000
                                 + 0.9134 w_k1_INTC_s000 + 1.01984 w_k1_KO_s000
                                 + 1.45092 w_k1_MAT_s000
                                 + 0.906272 w_k1_MSFT_s000
                                 + 0.901885 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.767073 w_k1_ADM_s001
                                 + 1.01204 w_k1_AXP_s001
                                 + 0.60895 w_k1_DIS_s001
                                 + 0.907696 w_k1_FDX_s001
                                 + 0.586033 w_k1_IBM_s001
                                 + 0.944172 w_k1_INTC_s001
                                 + 0.60784 w_k1_KO_s001
                                 + 0.757909 w_k1_MAT_s001
                                 + 0.900893 w_k1_MSFT_s001
                                 + 0.812325 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.12464 w_k1_ADM_s001
                                 + 1.15192 w_k1_AXP_s001
                                 + 1.02573 w_k1_DIS_s001
                                 + 1.16296 w_k1_FDX_s001
                                 + 0.638513 w_k1_IBM_s001
                                 + 1.11973 w_k1_INTC_s001
                                 + 0.738333 w_k1_KO_s001
                                 + 0.646671 w_k1_MAT_s001
                                 + 1.16122 w_k1_MSFT_s001
                                 + 0.845102 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.719319 w_k1_ADM_s001
                                 + 0.764166 w_k1_AXP_s001
                                 + 0.572251 w_k1_DIS_s001
                                 + 0.788381 w_k1_FDX_s001
                                 + 0.744468 w_k1_IBM_s001
                                 + 0.840883 w_k1_INTC_s001
                                 + 0.642837 w_k1_KO_s001
                                 + 0.779525 w_k1_MAT_s001
                                 + 0.796264 w_k1_MSFT_s001
                                 + 0.805259 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.929926 w_k1_ADM_s001
                                 + 0.850967 w_k1_AXP_s001
                                 + 0.910756 w_k1_DIS_s001
                                 + 0.968172 w_k1_FDX_s001
                                 + 1.00564 w_k1_IBM_s001
                                 + 0.963134 w_k1_INTC_s001
                                 + 0.952977 w_k1_KO_s001
                                 + 0.99093 w_k1_MAT_s001
                                 + 0.948805 w_k1_MSFT_s001
                                 + 1.05187 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.28366 w_k1_ADM_s001
                                 + 1.57138 w_k1_AXP_s001
                                 + 1.50274 w_k1_DIS_s001
                                 + 1.27608 w_k1_FDX_s001
                                 + 1.37964 w_k1_IBM_s001
                                 + 1.2203 w_k1_INTC_s001 + 1.05683 w_k1_KO_s001
                                 + 1.22439 w_k1_MAT_s001
                                 + 1.25128 w_k1_MSFT_s001
                                 + 1.12202 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.14564 w_k1_ADM_s001
                                 + 1.21063 w_k1_AXP_s001
                                 + 1.24533 w_k1_DIS_s001
                                 + 1.00812 w_k1_FDX_s001
                                 + 1.76097 w_k1_IBM_s001
                                 + 1.07691 w_k1_INTC_s001
                                 + 1.24413 w_k1_KO_s001 + 1.12058 w_k1_MAT_s001
                                 + 0.906729 w_k1_MSFT_s001
                                 + 1.09233 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.3131 w_k1_ADM_s001
                                 + 1.31899 w_k1_AXP_s001
                                 + 0.855604 w_k1_DIS_s001
                                 + 1.10693 w_k1_FDX_s001
                                 + 0.964033 w_k1_IBM_s001
                                 + 1.15166 w_k1_INTC_s001
                                 + 0.905074 w_k1_KO_s001
                                 + 0.906045 w_k1_MAT_s001
                                 + 1.12627 w_k1_MSFT_s001
                                 + 1.08187 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.25342 w_k1_ADM_s001
                                 + 1.33643 w_k1_AXP_s001
                                 + 1.03702 w_k1_DIS_s001 + 1.1691 w_k1_FDX_s001
                                 + 0.835266 w_k1_IBM_s001
                                 + 0.950387 w_k1_INTC_s001
                                 + 0.896979 w_k1_KO_s001
                                 + 0.95517 w_k1_MAT_s001
                                 + 0.982738 w_k1_MSFT_s001
                                 + 1.00554 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.976561 w_k1_ADM_s002
                                 + 1.51381 w_k1_AXP_s002
                                 + 1.82969 w_k1_DIS_s002
                                 + 1.49667 w_k1_FDX_s002
                                 + 1.39546 w_k1_IBM_s002
                                 + 1.57174 w_k1_INTC_s002
                                 + 1.68265 w_k1_KO_s002 + 1.41413 w_k1_MAT_s002
                                 + 1.56614 w_k1_MSFT_s002
                                 + 1.28301 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.708899 w_k1_ADM_s002
                                 + 0.822408 w_k1_AXP_s002
                                 + 0.630514 w_k1_DIS_s002
                                 + 0.765418 w_k1_FDX_s002
                                 + 0.959642 w_k1_IBM_s002
                                 + 0.752343 w_k1_INTC_s002
                                 + 0.831886 w_k1_KO_s002
                                 + 0.936386 w_k1_MAT_s002
                                 + 0.716312 w_k1_MSFT_s002
                                 + 0.78521 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.919566 w_k1_ADM_s002
                                 + 1.1191 w_k1_AXP_s002
                                 + 0.927469 w_k1_DIS_s002
                                 + 1.01259 w_k1_FDX_s002
                                 + 0.863831 w_k1_IBM_s002
                                 + 1.02391 w_k1_INTC_s002
                                 + 0.918736 w_k1_KO_s002
                                 + 0.915859 w_k1_MAT_s002
                                 + 0.966243 w_k1_MSFT_s002
                                 + 0.987033 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.937608 w_k1_ADM_s002
                                 + 0.929306 w_k1_AXP_s002
                                 + 1.03684 w_k1_DIS_s002
                                 + 1.02166 w_k1_FDX_s002
                                 + 0.675042 w_k1_IBM_s002
                                 + 0.954172 w_k1_INTC_s002
                                 + 0.927098 w_k1_KO_s002
                                 + 0.638754 w_k1_MAT_s002
                                 + 1.18657 w_k1_MSFT_s002
                                 + 0.971185 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.704279 w_k1_ADM_s002
                                 + 1.08196 w_k1_AXP_s002
                                 + 0.912844 w_k1_DIS_s002
                                 + 0.822213 w_k1_FDX_s002
                                 + 1.56696 w_k1_IBM_s002
                                 + 1.08988 w_k1_INTC_s002
                                 + 1.29159 w_k1_KO_s002 + 1.21052 w_k1_MAT_s002
                                 + 0.925627 w_k1_MSFT_s002
                                 + 0.916434 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.717334 w_k1_ADM_s002
                                 + 0.861145 w_k1_AXP_s002
                                 + 0.699326 w_k1_DIS_s002
                                 + 0.890086 w_k1_FDX_s002
                                 + 0.542126 w_k1_IBM_s002
                                 + 1.07651 w_k1_INTC_s002
                                 + 0.732787 w_k1_KO_s002
                                 + 0.505227 w_k1_MAT_s002
                                 + 0.935087 w_k1_MSFT_s002
                                 + 0.800385 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.939511 w_k1_ADM_s002
                                 + 1.30781 w_k1_AXP_s002
                                 + 1.12123 w_k1_DIS_s002
                                 + 1.08072 w_k1_FDX_s002
                                 + 1.22893 w_k1_IBM_s002
                                 + 1.29057 w_k1_INTC_s002
                                 + 1.15725 w_k1_KO_s002 + 1.05577 w_k1_MAT_s002
                                 + 1.06948 w_k1_MSFT_s002
                                 + 1.10395 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.28974 w_k1_ADM_s002
                                 + 1.31552 w_k1_AXP_s002
                                 + 1.44208 w_k1_DIS_s002
                                 + 1.39625 w_k1_FDX_s002
                                 + 0.821844 w_k1_IBM_s002
                                 + 1.16338 w_k1_INTC_s002
                                 + 1.28985 w_k1_KO_s002
                                 + 0.671825 w_k1_MAT_s002
                                 + 1.25491 w_k1_MSFT_s002
                                 + 1.18395 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.1321 w_k1_ADM_s003
                                 + 1.18147 w_k1_AXP_s003
                                 + 1.24461 w_k1_DIS_s003
                                 + 1.10786 w_k1_FDX_s003
                                 + 1.32156 w_k1_IBM_s003
                                 + 1.26391 w_k1_INTC_s003
                                 + 1.15895 w_k1_KO_s003 + 1.51441 w_k1_MAT_s003
                                 + 1.21602 w_k1_MSFT_s003 + 1.0822 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.1473 w_k1_ADM_s003
                                 + 1.02696 w_k1_AXP_s003
                                 + 0.887985 w_k1_DIS_s003
                                 + 1.13664 w_k1_FDX_s003
                                 + 0.71652 w_k1_IBM_s003
                                 + 1.05537 w_k1_INTC_s003
                                 + 0.81947 w_k1_KO_s003
                                 + 0.670106 w_k1_MAT_s003
                                 + 1.14937 w_k1_MSFT_s003
                                 + 0.92881 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.15124 w_k1_ADM_s003
                                 + 1.78071 w_k1_AXP_s003
                                 + 1.67555 w_k1_DIS_s003
                                 + 1.30832 w_k1_FDX_s003
                                 + 1.78438 w_k1_IBM_s003
                                 + 1.50947 w_k1_INTC_s003
                                 + 1.46856 w_k1_KO_s003 + 1.86362 w_k1_MAT_s003
                                 + 1.35549 w_k1_MSFT_s003
                                 + 1.29829 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.779443 w_k1_ADM_s003
                                 + 0.800145 w_k1_AXP_s003
                                 + 0.679385 w_k1_DIS_s003
                                 + 0.750791 w_k1_FDX_s003
                                 + 0.808814 w_k1_IBM_s003
                                 + 0.902459 w_k1_INTC_s003
                                 + 0.808482 w_k1_KO_s003
                                 + 0.617821 w_k1_MAT_s003
                                 + 0.876669 w_k1_MSFT_s003
                                 + 0.83032 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.856445 w_k1_ADM_s003
                                 + 0.962942 w_k1_AXP_s003
                                 + 1.03965 w_k1_DIS_s003
                                 + 1.01532 w_k1_FDX_s003
                                 + 0.955229 w_k1_IBM_s003
                                 + 1.1993 w_k1_INTC_s003
                                 + 0.996083 w_k1_KO_s003
                                 + 0.964157 w_k1_MAT_s003
                                 + 1.06355 w_k1_MSFT_s003
                                 + 0.936952 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.898188 w_k1_ADM_s003
                                 + 1.10989 w_k1_AXP_s003
                                 + 0.904721 w_k1_DIS_s003
                                 + 0.913689 w_k1_FDX_s003
                                 + 1.5103 w_k1_IBM_s003
                                 + 1.05312 w_k1_INTC_s003 + 1.0286 w_k1_KO_s003
                                 + 1.15018 w_k1_MAT_s003
                                 + 0.963075 w_k1_MSFT_s003
                                 + 1.00555 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.23859 w_k1_ADM_s003
                                 + 1.49242 w_k1_AXP_s003
                                 + 1.49161 w_k1_DIS_s003
                                 + 1.31589 w_k1_FDX_s003
                                 + 0.795255 w_k1_IBM_s003
                                 + 1.31947 w_k1_INTC_s003
                                 + 1.09475 w_k1_KO_s003
                                 + 0.952581 w_k1_MAT_s003
                                 + 1.35985 w_k1_MSFT_s003 + 1.1815 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.9324 w_k1_ADM_s003
                                 + 1.3591 w_k1_AXP_s003
                                 + 0.953977 w_k1_DIS_s003
                                 + 1.05502 w_k1_FDX_s003
                                 + 0.832081 w_k1_IBM_s003
                                 + 1.01176 w_k1_INTC_s003
                                 + 0.847269 w_k1_KO_s003
                                 + 0.930679 w_k1_MAT_s003
                                 + 1.00951 w_k1_MSFT_s003
                                 + 0.954918 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.18754 w_k1_ADM_s004
                                 + 1.20093 w_k1_AXP_s004
                                 + 1.18622 w_k1_DIS_s004
                                 + 1.15477 w_k1_FDX_s004
                                 + 1.48866 w_k1_IBM_s004
                                 + 1.24613 w_k1_INTC_s004
                                 + 1.35492 w_k1_KO_s004 + 1.66723 w_k1_MAT_s004
                                 + 1.15922 w_k1_MSFT_s004
                                 + 1.10573 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.07707 w_k1_ADM_s004
                                 + 0.988855 w_k1_AXP_s004
                                 + 0.881243 w_k1_DIS_s004
                                 + 1.07713 w_k1_FDX_s004
                                 + 0.679759 w_k1_IBM_s004
                                 + 0.927413 w_k1_INTC_s004
                                 + 0.805735 w_k1_KO_s004
                                 + 1.26517 w_k1_MAT_s004
                                 + 0.841117 w_k1_MSFT_s004
                                 + 0.920033 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.792654 w_k1_ADM_s004
                                 + 0.804259 w_k1_AXP_s004
                                 + 0.706354 w_k1_DIS_s004
                                 + 0.856562 w_k1_FDX_s004
                                 + 0.922805 w_k1_IBM_s004
                                 + 0.945685 w_k1_INTC_s004
                                 + 0.878351 w_k1_KO_s004
                                 + 1.10137 w_k1_MAT_s004
                                 + 0.770111 w_k1_MSFT_s004
                                 + 0.847174 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.744576 w_k1_ADM_s004
                                 + 0.665378 w_k1_AXP_s004
                                 + 0.576532 w_k1_DIS_s004
                                 + 0.762482 w_k1_FDX_s004
                                 + 0.697401 w_k1_IBM_s004
                                 + 0.86443 w_k1_INTC_s004
                                 + 0.675787 w_k1_KO_s004
                                 + 0.672027 w_k1_MAT_s004
                                 + 0.769278 w_k1_MSFT_s004
                                 + 0.759574 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.55867 w_k1_ADM_s004
                                 + 1.30974 w_k1_AXP_s004
                                 + 1.36015 w_k1_DIS_s004
                                 + 1.37486 w_k1_FDX_s004
                                 + 0.905768 w_k1_IBM_s004
                                 + 1.02722 w_k1_INTC_s004
                                 + 1.03388 w_k1_KO_s004 + 1.08708 w_k1_MAT_s004
                                 + 1.04839 w_k1_MSFT_s004
                                 + 1.12285 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.09891 w_k1_ADM_s004
                                 + 0.932266 w_k1_AXP_s004
                                 + 0.901809 w_k1_DIS_s004
                                 + 1.07529 w_k1_FDX_s004
                                 + 0.88813 w_k1_IBM_s004
                                 + 1.05841 w_k1_INTC_s004
                                 + 0.997765 w_k1_KO_s004
                                 + 1.13053 w_k1_MAT_s004
                                 + 0.937641 w_k1_MSFT_s004
                                 + 0.97887 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.970445 w_k1_ADM_s004
                                 + 0.95238 w_k1_AXP_s004
                                 + 1.00686 w_k1_DIS_s004 + 1.0391 w_k1_FDX_s004
                                 + 1.0703 w_k1_IBM_s004 + 1.1233 w_k1_INTC_s004
                                 + 1.00267 w_k1_KO_s004 + 1.32689 w_k1_MAT_s004
                                 + 0.966361 w_k1_MSFT_s004
                                 + 0.987848 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.07039 w_k1_ADM_s004
                                 + 0.694896 w_k1_AXP_s004
                                 + 0.886941 w_k1_DIS_s004
                                 + 0.95397 w_k1_FDX_s004
                                 + 0.698419 w_k1_IBM_s004
                                 + 0.966687 w_k1_INTC_s004
                                 + 0.827674 w_k1_KO_s004
                                 + 0.691569 w_k1_MAT_s004
                                 + 0.895386 w_k1_MSFT_s004
                                 + 0.943223 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.835302 w_k1_ADM_s005
                                 + 0.656184 w_k1_AXP_s005
                                 + 0.647387 w_k1_DIS_s005
                                 + 0.682549 w_k1_FDX_s005
                                 + 0.995164 w_k1_IBM_s005
                                 + 0.813378 w_k1_INTC_s005
                                 + 0.873212 w_k1_KO_s005
                                 + 0.747137 w_k1_MAT_s005
                                 + 0.824911 w_k1_MSFT_s005
                                 + 0.885426 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.20567 w_k1_ADM_s005
                                 + 1.16435 w_k1_AXP_s005
                                 + 1.11711 w_k1_DIS_s005
                                 + 1.12111 w_k1_FDX_s005
                                 + 0.966684 w_k1_IBM_s005
                                 + 0.954543 w_k1_INTC_s005
                                 + 1.01445 w_k1_KO_s005 + 1.15839 w_k1_MAT_s005
                                 + 1.07882 w_k1_MSFT_s005
                                 + 1.04026 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.41678 w_k1_ADM_s005
                                 + 1.40667 w_k1_AXP_s005
                                 + 1.35026 w_k1_DIS_s005 + 1.3554 w_k1_FDX_s005
                                 + 0.913816 w_k1_IBM_s005
                                 + 1.27896 w_k1_INTC_s005 + 1.215 w_k1_KO_s005
                                 + 0.976394 w_k1_MAT_s005
                                 + 1.20774 w_k1_MSFT_s005
                                 + 1.16133 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.10549 w_k1_ADM_s005
                                 + 0.882064 w_k1_AXP_s005
                                 + 0.954029 w_k1_DIS_s005
                                 + 0.936113 w_k1_FDX_s005
                                 + 0.964872 w_k1_IBM_s005
                                 + 1.12235 w_k1_INTC_s005
                                 + 1.00485 w_k1_KO_s005
                                 + 0.779049 w_k1_MAT_s005
                                 + 1.00575 w_k1_MSFT_s005
                                 + 1.08207 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.912762 w_k1_ADM_s005
                                 + 0.812866 w_k1_AXP_s005
                                 + 0.913207 w_k1_DIS_s005
                                 + 1.00972 w_k1_FDX_s005
                                 + 0.678199 w_k1_IBM_s005
                                 + 0.922851 w_k1_INTC_s005
                                 + 0.812621 w_k1_KO_s005
                                 + 0.629332 w_k1_MAT_s005
                                 + 1.07982 w_k1_MSFT_s005
                                 + 1.01266 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.873581 w_k1_ADM_s005
                                 + 1.00942 w_k1_AXP_s005
                                 + 0.601077 w_k1_DIS_s005
                                 + 0.835442 w_k1_FDX_s005
                                 + 0.97645 w_k1_IBM_s005
                                 + 0.781931 w_k1_INTC_s005
                                 + 0.744244 w_k1_KO_s005
                                 + 1.11542 w_k1_MAT_s005
                                 + 0.829663 w_k1_MSFT_s005
                                 + 0.944934 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.20118 w_k1_ADM_s005
                                 + 1.27635 w_k1_AXP_s005
                                 + 1.57446 w_k1_DIS_s005 + 1.2479 w_k1_FDX_s005
                                 + 1.46883 w_k1_IBM_s005
                                 + 1.12221 w_k1_INTC_s005
                                 + 1.42413 w_k1_KO_s005 + 1.66974 w_k1_MAT_s005
                                 + 1.21485 w_k1_MSFT_s005
                                 + 1.38162 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.32192 w_k1_ADM_s005
                                 + 1.05616 w_k1_AXP_s005
                                 + 0.971308 w_k1_DIS_s005
                                 + 0.970089 w_k1_FDX_s005
                                 + 1.36771 w_k1_IBM_s005
                                 + 0.91775 w_k1_INTC_s005
                                 + 1.16155 w_k1_KO_s005 + 1.04301 w_k1_MAT_s005
                                 + 1.03059 w_k1_MSFT_s005
                                 + 1.09939 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.80031 w_k1_ADM_s006
                                 + 0.957745 w_k1_AXP_s006
                                 + 0.768313 w_k1_DIS_s006
                                 + 0.879972 w_k1_FDX_s006
                                 + 1.23909 w_k1_IBM_s006
                                 + 0.930508 w_k1_INTC_s006
                                 + 0.917057 w_k1_KO_s006
                                 + 1.48127 w_k1_MAT_s006
                                 + 0.867028 w_k1_MSFT_s006
                                 + 1.05123 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.672173 w_k1_ADM_s006
                                 + 0.650422 w_k1_AXP_s006
                                 + 0.578995 w_k1_DIS_s006
                                 + 0.748314 w_k1_FDX_s006
                                 + 0.994863 w_k1_IBM_s006
                                 + 0.909597 w_k1_INTC_s006
                                 + 0.752595 w_k1_KO_s006
                                 + 0.868658 w_k1_MAT_s006
                                 + 0.748782 w_k1_MSFT_s006
                                 + 0.828609 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.810307 w_k1_ADM_s006
                                 + 0.717908 w_k1_AXP_s006
                                 + 1.09302 w_k1_DIS_s006
                                 + 1.00437 w_k1_FDX_s006
                                 + 0.959794 w_k1_IBM_s006
                                 + 1.06108 w_k1_INTC_s006
                                 + 0.98449 w_k1_KO_s006
                                 + 0.947618 w_k1_MAT_s006
                                 + 1.14236 w_k1_MSFT_s006
                                 + 1.01172 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.22668 w_k1_ADM_s006
                                 + 1.1889 w_k1_AXP_s006 + 1.05057 w_k1_DIS_s006
                                 + 1.10043 w_k1_FDX_s006 + 1.3581 w_k1_IBM_s006
                                 + 0.946894 w_k1_INTC_s006
                                 + 1.12923 w_k1_KO_s006 + 1.2182 w_k1_MAT_s006
                                 + 1.02078 w_k1_MSFT_s006
                                 + 1.12919 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.2027 w_k1_ADM_s006
                                 + 1.24582 w_k1_AXP_s006
                                 + 1.28288 w_k1_DIS_s006
                                 + 1.34247 w_k1_FDX_s006
                                 + 0.733913 w_k1_IBM_s006
                                 + 1.18789 w_k1_INTC_s006
                                 + 0.952519 w_k1_KO_s006
                                 + 0.914208 w_k1_MAT_s006
                                 + 1.20664 w_k1_MSFT_s006
                                 + 0.913643 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.26508 w_k1_ADM_s006
                                 + 1.2538 w_k1_AXP_s006 + 1.433 w_k1_DIS_s006
                                 + 1.36902 w_k1_FDX_s006
                                 + 1.07615 w_k1_IBM_s006
                                 + 1.14637 w_k1_INTC_s006
                                 + 1.18765 w_k1_KO_s006 + 1.05701 w_k1_MAT_s006
                                 + 1.28543 w_k1_MSFT_s006
                                 + 1.19012 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.17904 w_k1_ADM_s006
                                 + 0.989651 w_k1_AXP_s006
                                 + 0.758499 w_k1_DIS_s006
                                 + 1.03865 w_k1_FDX_s006
                                 + 0.840653 w_k1_IBM_s006
                                 + 0.863032 w_k1_INTC_s006
                                 + 0.863874 w_k1_KO_s006
                                 + 1.06071 w_k1_MAT_s006
                                 + 0.880699 w_k1_MSFT_s006
                                 + 0.94801 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.78814 w_k1_ADM_s006
                                 + 1.18134 w_k1_AXP_s006
                                 + 1.26566 w_k1_DIS_s006
                                 + 1.17698 w_k1_FDX_s006
                                 + 1.35695 w_k1_IBM_s006
                                 + 1.21924 w_k1_INTC_s006
                                 + 1.34423 w_k1_KO_s006 + 1.91291 w_k1_MAT_s006
                                 + 1.17051 w_k1_MSFT_s006
                                 + 1.14069 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.829479 w_k1_ADM_s007
                                 + 0.930942 w_k1_AXP_s007
                                 + 0.991524 w_k1_DIS_s007
                                 + 0.866969 w_k1_FDX_s007
                                 + 1.39869 w_k1_IBM_s007
                                 + 1.04902 w_k1_INTC_s007
                                 + 1.36157 w_k1_KO_s007 + 1.02697 w_k1_MAT_s007
                                 + 0.999159 w_k1_MSFT_s007
                                 + 0.980178 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.913242 w_k1_ADM_s007
                                 + 0.931785 w_k1_AXP_s007
                                 + 1.17436 w_k1_DIS_s007
                                 + 1.17319 w_k1_FDX_s007
                                 + 0.706881 w_k1_IBM_s007
                                 + 1.19885 w_k1_INTC_s007
                                 + 0.93967 w_k1_KO_s007
                                 + 0.559429 w_k1_MAT_s007
                                 + 1.24554 w_k1_MSFT_s007
                                 + 0.980233 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.896425 w_k1_ADM_s007
                                 + 1.12079 w_k1_AXP_s007 + 1.0018 w_k1_DIS_s007
                                 + 1.02458 w_k1_FDX_s007
                                 + 1.14405 w_k1_IBM_s007
                                 + 1.01977 w_k1_INTC_s007 + 1.1898 w_k1_KO_s007
                                 + 1.24941 w_k1_MAT_s007
                                 + 1.14956 w_k1_MSFT_s007
                                 + 1.03942 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.872461 w_k1_ADM_s007
                                 + 0.755672 w_k1_AXP_s007
                                 + 0.863739 w_k1_DIS_s007
                                 + 0.861396 w_k1_FDX_s007
                                 + 0.675782 w_k1_IBM_s007
                                 + 1.00175 w_k1_INTC_s007
                                 + 0.919412 w_k1_KO_s007
                                 + 0.678971 w_k1_MAT_s007
                                 + 1.0747 w_k1_MSFT_s007
                                 + 0.804418 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.0231 w_k1_ADM_s007
                                 + 0.882616 w_k1_AXP_s007
                                 + 0.787862 w_k1_DIS_s007
                                 + 0.996659 w_k1_FDX_s007
                                 + 0.578696 w_k1_IBM_s007
                                 + 0.9938 w_k1_INTC_s007
                                 + 0.689961 w_k1_KO_s007
                                 + 0.482342 w_k1_MAT_s007
                                 + 0.922883 w_k1_MSFT_s007
                                 + 0.812436 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.794057 w_k1_ADM_s007
                                 + 0.871401 w_k1_AXP_s007
                                 + 0.722752 w_k1_DIS_s007
                                 + 0.803945 w_k1_FDX_s007
                                 + 0.910294 w_k1_IBM_s007
                                 + 0.974883 w_k1_INTC_s007
                                 + 0.911231 w_k1_KO_s007
                                 + 0.833299 w_k1_MAT_s007
                                 + 0.944415 w_k1_MSFT_s007
                                 + 0.851928 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.0599 w_k1_ADM_s007
                                 + 1.20391 w_k1_AXP_s007
                                 + 1.46355 w_k1_DIS_s007
                                 + 1.11739 w_k1_FDX_s007
                                 + 2.01833 w_k1_IBM_s007
                                 + 1.07716 w_k1_INTC_s007
                                 + 1.70366 w_k1_KO_s007 + 1.25118 w_k1_MAT_s007
                                 + 1.15716 w_k1_MSFT_s007
                                 + 1.12849 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.39052 w_k1_ADM_s007
                                 + 1.26476 w_k1_AXP_s007
                                 + 1.58949 w_k1_DIS_s007
                                 + 1.28492 w_k1_FDX_s007
                                 + 1.17942 w_k1_IBM_s007
                                 + 1.29635 w_k1_INTC_s007
                                 + 1.51636 w_k1_KO_s007 + 1.19864 w_k1_MAT_s007
                                 + 1.46707 w_k1_MSFT_s007
                                 + 1.21088 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.968716 w_k1_ADM_s010
                                 + 1.03599 w_k1_AXP_s010
                                 + 1.11205 w_k1_DIS_s010
                                 + 0.999305 w_k1_FDX_s010
                                 + 1.39163 w_k1_IBM_s010
                                 + 0.825504 w_k1_INTC_s010
                                 + 1.04992 w_k1_KO_s010 + 1.27759 w_k1_MAT_s010
                                 + 0.855214 w_k1_MSFT_s010
                                 + 1.07728 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.39046 w_k1_ADM_s010
                                 + 1.59755 w_k1_AXP_s010
                                 + 1.51424 w_k1_DIS_s010
                                 + 1.48538 w_k1_FDX_s010 + 1.1605 w_k1_IBM_s010
                                 + 1.05277 w_k1_INTC_s010
                                 + 1.20107 w_k1_KO_s010 + 1.84865 w_k1_MAT_s010
                                 + 1.17845 w_k1_MSFT_s010
                                 + 1.19204 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.795518 w_k1_ADM_s010
                                 + 0.98311 w_k1_AXP_s010
                                 + 1.32114 w_k1_DIS_s010
                                 + 1.14439 w_k1_FDX_s010
                                 + 0.859604 w_k1_IBM_s010
                                 + 1.20809 w_k1_INTC_s010
                                 + 1.02105 w_k1_KO_s010 + 1.06068 w_k1_MAT_s010
                                 + 1.11727 w_k1_MSFT_s010
                                 + 0.975619 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.77936 w_k1_ADM_s010
                                 + 0.618331 w_k1_AXP_s010
                                 + 0.827764 w_k1_DIS_s010
                                 + 0.852417 w_k1_FDX_s010
                                 + 0.751473 w_k1_IBM_s010
                                 + 0.697115 w_k1_INTC_s010
                                 + 0.724382 w_k1_KO_s010
                                 + 0.659339 w_k1_MAT_s010
                                 + 0.753489 w_k1_MSFT_s010
                                 + 0.893373 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.773737 w_k1_ADM_s010
                                 + 0.862758 w_k1_AXP_s010
                                 + 0.745882 w_k1_DIS_s010
                                 + 0.823557 w_k1_FDX_s010
                                 + 0.928622 w_k1_IBM_s010
                                 + 0.69177 w_k1_INTC_s010
                                 + 0.696171 w_k1_KO_s010
                                 + 1.03442 w_k1_MAT_s010
                                 + 0.682143 w_k1_MSFT_s010
                                 + 0.800668 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.18476 w_k1_ADM_s010
                                 + 1.24906 w_k1_AXP_s010
                                 + 1.28831 w_k1_DIS_s010
                                 + 1.31801 w_k1_FDX_s010
                                 + 1.25021 w_k1_IBM_s010
                                 + 1.17946 w_k1_INTC_s010
                                 + 1.12666 w_k1_KO_s010 + 1.5746 w_k1_MAT_s010
                                 + 1.08642 w_k1_MSFT_s010
                                 + 1.14082 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.03829 w_k1_ADM_s010
                                 + 0.82857 w_k1_AXP_s010
                                 + 0.902379 w_k1_DIS_s010
                                 + 1.03379 w_k1_FDX_s010
                                 + 0.804706 w_k1_IBM_s010
                                 + 0.73718 w_k1_INTC_s010
                                 + 0.681936 w_k1_KO_s010
                                 + 0.754356 w_k1_MAT_s010
                                 + 0.840923 w_k1_MSFT_s010
                                 + 0.934113 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.05571 w_k1_ADM_s010
                                 + 1.57385 w_k1_AXP_s010
                                 + 0.979263 w_k1_DIS_s010
                                 + 1.14259 w_k1_FDX_s010
                                 + 1.24434 w_k1_IBM_s010
                                 + 1.08757 w_k1_INTC_s010
                                 + 0.944392 w_k1_KO_s010
                                 + 1.92536 w_k1_MAT_s010
                                 + 0.848043 w_k1_MSFT_s010 + 1.031 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.05205 w_k1_ADM_s011
                                 + 0.651676 w_k1_AXP_s011
                                 + 0.587209 w_k1_DIS_s011
                                 + 0.672613 w_k1_FDX_s011
                                 + 0.973243 w_k1_IBM_s011
                                 + 0.723303 w_k1_INTC_s011
                                 + 0.808844 w_k1_KO_s011
                                 + 0.52927 w_k1_MAT_s011
                                 + 0.611057 w_k1_MSFT_s011
                                 + 0.754405 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.02836 w_k1_ADM_s011
                                 + 0.889574 w_k1_AXP_s011
                                 + 0.813091 w_k1_DIS_s011
                                 + 0.815912 w_k1_FDX_s011
                                 + 1.01227 w_k1_IBM_s011
                                 + 0.905191 w_k1_INTC_s011
                                 + 0.824169 w_k1_KO_s011
                                 + 0.666375 w_k1_MAT_s011
                                 + 0.859867 w_k1_MSFT_s011
                                 + 0.829259 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.07911 w_k1_ADM_s011
                                 + 1.04522 w_k1_AXP_s011
                                 + 0.912942 w_k1_DIS_s011
                                 + 0.88036 w_k1_FDX_s011
                                 + 1.51985 w_k1_IBM_s011
                                 + 0.947327 w_k1_INTC_s011
                                 + 1.17116 w_k1_KO_s011 + 1.04001 w_k1_MAT_s011
                                 + 0.886884 w_k1_MSFT_s011
                                 + 1.05036 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.33477 w_k1_ADM_s011
                                 + 1.31006 w_k1_AXP_s011 + 1.2534 w_k1_DIS_s011
                                 + 1.0973 w_k1_FDX_s011 + 1.42403 w_k1_IBM_s011
                                 + 1.21712 w_k1_INTC_s011
                                 + 1.37989 w_k1_KO_s011 + 1.39428 w_k1_MAT_s011
                                 + 1.2531 w_k1_MSFT_s011 + 1.07653 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.667 w_k1_ADM_s011
                                 + 1.36854 w_k1_AXP_s011
                                 + 1.14839 w_k1_DIS_s011
                                 + 1.20466 w_k1_FDX_s011
                                 + 1.07872 w_k1_IBM_s011
                                 + 0.870818 w_k1_INTC_s011
                                 + 0.977632 w_k1_KO_s011 + 1.117 w_k1_MAT_s011
                                 + 1.05346 w_k1_MSFT_s011
                                 + 1.05036 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.37077 w_k1_ADM_s011
                                 + 1.37572 w_k1_AXP_s011 + 1.4777 w_k1_DIS_s011
                                 + 1.35639 w_k1_FDX_s011
                                 + 1.01872 w_k1_IBM_s011
                                 + 1.01762 w_k1_INTC_s011
                                 + 1.12908 w_k1_KO_s011 + 1.00174 w_k1_MAT_s011
                                 + 1.16537 w_k1_MSFT_s011 + 1.1738 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.40148 w_k1_ADM_s011
                                 + 0.977644 w_k1_AXP_s011
                                 + 1.14343 w_k1_DIS_s011
                                 + 1.03542 w_k1_FDX_s011
                                 + 0.846813 w_k1_IBM_s011
                                 + 0.943213 w_k1_INTC_s011
                                 + 0.973452 w_k1_KO_s011
                                 + 0.54034 w_k1_MAT_s011
                                 + 1.01578 w_k1_MSFT_s011
                                 + 0.978588 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.07011 w_k1_ADM_s011
                                 + 0.880695 w_k1_AXP_s011
                                 + 0.832785 w_k1_DIS_s011
                                 + 0.899057 w_k1_FDX_s011
                                 + 0.804822 w_k1_IBM_s011
                                 + 0.73297 w_k1_INTC_s011
                                 + 0.775662 w_k1_KO_s011
                                 + 0.63909 w_k1_MAT_s011
                                 + 0.795975 w_k1_MSFT_s011
                                 + 0.909804 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.31202 w_k1_ADM_s012
                                 + 1.37757 w_k1_AXP_s012
                                 + 1.83111 w_k1_DIS_s012
                                 + 1.46463 w_k1_FDX_s012
                                 + 1.36823 w_k1_IBM_s012
                                 + 1.45159 w_k1_INTC_s012
                                 + 1.58105 w_k1_KO_s012 + 1.31076 w_k1_MAT_s012
                                 + 1.54027 w_k1_MSFT_s012
                                 + 1.20763 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.814983 w_k1_ADM_s012
                                 + 0.794798 w_k1_AXP_s012
                                 + 0.917793 w_k1_DIS_s012
                                 + 0.935348 w_k1_FDX_s012
                                 + 1.03135 w_k1_IBM_s012
                                 + 0.985762 w_k1_INTC_s012
                                 + 1.08199 w_k1_KO_s012 + 1.16835 w_k1_MAT_s012
                                 + 1.01927 w_k1_MSFT_s012
                                 + 0.903226 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.991074 w_k1_ADM_s012
                                 + 1.25507 w_k1_AXP_s012
                                 + 0.864281 w_k1_DIS_s012
                                 + 1.09529 w_k1_FDX_s012
                                 + 0.740877 w_k1_IBM_s012
                                 + 1.06952 w_k1_INTC_s012
                                 + 0.897192 w_k1_KO_s012
                                 + 1.19309 w_k1_MAT_s012
                                 + 1.02659 w_k1_MSFT_s012
                                 + 0.974427 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.637088 w_k1_ADM_s012
                                 + 0.682759 w_k1_AXP_s012
                                 + 0.558823 w_k1_DIS_s012
                                 + 0.737734 w_k1_FDX_s012
                                 + 0.635095 w_k1_IBM_s012
                                 + 0.845628 w_k1_INTC_s012
                                 + 0.653321 w_k1_KO_s012
                                 + 0.76218 w_k1_MAT_s012
                                 + 0.847169 w_k1_MSFT_s012
                                 + 0.740534 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.12605 w_k1_ADM_s012
                                 + 1.2922 w_k1_AXP_s012 + 1.18144 w_k1_DIS_s012
                                 + 1.13804 w_k1_FDX_s012
                                 + 1.20767 w_k1_IBM_s012
                                 + 0.927912 w_k1_INTC_s012
                                 + 1.18359 w_k1_KO_s012 + 1.70946 w_k1_MAT_s012
                                 + 1.02213 w_k1_MSFT_s012
                                 + 1.02507 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.951092 w_k1_ADM_s012
                                 + 1.20037 w_k1_AXP_s012
                                 + 1.19974 w_k1_DIS_s012
                                 + 1.02855 w_k1_FDX_s012
                                 + 1.80082 w_k1_IBM_s012
                                 + 1.25689 w_k1_INTC_s012 + 1.3471 w_k1_KO_s012
                                 + 1.61489 w_k1_MAT_s012
                                 + 1.15168 w_k1_MSFT_s012
                                 + 0.980894 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.996763 w_k1_ADM_s012
                                 + 1.02681 w_k1_AXP_s012
                                 + 1.19742 w_k1_DIS_s012
                                 + 1.12202 w_k1_FDX_s012
                                 + 0.771868 w_k1_IBM_s012
                                 + 1.08872 w_k1_INTC_s012
                                 + 1.00778 w_k1_KO_s012 + 1.2793 w_k1_MAT_s012
                                 + 1.17349 w_k1_MSFT_s012
                                 + 1.06241 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.638311 w_k1_ADM_s012
                                 + 0.748054 w_k1_AXP_s012
                                 + 0.813533 w_k1_DIS_s012
                                 + 0.692678 w_k1_FDX_s012
                                 + 1.28999 w_k1_IBM_s012
                                 + 0.833189 w_k1_INTC_s012
                                 + 0.949108 w_k1_KO_s012
                                 + 1.24438 w_k1_MAT_s012
                                 + 0.809835 w_k1_MSFT_s012
                                 + 0.839928 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.837882 w_k1_ADM_s013
                                 + 0.920619 w_k1_AXP_s013
                                 + 0.907977 w_k1_DIS_s013
                                 + 0.841665 w_k1_FDX_s013
                                 + 1.15274 w_k1_IBM_s013
                                 + 0.968294 w_k1_INTC_s013
                                 + 1.1314 w_k1_KO_s013 + 0.839444 w_k1_MAT_s013
                                 + 0.869932 w_k1_MSFT_s013
                                 + 0.961661 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.972216 w_k1_ADM_s013
                                 + 0.874725 w_k1_AXP_s013
                                 + 0.765268 w_k1_DIS_s013
                                 + 0.958953 w_k1_FDX_s013
                                 + 0.635112 w_k1_IBM_s013
                                 + 0.969414 w_k1_INTC_s013
                                 + 0.714135 w_k1_KO_s013
                                 + 0.602032 w_k1_MAT_s013
                                 + 0.928749 w_k1_MSFT_s013
                                 + 0.816659 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.933906 w_k1_ADM_s013
                                 + 0.837261 w_k1_AXP_s013
                                 + 1.03676 w_k1_DIS_s013
                                 + 0.964506 w_k1_FDX_s013
                                 + 0.857453 w_k1_IBM_s013
                                 + 0.977463 w_k1_INTC_s013
                                 + 0.984398 w_k1_KO_s013
                                 + 0.660603 w_k1_MAT_s013
                                 + 1.01858 w_k1_MSFT_s013
                                 + 0.952088 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.31093 w_k1_ADM_s013
                                 + 1.15266 w_k1_AXP_s013
                                 + 1.25478 w_k1_DIS_s013
                                 + 1.22186 w_k1_FDX_s013
                                 + 0.986203 w_k1_IBM_s013
                                 + 1.11709 w_k1_INTC_s013
                                 + 1.02599 w_k1_KO_s013 + 1.27749 w_k1_MAT_s013
                                 + 1.17997 w_k1_MSFT_s013
                                 + 1.26166 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.29978 w_k1_ADM_s013
                                 + 1.37233 w_k1_AXP_s013
                                 + 1.63895 w_k1_DIS_s013
                                 + 1.23581 w_k1_FDX_s013
                                 + 1.49137 w_k1_IBM_s013
                                 + 1.22161 w_k1_INTC_s013
                                 + 1.39565 w_k1_KO_s013 + 1.1922 w_k1_MAT_s013
                                 + 1.28368 w_k1_MSFT_s013
                                 + 1.23648 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.15908 w_k1_ADM_s013
                                 + 1.04935 w_k1_AXP_s013
                                 + 0.95871 w_k1_DIS_s013
                                 + 0.956343 w_k1_FDX_s013
                                 + 1.18133 w_k1_IBM_s013
                                 + 0.952863 w_k1_INTC_s013
                                 + 1.21017 w_k1_KO_s013 + 1.23317 w_k1_MAT_s013
                                 + 1.04031 w_k1_MSFT_s013
                                 + 1.05602 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.978504 w_k1_ADM_s013
                                 + 0.933391 w_k1_AXP_s013
                                 + 0.760392 w_k1_DIS_s013
                                 + 0.807566 w_k1_FDX_s013
                                 + 0.916904 w_k1_IBM_s013
                                 + 0.833853 w_k1_INTC_s013
                                 + 0.886686 w_k1_KO_s013
                                 + 0.988399 w_k1_MAT_s013
                                 + 0.813208 w_k1_MSFT_s013
                                 + 0.971546 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.752261 w_k1_ADM_s013
                                 + 0.631007 w_k1_AXP_s013
                                 + 0.531165 w_k1_DIS_s013
                                 + 0.685154 w_k1_FDX_s013
                                 + 0.726451 w_k1_IBM_s013
                                 + 0.727705 w_k1_INTC_s013
                                 + 0.683353 w_k1_KO_s013
                                 + 0.564339 w_k1_MAT_s013
                                 + 0.715559 w_k1_MSFT_s013
                                 + 0.859555 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.2941 w_k1_ADM_s014
                                 + 1.2704 w_k1_AXP_s014 + 1.88529 w_k1_DIS_s014
                                 + 1.36699 w_k1_FDX_s014
                                 + 1.25081 w_k1_IBM_s014
                                 + 1.52749 w_k1_INTC_s014 + 1.4517 w_k1_KO_s014
                                 + 1.11651 w_k1_MAT_s014
                                 + 1.45354 w_k1_MSFT_s014
                                 + 1.30606 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.07356 w_k1_ADM_s014
                                 + 1.22872 w_k1_AXP_s014
                                 + 1.28651 w_k1_DIS_s014
                                 + 1.23633 w_k1_FDX_s014
                                 + 0.929141 w_k1_IBM_s014
                                 + 1.27719 w_k1_INTC_s014
                                 + 1.15106 w_k1_KO_s014 + 1.15793 w_k1_MAT_s014
                                 + 1.14193 w_k1_MSFT_s014
                                 + 1.14236 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.850727 w_k1_ADM_s014
                                 + 0.773384 w_k1_AXP_s014
                                 + 1.10754 w_k1_DIS_s014
                                 + 0.948208 w_k1_FDX_s014
                                 + 1.06572 w_k1_IBM_s014
                                 + 1.12829 w_k1_INTC_s014
                                 + 1.03514 w_k1_KO_s014
                                 + 0.788568 w_k1_MAT_s014
                                 + 0.994855 w_k1_MSFT_s014
                                 + 0.923122 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.04288 w_k1_ADM_s014
                                 + 0.942608 w_k1_AXP_s014
                                 + 1.1298 w_k1_DIS_s014 + 1.09399 w_k1_FDX_s014
                                 + 0.798989 w_k1_IBM_s014
                                 + 1.06848 w_k1_INTC_s014 + 1.0777 w_k1_KO_s014
                                 + 0.731596 w_k1_MAT_s014
                                 + 1.19596 w_k1_MSFT_s014
                                 + 0.953406 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.803295 w_k1_ADM_s014
                                 + 0.775342 w_k1_AXP_s014
                                 + 0.685304 w_k1_DIS_s014
                                 + 0.783731 w_k1_FDX_s014
                                 + 0.867675 w_k1_IBM_s014
                                 + 0.913391 w_k1_INTC_s014
                                 + 0.730691 w_k1_KO_s014
                                 + 0.710394 w_k1_MAT_s014
                                 + 0.882064 w_k1_MSFT_s014
                                 + 0.81503 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.836373 w_k1_ADM_s014
                                 + 1.0514 w_k1_AXP_s014 + 1.26715 w_k1_DIS_s014
                                 + 1.07172 w_k1_FDX_s014
                                 + 1.38382 w_k1_IBM_s014
                                 + 1.33099 w_k1_INTC_s014
                                 + 1.23862 w_k1_KO_s014 + 0.94179 w_k1_MAT_s014
                                 + 1.25612 w_k1_MSFT_s014
                                 + 1.06867 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.711866 w_k1_ADM_s014
                                 + 1.01262 w_k1_AXP_s014
                                 + 1.11623 w_k1_DIS_s014
                                 + 0.929944 w_k1_FDX_s014
                                 + 1.63201 w_k1_IBM_s014
                                 + 0.991778 w_k1_INTC_s014
                                 + 1.15753 w_k1_KO_s014 + 1.80807 w_k1_MAT_s014
                                 + 1.06375 w_k1_MSFT_s014
                                 + 1.08472 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.11588 w_k1_ADM_s014
                                 + 1.06237 w_k1_AXP_s014
                                 + 0.802657 w_k1_DIS_s014
                                 + 0.954028 w_k1_FDX_s014
                                 + 1.00059 w_k1_IBM_s014
                                 + 0.984988 w_k1_INTC_s014
                                 + 0.845223 w_k1_KO_s014
                                 + 0.939861 w_k1_MAT_s014
                                 + 0.970272 w_k1_MSFT_s014
                                 + 0.978874 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.05519 w_k1_ADM_s015
                                 + 0.890514 w_k1_AXP_s015
                                 + 0.80981 w_k1_DIS_s015
                                 + 0.951547 w_k1_FDX_s015
                                 + 1.01776 w_k1_IBM_s015
                                 + 1.06782 w_k1_INTC_s015
                                 + 0.930585 w_k1_KO_s015
                                 + 0.624548 w_k1_MAT_s015
                                 + 0.935046 w_k1_MSFT_s015
                                 + 0.999025 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.54806 w_k1_ADM_s015
                                 + 1.23137 w_k1_AXP_s015
                                 + 1.79087 w_k1_DIS_s015
                                 + 1.30023 w_k1_FDX_s015
                                 + 1.52625 w_k1_IBM_s015
                                 + 1.34119 w_k1_INTC_s015 + 1.4586 w_k1_KO_s015
                                 + 0.973841 w_k1_MAT_s015
                                 + 1.28885 w_k1_MSFT_s015
                                 + 1.20634 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.825773 w_k1_ADM_s015
                                 + 0.977079 w_k1_AXP_s015
                                 + 0.784128 w_k1_DIS_s015
                                 + 0.799964 w_k1_FDX_s015
                                 + 1.27713 w_k1_IBM_s015
                                 + 0.907824 w_k1_INTC_s015
                                 + 0.777459 w_k1_KO_s015
                                 + 0.721468 w_k1_MAT_s015
                                 + 0.776929 w_k1_MSFT_s015
                                 + 0.853499 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.893251 w_k1_ADM_s015
                                 + 0.65809 w_k1_AXP_s015
                                 + 0.719404 w_k1_DIS_s015
                                 + 0.811058 w_k1_FDX_s015
                                 + 0.663422 w_k1_IBM_s015
                                 + 0.912782 w_k1_INTC_s015
                                 + 0.717372 w_k1_KO_s015
                                 + 0.642672 w_k1_MAT_s015
                                 + 0.936208 w_k1_MSFT_s015
                                 + 0.827074 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.18012 w_k1_ADM_s015
                                 + 1.26997 w_k1_AXP_s015
                                 + 1.46311 w_k1_DIS_s015
                                 + 1.27588 w_k1_FDX_s015
                                 + 0.987258 w_k1_IBM_s015
                                 + 0.918656 w_k1_INTC_s015
                                 + 1.11172 w_k1_KO_s015
                                 + 0.782954 w_k1_MAT_s015
                                 + 1.05941 w_k1_MSFT_s015
                                 + 1.05596 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.989431 w_k1_ADM_s015
                                 + 1.39451 w_k1_AXP_s015
                                 + 1.13258 w_k1_DIS_s015
                                 + 1.12308 w_k1_FDX_s015 + 1.2901 w_k1_IBM_s015
                                 + 1.20307 w_k1_INTC_s015
                                 + 1.05166 w_k1_KO_s015 + 1.27785 w_k1_MAT_s015
                                 + 1.09875 w_k1_MSFT_s015 + 1.1573 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.744444 w_k1_ADM_s015
                                 + 0.916709 w_k1_AXP_s015
                                 + 1.29446 w_k1_DIS_s015
                                 + 0.910025 w_k1_FDX_s015
                                 + 1.51728 w_k1_IBM_s015
                                 + 1.22522 w_k1_INTC_s015
                                 + 1.23392 w_k1_KO_s015 + 1.04257 w_k1_MAT_s015
                                 + 1.15126 w_k1_MSFT_s015
                                 + 0.966999 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.1103 w_k1_ADM_s015
                                 + 0.984205 w_k1_AXP_s015
                                 + 1.11511 w_k1_DIS_s015
                                 + 0.970836 w_k1_FDX_s015
                                 + 0.838458 w_k1_IBM_s015
                                 + 1.08945 w_k1_INTC_s015
                                 + 0.963214 w_k1_KO_s015
                                 + 0.79277 w_k1_MAT_s015
                                 + 0.9815 w_k1_MSFT_s015
                                 + 0.919837 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.28689 w_k1_ADM_s016
                                 + 0.887562 w_k1_AXP_s016
                                 + 1.13464 w_k1_DIS_s016
                                 + 1.05172 w_k1_FDX_s016
                                 + 1.00713 w_k1_IBM_s016
                                 + 0.859585 w_k1_INTC_s016
                                 + 1.04427 w_k1_KO_s016
                                 + 0.684379 w_k1_MAT_s016
                                 + 1.09822 w_k1_MSFT_s016
                                 + 1.04356 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.61292 w_k1_ADM_s016
                                 + 1.40642 w_k1_AXP_s016
                                 + 1.69242 w_k1_DIS_s016
                                 + 1.49578 w_k1_FDX_s016
                                 + 1.37437 w_k1_IBM_s016
                                 + 1.11539 w_k1_INTC_s016
                                 + 1.45956 w_k1_KO_s016 + 1.45609 w_k1_MAT_s016
                                 + 1.17639 w_k1_MSFT_s016
                                 + 1.25019 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.870818 w_k1_ADM_s016
                                 + 0.89504 w_k1_AXP_s016
                                 + 0.708709 w_k1_DIS_s016
                                 + 0.926493 w_k1_FDX_s016
                                 + 0.818198 w_k1_IBM_s016
                                 + 0.772856 w_k1_INTC_s016
                                 + 0.902455 w_k1_KO_s016
                                 + 1.25573 w_k1_MAT_s016
                                 + 0.94589 w_k1_MSFT_s016
                                 + 0.98336 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.06495 w_k1_ADM_s016
                                 + 0.949429 w_k1_AXP_s016
                                 + 0.965574 w_k1_DIS_s016
                                 + 0.930545 w_k1_FDX_s016
                                 + 1.24731 w_k1_IBM_s016
                                 + 1.01347 w_k1_INTC_s016
                                 + 1.13236 w_k1_KO_s016
                                 + 0.956245 w_k1_MAT_s016
                                 + 0.991623 w_k1_MSFT_s016
                                 + 0.977162 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.829756 w_k1_ADM_s016
                                 + 0.724479 w_k1_AXP_s016
                                 + 0.79002 w_k1_DIS_s016
                                 + 0.892702 w_k1_FDX_s016
                                 + 0.640039 w_k1_IBM_s016
                                 + 1.00513 w_k1_INTC_s016
                                 + 0.846446 w_k1_KO_s016
                                 + 0.372449 w_k1_MAT_s016
                                 + 0.865837 w_k1_MSFT_s016
                                 + 0.825782 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.876013 w_k1_ADM_s016
                                 + 1.02451 w_k1_AXP_s016
                                 + 0.800345 w_k1_DIS_s016
                                 + 0.807209 w_k1_FDX_s016
                                 + 1.22568 w_k1_IBM_s016
                                 + 0.885694 w_k1_INTC_s016
                                 + 0.941334 w_k1_KO_s016
                                 + 0.940686 w_k1_MAT_s016
                                 + 0.749477 w_k1_MSFT_s016
                                 + 0.907518 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.26854 w_k1_ADM_s016
                                 + 1.06093 w_k1_AXP_s016
                                 + 1.02422 w_k1_DIS_s016
                                 + 1.09124 w_k1_FDX_s016
                                 + 0.90544 w_k1_IBM_s016
                                 + 0.929817 w_k1_INTC_s016
                                 + 1.15069 w_k1_KO_s016 + 0.99981 w_k1_MAT_s016
                                 + 1.02441 w_k1_MSFT_s016
                                 + 0.975744 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.00777 w_k1_ADM_s016
                                 + 0.872688 w_k1_AXP_s016
                                 + 1.42556 w_k1_DIS_s016
                                 + 1.13149 w_k1_FDX_s016
                                 + 0.922028 w_k1_IBM_s016
                                 + 1.02064 w_k1_INTC_s016
                                 + 1.19776 w_k1_KO_s016 + 0.76184 w_k1_MAT_s016
                                 + 1.23543 w_k1_MSFT_s016
                                 + 0.91396 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.819847 w_k1_ADM_s017
                                 + 0.860743 w_k1_AXP_s017
                                 + 0.657865 w_k1_DIS_s017
                                 + 0.712704 w_k1_FDX_s017
                                 + 1.33659 w_k1_IBM_s017
                                 + 0.669746 w_k1_INTC_s017
                                 + 0.929482 w_k1_KO_s017 + 1.5938 w_k1_MAT_s017
                                 + 0.749648 w_k1_MSFT_s017
                                 + 0.905131 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.27996 w_k1_ADM_s017
                                 + 1.60954 w_k1_AXP_s017
                                 + 1.68096 w_k1_DIS_s017
                                 + 1.28969 w_k1_FDX_s017
                                 + 1.29416 w_k1_IBM_s017
                                 + 1.24634 w_k1_INTC_s017
                                 + 1.38527 w_k1_KO_s017 + 1.09625 w_k1_MAT_s017
                                 + 1.24757 w_k1_MSFT_s017
                                 + 1.28298 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.20514 w_k1_ADM_s017
                                 + 1.67523 w_k1_AXP_s017
                                 + 1.31569 w_k1_DIS_s017
                                 + 1.16188 w_k1_FDX_s017 + 1.1379 w_k1_IBM_s017
                                 + 1.09352 w_k1_INTC_s017 + 1.128 w_k1_KO_s017
                                 + 1.22037 w_k1_MAT_s017
                                 + 1.10352 w_k1_MSFT_s017
                                 + 1.17619 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.95263 w_k1_ADM_s017
                                 + 1.102 w_k1_AXP_s017 + 1.03203 w_k1_DIS_s017
                                 + 1.08605 w_k1_FDX_s017
                                 + 0.82232 w_k1_IBM_s017
                                 + 1.05067 w_k1_INTC_s017
                                 + 0.978407 w_k1_KO_s017
                                 + 0.802599 w_k1_MAT_s017
                                 + 1.29456 w_k1_MSFT_s017
                                 + 1.08554 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.860899 w_k1_ADM_s017
                                 + 0.819427 w_k1_AXP_s017
                                 + 0.78501 w_k1_DIS_s017
                                 + 0.841348 w_k1_FDX_s017
                                 + 0.720414 w_k1_IBM_s017
                                 + 0.866433 w_k1_INTC_s017
                                 + 0.857149 w_k1_KO_s017
                                 + 0.879878 w_k1_MAT_s017
                                 + 0.875428 w_k1_MSFT_s017
                                 + 0.925196 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.716862 w_k1_ADM_s017
                                 + 0.88379 w_k1_AXP_s017
                                 + 0.672807 w_k1_DIS_s017
                                 + 0.804944 w_k1_FDX_s017
                                 + 0.86792 w_k1_IBM_s017
                                 + 0.89311 w_k1_INTC_s017
                                 + 0.881561 w_k1_KO_s017
                                 + 0.843018 w_k1_MAT_s017
                                 + 0.947454 w_k1_MSFT_s017
                                 + 0.869164 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.926787 w_k1_ADM_s017
                                 + 1.45698 w_k1_AXP_s017
                                 + 1.21359 w_k1_DIS_s017
                                 + 1.04158 w_k1_FDX_s017
                                 + 1.48645 w_k1_IBM_s017
                                 + 1.14335 w_k1_INTC_s017
                                 + 1.16773 w_k1_KO_s017 + 1.76089 w_k1_MAT_s017
                                 + 1.22984 w_k1_MSFT_s017
                                 + 1.20322 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.725563 w_k1_ADM_s017
                                 + 1.1995 w_k1_AXP_s017
                                 + 0.937228 w_k1_DIS_s017
                                 + 0.922798 w_k1_FDX_s017
                                 + 1.27111 w_k1_IBM_s017
                                 + 0.873882 w_k1_INTC_s017
                                 + 1.15697 w_k1_KO_s017 + 1.67216 w_k1_MAT_s017
                                 + 1.05331 w_k1_MSFT_s017
                                 + 1.04621 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.22871 w_k1_ADM_s100
                                 + 1.34655 w_k1_AXP_s100
                                 + 1.50909 w_k1_DIS_s100
                                 + 1.23138 w_k1_FDX_s100
                                 + 1.16006 w_k1_IBM_s100
                                 + 1.40631 w_k1_INTC_s100
                                 + 1.23497 w_k1_KO_s100 + 1.38025 w_k1_MAT_s100
                                 + 1.42231 w_k1_MSFT_s100
                                 + 1.13382 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.753027 w_k1_ADM_s100
                                 + 0.791281 w_k1_AXP_s100
                                 + 0.711086 w_k1_DIS_s100
                                 + 0.795205 w_k1_FDX_s100
                                 + 1.09236 w_k1_IBM_s100
                                 + 0.883937 w_k1_INTC_s100
                                 + 0.871969 w_k1_KO_s100
                                 + 1.06308 w_k1_MAT_s100
                                 + 0.868357 w_k1_MSFT_s100
                                 + 0.943514 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.969055 w_k1_ADM_s100
                                 + 1.10347 w_k1_AXP_s100
                                 + 0.942853 w_k1_DIS_s100
                                 + 0.940706 w_k1_FDX_s100
                                 + 1.67187 w_k1_IBM_s100
                                 + 1.05562 w_k1_INTC_s100
                                 + 1.14547 w_k1_KO_s100 + 1.97951 w_k1_MAT_s100
                                 + 1.0223 w_k1_MSFT_s100 + 1.18938 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.14995 w_k1_ADM_s100
                                 + 1.17759 w_k1_AXP_s100
                                 + 0.949892 w_k1_DIS_s100
                                 + 1.0324 w_k1_FDX_s100
                                 + 0.939664 w_k1_IBM_s100
                                 + 0.988657 w_k1_INTC_s100
                                 + 1.02675 w_k1_KO_s100 + 1.38291 w_k1_MAT_s100
                                 + 1.10902 w_k1_MSFT_s100
                                 + 1.01549 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.762778 w_k1_ADM_s100
                                 + 0.819652 w_k1_AXP_s100
                                 + 0.647233 w_k1_DIS_s100
                                 + 0.805876 w_k1_FDX_s100
                                 + 0.780386 w_k1_IBM_s100
                                 + 0.927148 w_k1_INTC_s100
                                 + 0.768237 w_k1_KO_s100
                                 + 0.948961 w_k1_MAT_s100
                                 + 0.870168 w_k1_MSFT_s100
                                 + 0.823905 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.974966 w_k1_ADM_s100
                                 + 0.679071 w_k1_AXP_s100
                                 + 0.912562 w_k1_DIS_s100
                                 + 0.987365 w_k1_FDX_s100
                                 + 0.538569 w_k1_IBM_s100
                                 + 1.10397 w_k1_INTC_s100
                                 + 0.735852 w_k1_KO_s100
                                 + 0.446748 w_k1_MAT_s100
                                 + 0.985101 w_k1_MSFT_s100
                                 + 0.907809 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.05475 w_k1_ADM_s100
                                 + 0.849361 w_k1_AXP_s100
                                 + 1.19096 w_k1_DIS_s100
                                 + 1.02654 w_k1_FDX_s100
                                 + 0.821281 w_k1_IBM_s100
                                 + 0.953624 w_k1_INTC_s100
                                 + 0.92186 w_k1_KO_s100
                                 + 0.940404 w_k1_MAT_s100
                                 + 1.01163 w_k1_MSFT_s100 + 1.0333 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.09792 w_k1_ADM_s100
                                 + 1.35049 w_k1_AXP_s100
                                 + 1.00213 w_k1_DIS_s100
                                 + 1.30284 w_k1_FDX_s100
                                 + 0.691411 w_k1_IBM_s100
                                 + 1.02848 w_k1_INTC_s100
                                 + 0.86291 w_k1_KO_s100
                                 + 0.905383 w_k1_MAT_s100
                                 + 1.1707 w_k1_MSFT_s100
                                 + 0.987042 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.46551 w_k1_ADM_s101
                                 + 1.41535 w_k1_AXP_s101
                                 + 1.79042 w_k1_DIS_s101
                                 + 1.40832 w_k1_FDX_s101
                                 + 1.31071 w_k1_IBM_s101
                                 + 0.979797 w_k1_INTC_s101
                                 + 1.35898 w_k1_KO_s101
                                 + 0.991337 w_k1_MAT_s101
                                 + 1.44577 w_k1_MSFT_s101
                                 + 1.32293 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.933762 w_k1_ADM_s101
                                 + 0.979958 w_k1_AXP_s101
                                 + 0.9161 w_k1_DIS_s101
                                 + 0.944552 w_k1_FDX_s101
                                 + 1.28484 w_k1_IBM_s101
                                 + 0.895305 w_k1_INTC_s101
                                 + 1.05032 w_k1_KO_s101 + 1.17386 w_k1_MAT_s101
                                 + 0.981428 w_k1_MSFT_s101
                                 + 0.885145 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.15041 w_k1_ADM_s101
                                 + 0.752505 w_k1_AXP_s101
                                 + 0.819443 w_k1_DIS_s101
                                 + 0.911217 w_k1_FDX_s101
                                 + 0.636818 w_k1_IBM_s101
                                 + 1.02799 w_k1_INTC_s101
                                 + 0.792596 w_k1_KO_s101
                                 + 0.476792 w_k1_MAT_s101
                                 + 1.01405 w_k1_MSFT_s101
                                 + 0.855244 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.865186 w_k1_ADM_s101
                                 + 0.760803 w_k1_AXP_s101
                                 + 0.685349 w_k1_DIS_s101
                                 + 0.785258 w_k1_FDX_s101
                                 + 0.910717 w_k1_IBM_s101
                                 + 0.769063 w_k1_INTC_s101
                                 + 0.862247 w_k1_KO_s101
                                 + 0.583785 w_k1_MAT_s101
                                 + 0.84788 w_k1_MSFT_s101
                                 + 0.794446 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.27206 w_k1_ADM_s101
                                 + 1.27269 w_k1_AXP_s101
                                 + 1.34266 w_k1_DIS_s101
                                 + 1.19556 w_k1_FDX_s101
                                 + 1.00535 w_k1_IBM_s101
                                 + 0.91007 w_k1_INTC_s101
                                 + 1.17165 w_k1_KO_s101
                                 + 0.798485 w_k1_MAT_s101
                                 + 1.1047 w_k1_MSFT_s101 + 1.18622 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.1402 w_k1_ADM_s101
                                 + 1.47826 w_k1_AXP_s101
                                 + 1.36633 w_k1_DIS_s101
                                 + 1.10363 w_k1_FDX_s101
                                 + 1.44485 w_k1_IBM_s101
                                 + 1.05558 w_k1_INTC_s101
                                 + 1.34623 w_k1_KO_s101 + 1.5823 w_k1_MAT_s101
                                 + 1.18505 w_k1_MSFT_s101
                                 + 1.12498 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.25702 w_k1_ADM_s101
                                 + 1.73059 w_k1_AXP_s101
                                 + 1.02325 w_k1_DIS_s101
                                 + 1.19676 w_k1_FDX_s101
                                 + 1.05803 w_k1_IBM_s101
                                 + 1.14765 w_k1_INTC_s101
                                 + 1.10403 w_k1_KO_s101 + 1.35075 w_k1_MAT_s101
                                 + 1.24617 w_k1_MSFT_s101
                                 + 1.01052 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.04757 w_k1_ADM_s101
                                 + 1.0813 w_k1_AXP_s101 + 1.13507 w_k1_DIS_s101
                                 + 0.995336 w_k1_FDX_s101
                                 + 0.884456 w_k1_IBM_s101
                                 + 0.989153 w_k1_INTC_s101
                                 + 1.09049 w_k1_KO_s101
                                 + 0.814926 w_k1_MAT_s101
                                 + 1.10543 w_k1_MSFT_s101
                                 + 1.03019 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.18373 w_k1_ADM_s102
                                 + 1.1723 w_k1_AXP_s102 + 0.89314 w_k1_DIS_s102
                                 + 0.936414 w_k1_FDX_s102
                                 + 0.979677 w_k1_IBM_s102
                                 + 1.01285 w_k1_INTC_s102
                                 + 1.08224 w_k1_KO_s102
                                 + 0.950549 w_k1_MAT_s102
                                 + 0.905477 w_k1_MSFT_s102
                                 + 0.943856 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.03627 w_k1_ADM_s102
                                 + 0.797329 w_k1_AXP_s102
                                 + 0.978247 w_k1_DIS_s102
                                 + 1.03443 w_k1_FDX_s102
                                 + 0.592174 w_k1_IBM_s102
                                 + 1.18045 w_k1_INTC_s102
                                 + 0.903341 w_k1_KO_s102
                                 + 0.518325 w_k1_MAT_s102
                                 + 1.09259 w_k1_MSFT_s102
                                 + 0.983641 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.27097 w_k1_ADM_s102
                                 + 1.02428 w_k1_AXP_s102
                                 + 1.14597 w_k1_DIS_s102
                                 + 1.17633 w_k1_FDX_s102
                                 + 0.922325 w_k1_IBM_s102
                                 + 1.3252 w_k1_INTC_s102 + 1.0292 w_k1_KO_s102
                                 + 0.48712 w_k1_MAT_s102
                                 + 1.00821 w_k1_MSFT_s102
                                 + 1.12156 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.39891 w_k1_ADM_s102
                                 + 1.56205 w_k1_AXP_s102
                                 + 1.65083 w_k1_DIS_s102
                                 + 1.38467 w_k1_FDX_s102
                                 + 1.27209 w_k1_IBM_s102
                                 + 1.39322 w_k1_INTC_s102
                                 + 1.21699 w_k1_KO_s102 + 1.00017 w_k1_MAT_s102
                                 + 1.11359 w_k1_MSFT_s102
                                 + 1.06019 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.00854 w_k1_ADM_s102
                                 + 0.855945 w_k1_AXP_s102
                                 + 0.774881 w_k1_DIS_s102
                                 + 0.844318 w_k1_FDX_s102
                                 + 0.876643 w_k1_IBM_s102
                                 + 0.916868 w_k1_INTC_s102
                                 + 0.895226 w_k1_KO_s102
                                 + 0.619501 w_k1_MAT_s102
                                 + 0.806895 w_k1_MSFT_s102
                                 + 0.914316 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.39095 w_k1_ADM_s102
                                 + 1.29875 w_k1_AXP_s102
                                 + 1.44291 w_k1_DIS_s102
                                 + 1.20427 w_k1_FDX_s102
                                 + 1.36847 w_k1_IBM_s102
                                 + 1.47337 w_k1_INTC_s102
                                 + 1.48068 w_k1_KO_s102 + 1.3212 w_k1_MAT_s102
                                 + 1.06446 w_k1_MSFT_s102
                                 + 1.27849 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.772768 w_k1_ADM_s102
                                 + 0.606898 w_k1_AXP_s102
                                 + 0.542064 w_k1_DIS_s102
                                 + 0.713967 w_k1_FDX_s102
                                 + 0.641937 w_k1_IBM_s102
                                 + 0.902516 w_k1_INTC_s102
                                 + 0.707442 w_k1_KO_s102
                                 + 0.699867 w_k1_MAT_s102
                                 + 0.846618 w_k1_MSFT_s102
                                 + 0.793378 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.01787 w_k1_ADM_s102
                                 + 1.19586 w_k1_AXP_s102
                                 + 1.09306 w_k1_DIS_s102
                                 + 0.941582 w_k1_FDX_s102
                                 + 1.61247 w_k1_IBM_s102
                                 + 1.10455 w_k1_INTC_s102
                                 + 1.43079 w_k1_KO_s102 + 1.44016 w_k1_MAT_s102
                                 + 0.925711 w_k1_MSFT_s102
                                 + 1.18191 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.04471 w_k1_ADM_s103
                                 + 1.10611 w_k1_AXP_s103
                                 + 1.17277 w_k1_DIS_s103
                                 + 1.12616 w_k1_FDX_s103
                                 + 1.30452 w_k1_IBM_s103
                                 + 1.0114 w_k1_INTC_s103 + 1.08268 w_k1_KO_s103
                                 + 1.50352 w_k1_MAT_s103
                                 + 0.966434 w_k1_MSFT_s103
                                 + 1.11106 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.02589 w_k1_ADM_s103
                                 + 0.876065 w_k1_AXP_s103
                                 + 1.30651 w_k1_DIS_s103
                                 + 1.16776 w_k1_FDX_s103
                                 + 0.725138 w_k1_IBM_s103
                                 + 0.982015 w_k1_INTC_s103
                                 + 1.03447 w_k1_KO_s103
                                 + 0.872676 w_k1_MAT_s103
                                 + 1.04207 w_k1_MSFT_s103
                                 + 0.962393 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.867278 w_k1_ADM_s103
                                 + 0.559884 w_k1_AXP_s103
                                 + 0.814037 w_k1_DIS_s103
                                 + 0.83546 w_k1_FDX_s103
                                 + 0.770123 w_k1_IBM_s103
                                 + 0.855229 w_k1_INTC_s103
                                 + 0.844301 w_k1_KO_s103
                                 + 0.652863 w_k1_MAT_s103
                                 + 0.863281 w_k1_MSFT_s103
                                 + 0.783412 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.41185 w_k1_ADM_s103
                                 + 1.12895 w_k1_AXP_s103
                                 + 1.62228 w_k1_DIS_s103
                                 + 1.36515 w_k1_FDX_s103
                                 + 0.991442 w_k1_IBM_s103
                                 + 1.06499 w_k1_INTC_s103
                                 + 1.15513 w_k1_KO_s103 + 0.84218 w_k1_MAT_s103
                                 + 1.23427 w_k1_MSFT_s103
                                 + 1.09551 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.09663 w_k1_ADM_s103
                                 + 0.888986 w_k1_AXP_s103
                                 + 1.14102 w_k1_DIS_s103
                                 + 0.915101 w_k1_FDX_s103
                                 + 1.32497 w_k1_IBM_s103
                                 + 1.1968 w_k1_INTC_s103 + 1.13345 w_k1_KO_s103
                                 + 1.14064 w_k1_MAT_s103
                                 + 1.04149 w_k1_MSFT_s103
                                 + 1.07954 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.35541 w_k1_ADM_s103
                                 + 1.53378 w_k1_AXP_s103
                                 + 1.73179 w_k1_DIS_s103
                                 + 1.31508 w_k1_FDX_s103
                                 + 1.46967 w_k1_IBM_s103
                                 + 1.29498 w_k1_INTC_s103
                                 + 1.42364 w_k1_KO_s103 + 1.86003 w_k1_MAT_s103
                                 + 1.16302 w_k1_MSFT_s103
                                 + 1.15843 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.46209 w_k1_ADM_s103
                                 + 1.18881 w_k1_AXP_s103
                                 + 0.985785 w_k1_DIS_s103
                                 + 1.14834 w_k1_FDX_s103
                                 + 0.900182 w_k1_IBM_s103
                                 + 1.19662 w_k1_INTC_s103
                                 + 0.871953 w_k1_KO_s103
                                 + 0.591984 w_k1_MAT_s103
                                 + 1.05892 w_k1_MSFT_s103
                                 + 0.977732 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.14381 w_k1_ADM_s103
                                 + 0.73499 w_k1_AXP_s103
                                 + 0.782935 w_k1_DIS_s103
                                 + 0.811582 w_k1_FDX_s103
                                 + 0.988687 w_k1_IBM_s103
                                 + 0.990039 w_k1_INTC_s103
                                 + 0.792309 w_k1_KO_s103
                                 + 0.89956 w_k1_MAT_s103
                                 + 0.871429 w_k1_MSFT_s103
                                 + 0.835091 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.09643 w_k1_ADM_s104
                                 + 1.07938 w_k1_AXP_s104
                                 + 0.878145 w_k1_DIS_s104
                                 + 1.01003 w_k1_FDX_s104
                                 + 0.896388 w_k1_IBM_s104
                                 + 0.902861 w_k1_INTC_s104
                                 + 0.812075 w_k1_KO_s104
                                 + 1.34765 w_k1_MAT_s104
                                 + 0.862027 w_k1_MSFT_s104
                                 + 0.973108 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.841153 w_k1_ADM_s104
                                 + 0.901557 w_k1_AXP_s104
                                 + 0.796666 w_k1_DIS_s104
                                 + 0.849592 w_k1_FDX_s104
                                 + 1.29112 w_k1_IBM_s104
                                 + 1.00087 w_k1_INTC_s104
                                 + 0.93725 w_k1_KO_s104 + 1.4547 w_k1_MAT_s104
                                 + 0.862098 w_k1_MSFT_s104
                                 + 0.846563 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.796257 w_k1_ADM_s104
                                 + 0.74743 w_k1_AXP_s104
                                 + 0.622847 w_k1_DIS_s104
                                 + 0.722755 w_k1_FDX_s104
                                 + 0.85581 w_k1_IBM_s104
                                 + 0.605641 w_k1_INTC_s104
                                 + 0.593219 w_k1_KO_s104
                                 + 0.941392 w_k1_MAT_s104
                                 + 0.650613 w_k1_MSFT_s104
                                 + 0.706282 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.845776 w_k1_ADM_s104
                                 + 0.816075 w_k1_AXP_s104
                                 + 0.89409 w_k1_DIS_s104
                                 + 0.913665 w_k1_FDX_s104
                                 + 0.871397 w_k1_IBM_s104
                                 + 0.94262 w_k1_INTC_s104
                                 + 0.927904 w_k1_KO_s104
                                 + 0.790702 w_k1_MAT_s104
                                 + 0.934381 w_k1_MSFT_s104
                                 + 0.944853 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.22988 w_k1_ADM_s104
                                 + 1.51506 w_k1_AXP_s104
                                 + 1.80029 w_k1_DIS_s104
                                 + 1.38584 w_k1_FDX_s104
                                 + 1.13703 w_k1_IBM_s104
                                 + 1.29326 w_k1_INTC_s104
                                 + 1.17305 w_k1_KO_s104 + 1.14587 w_k1_MAT_s104
                                 + 1.25861 w_k1_MSFT_s104
                                 + 1.18621 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.908382 w_k1_ADM_s104
                                 + 1.37967 w_k1_AXP_s104 + 1.2121 w_k1_DIS_s104
                                 + 1.14975 w_k1_FDX_s104
                                 + 1.07642 w_k1_IBM_s104
                                 + 1.12287 w_k1_INTC_s104
                                 + 1.03761 w_k1_KO_s104 + 1.18666 w_k1_MAT_s104
                                 + 1.03333 w_k1_MSFT_s104
                                 + 1.03129 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.28102 w_k1_ADM_s104
                                 + 1.11243 w_k1_AXP_s104
                                 + 1.30239 w_k1_DIS_s104
                                 + 1.08912 w_k1_FDX_s104
                                 + 1.27978 w_k1_IBM_s104
                                 + 1.0725 w_k1_INTC_s104 + 1.09281 w_k1_KO_s104
                                 + 1.19868 w_k1_MAT_s104
                                 + 1.04733 w_k1_MSFT_s104
                                 + 0.951922 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.79908 w_k1_ADM_s104
                                 + 0.796577 w_k1_AXP_s104
                                 + 0.634045 w_k1_DIS_s104
                                 + 0.885324 w_k1_FDX_s104
                                 + 0.746534 w_k1_IBM_s104
                                 + 0.879939 w_k1_INTC_s104
                                 + 0.722936 w_k1_KO_s104
                                 + 0.736301 w_k1_MAT_s104
                                 + 0.750879 w_k1_MSFT_s104
                                 + 0.836672 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.93835 w_k1_ADM_s105
                                 + 0.993408 w_k1_AXP_s105
                                 + 0.768179 w_k1_DIS_s105
                                 + 0.946612 w_k1_FDX_s105
                                 + 0.815878 w_k1_IBM_s105
                                 + 0.945301 w_k1_INTC_s105
                                 + 0.793423 w_k1_KO_s105
                                 + 1.03651 w_k1_MAT_s105
                                 + 1.03823 w_k1_MSFT_s105
                                 + 0.936084 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.0666 w_k1_ADM_s105
                                 + 1.35439 w_k1_AXP_s105
                                 + 1.00191 w_k1_DIS_s105
                                 + 0.993689 w_k1_FDX_s105
                                 + 1.73505 w_k1_IBM_s105
                                 + 1.12339 w_k1_INTC_s105
                                 + 1.27497 w_k1_KO_s105 + 2.30433 w_k1_MAT_s105
                                 + 1.02933 w_k1_MSFT_s105
                                 + 0.961698 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.27155 w_k1_ADM_s105
                                 + 1.25638 w_k1_AXP_s105
                                 + 1.58895 w_k1_DIS_s105
                                 + 1.28793 w_k1_FDX_s105
                                 + 1.38099 w_k1_IBM_s105
                                 + 1.04089 w_k1_INTC_s105
                                 + 1.52273 w_k1_KO_s105 + 2.18589 w_k1_MAT_s105
                                 + 1.36793 w_k1_MSFT_s105
                                 + 1.34193 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.02467 w_k1_ADM_s105
                                 + 1.52115 w_k1_AXP_s105 + 1.0622 w_k1_DIS_s105
                                 + 1.10782 w_k1_FDX_s105
                                 + 1.09785 w_k1_IBM_s105
                                 + 1.07429 w_k1_INTC_s105
                                 + 0.938618 w_k1_KO_s105
                                 + 1.74431 w_k1_MAT_s105
                                 + 0.993401 w_k1_MSFT_s105
                                 + 1.13004 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.00472 w_k1_ADM_s105
                                 + 1.18117 w_k1_AXP_s105
                                 + 0.860356 w_k1_DIS_s105
                                 + 0.868445 w_k1_FDX_s105
                                 + 1.28041 w_k1_IBM_s105
                                 + 1.00874 w_k1_INTC_s105
                                 + 1.00476 w_k1_KO_s105 + 1.30847 w_k1_MAT_s105
                                 + 0.882717 w_k1_MSFT_s105
                                 + 1.00542 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.35561 w_k1_ADM_s105
                                 + 1.26311 w_k1_AXP_s105
                                 + 1.47865 w_k1_DIS_s105
                                 + 1.37179 w_k1_FDX_s105
                                 + 0.843226 w_k1_IBM_s105
                                 + 1.14356 w_k1_INTC_s105
                                 + 1.01582 w_k1_KO_s105 + 1.00361 w_k1_MAT_s105
                                 + 1.23034 w_k1_MSFT_s105
                                 + 1.09368 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.894945 w_k1_ADM_s105
                                 + 0.891614 w_k1_AXP_s105
                                 + 0.823253 w_k1_DIS_s105
                                 + 0.824897 w_k1_FDX_s105
                                 + 0.854846 w_k1_IBM_s105
                                 + 0.808309 w_k1_INTC_s105
                                 + 0.876009 w_k1_KO_s105
                                 + 1.24562 w_k1_MAT_s105
                                 + 0.976502 w_k1_MSFT_s105
                                 + 0.918726 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.00975 w_k1_ADM_s105
                                 + 1.08215 w_k1_AXP_s105 + 1.2966 w_k1_DIS_s105
                                 + 1.11411 w_k1_FDX_s105
                                 + 0.956483 w_k1_IBM_s105
                                 + 0.980103 w_k1_INTC_s105
                                 + 0.960726 w_k1_KO_s105 + 1.3261 w_k1_MAT_s105
                                 + 1.20233 w_k1_MSFT_s105
                                 + 0.956658 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.830292 w_k1_ADM_s106
                                 + 0.775766 w_k1_AXP_s106
                                 + 0.785494 w_k1_DIS_s106
                                 + 0.817848 w_k1_FDX_s106
                                 + 0.764807 w_k1_IBM_s106
                                 + 0.994246 w_k1_INTC_s106
                                 + 0.781288 w_k1_KO_s106
                                 + 0.502479 w_k1_MAT_s106
                                 + 0.830587 w_k1_MSFT_s106
                                 + 0.798096 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.996147 w_k1_ADM_s106
                                 + 0.949499 w_k1_AXP_s106
                                 + 1.01119 w_k1_DIS_s106
                                 + 1.10849 w_k1_FDX_s106
                                 + 0.784211 w_k1_IBM_s106
                                 + 1.11003 w_k1_INTC_s106
                                 + 0.982011 w_k1_KO_s106
                                 + 0.76812 w_k1_MAT_s106
                                 + 0.970873 w_k1_MSFT_s106
                                 + 1.05728 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.785559 w_k1_ADM_s106
                                 + 0.833935 w_k1_AXP_s106
                                 + 0.613308 w_k1_DIS_s106
                                 + 0.7493 w_k1_FDX_s106
                                 + 0.743866 w_k1_IBM_s106
                                 + 0.812225 w_k1_INTC_s106
                                 + 0.686818 w_k1_KO_s106
                                 + 0.672951 w_k1_MAT_s106
                                 + 0.742868 w_k1_MSFT_s106
                                 + 0.707621 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.988449 w_k1_ADM_s106
                                 + 0.82826 w_k1_AXP_s106
                                 + 1.33476 w_k1_DIS_s106
                                 + 0.997078 w_k1_FDX_s106
                                 + 1.11635 w_k1_IBM_s106
                                 + 1.13013 w_k1_INTC_s106
                                 + 1.08515 w_k1_KO_s106
                                 + 0.877661 w_k1_MAT_s106
                                 + 1.08592 w_k1_MSFT_s106
                                 + 1.04108 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.85795 w_k1_ADM_s106
                                 + 1.25357 w_k1_AXP_s106
                                 + 1.04751 w_k1_DIS_s106
                                 + 1.12092 w_k1_FDX_s106
                                 + 1.03253 w_k1_IBM_s106
                                 + 1.18952 w_k1_INTC_s106
                                 + 1.04098 w_k1_KO_s106 + 1.06804 w_k1_MAT_s106
                                 + 0.990228 w_k1_MSFT_s106
                                 + 0.9774 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.04772 w_k1_ADM_s106
                                 + 1.31889 w_k1_AXP_s106
                                 + 1.30235 w_k1_DIS_s106
                                 + 1.22646 w_k1_FDX_s106
                                 + 1.12618 w_k1_IBM_s106
                                 + 1.40283 w_k1_INTC_s106
                                 + 1.24135 w_k1_KO_s106 + 1.28431 w_k1_MAT_s106
                                 + 1.14388 w_k1_MSFT_s106
                                 + 1.11938 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.892194 w_k1_ADM_s106
                                 + 1.34904 w_k1_AXP_s106
                                 + 1.28222 w_k1_DIS_s106
                                 + 1.06152 w_k1_FDX_s106
                                 + 1.74091 w_k1_IBM_s106
                                 + 1.08995 w_k1_INTC_s106
                                 + 1.24175 w_k1_KO_s106 + 1.83343 w_k1_MAT_s106
                                 + 1.05905 w_k1_MSFT_s106
                                 + 1.05911 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.82915 w_k1_ADM_s106
                                 + 0.819661 w_k1_AXP_s106
                                 + 0.815468 w_k1_DIS_s106
                                 + 0.849662 w_k1_FDX_s106
                                 + 1.24128 w_k1_IBM_s106
                                 + 1.03605 w_k1_INTC_s106
                                 + 0.925529 w_k1_KO_s106
                                 + 1.15247 w_k1_MAT_s106
                                 + 0.858702 w_k1_MSFT_s106
                                 + 0.934654 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.784081 w_k1_ADM_s107
                                 + 0.577724 w_k1_AXP_s107
                                 + 0.583708 w_k1_DIS_s107
                                 + 0.791963 w_k1_FDX_s107
                                 + 0.7248 w_k1_IBM_s107
                                 + 0.844507 w_k1_INTC_s107
                                 + 0.684296 w_k1_KO_s107
                                 + 0.537216 w_k1_MAT_s107
                                 + 0.741557 w_k1_MSFT_s107
                                 + 0.713946 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.01765 w_k1_ADM_s107
                                 + 0.98035 w_k1_AXP_s107
                                 + 1.29006 w_k1_DIS_s107
                                 + 1.02491 w_k1_FDX_s107
                                 + 1.29288 w_k1_IBM_s107
                                 + 1.26699 w_k1_INTC_s107
                                 + 1.23594 w_k1_KO_s107 + 0.5986 w_k1_MAT_s107
                                 + 1.07987 w_k1_MSFT_s107
                                 + 1.09099 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.25483 w_k1_ADM_s107
                                 + 1.52773 w_k1_AXP_s107
                                 + 1.51777 w_k1_DIS_s107
                                 + 1.31787 w_k1_FDX_s107 + 1.3844 w_k1_IBM_s107
                                 + 1.37925 w_k1_INTC_s107
                                 + 1.37433 w_k1_KO_s107 + 1.32887 w_k1_MAT_s107
                                 + 1.29194 w_k1_MSFT_s107
                                 + 1.07239 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.853071 w_k1_ADM_s107
                                 + 0.734851 w_k1_AXP_s107
                                 + 0.803606 w_k1_DIS_s107
                                 + 0.862744 w_k1_FDX_s107
                                 + 0.926588 w_k1_IBM_s107
                                 + 1.15775 w_k1_INTC_s107
                                 + 0.656787 w_k1_KO_s107
                                 + 0.532183 w_k1_MAT_s107
                                 + 0.84197 w_k1_MSFT_s107
                                 + 0.77972 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.745476 w_k1_ADM_s107
                                 + 0.922844 w_k1_AXP_s107
                                 + 0.901362 w_k1_DIS_s107
                                 + 0.855654 w_k1_FDX_s107
                                 + 1.43771 w_k1_IBM_s107
                                 + 1.03251 w_k1_INTC_s107
                                 + 0.977792 w_k1_KO_s107
                                 + 0.978221 w_k1_MAT_s107
                                 + 0.851505 w_k1_MSFT_s107
                                 + 0.813262 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.35543 w_k1_ADM_s107
                                 + 1.47514 w_k1_AXP_s107
                                 + 1.29293 w_k1_DIS_s107
                                 + 1.45083 w_k1_FDX_s107
                                 + 0.868026 w_k1_IBM_s107
                                 + 1.5028 w_k1_INTC_s107 + 1.00911 w_k1_KO_s107
                                 + 0.768306 w_k1_MAT_s107
                                 + 1.19331 w_k1_MSFT_s107 + 1.0618 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.912908 w_k1_ADM_s107
                                 + 0.795705 w_k1_AXP_s107
                                 + 0.785417 w_k1_DIS_s107
                                 + 0.932397 w_k1_FDX_s107
                                 + 0.763256 w_k1_IBM_s107
                                 + 1.00505 w_k1_INTC_s107
                                 + 0.781184 w_k1_KO_s107
                                 + 0.605242 w_k1_MAT_s107
                                 + 0.838466 w_k1_MSFT_s107
                                 + 0.93588 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.1412 w_k1_ADM_s107
                                 + 1.11673 w_k1_AXP_s107
                                 + 1.04453 w_k1_DIS_s107
                                 + 1.12358 w_k1_FDX_s107
                                 + 1.02096 w_k1_IBM_s107
                                 + 1.15398 w_k1_INTC_s107
                                 + 0.972132 w_k1_KO_s107
                                 + 0.893016 w_k1_MAT_s107
                                 + 0.990341 w_k1_MSFT_s107
                                 + 1.01948 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13923 w_k1_ADM_s110
                                 + 1.10932 w_k1_AXP_s110
                                 + 1.48407 w_k1_DIS_s110
                                 + 1.25148 w_k1_FDX_s110
                                 + 1.39432 w_k1_IBM_s110
                                 + 1.3171 w_k1_INTC_s110 + 1.49087 w_k1_KO_s110
                                 + 1.6601 w_k1_MAT_s110
                                 + 1.27238 w_k1_MSFT_s110 + 1.1975 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.53611 w_k1_ADM_s110
                                 + 1.33525 w_k1_AXP_s110
                                 + 0.996471 w_k1_DIS_s110
                                 + 1.3666 w_k1_FDX_s110
                                 + 0.792354 w_k1_IBM_s110
                                 + 1.19116 w_k1_INTC_s110
                                 + 0.944617 w_k1_KO_s110
                                 + 0.925925 w_k1_MAT_s110
                                 + 1.2424 w_k1_MSFT_s110 + 1.00014 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09557 w_k1_ADM_s110
                                 + 1.19637 w_k1_AXP_s110
                                 + 1.20458 w_k1_DIS_s110
                                 + 1.16363 w_k1_FDX_s110
                                 + 1.11082 w_k1_IBM_s110
                                 + 1.16726 w_k1_INTC_s110
                                 + 1.20413 w_k1_KO_s110 + 1.35984 w_k1_MAT_s110
                                 + 1.19866 w_k1_MSFT_s110 + 1.0849 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.06944 w_k1_ADM_s110
                                 + 0.836359 w_k1_AXP_s110
                                 + 1.0544 w_k1_DIS_s110 + 1.10526 w_k1_FDX_s110
                                 + 0.807833 w_k1_IBM_s110
                                 + 1.07293 w_k1_INTC_s110
                                 + 0.942827 w_k1_KO_s110
                                 + 0.879804 w_k1_MAT_s110
                                 + 1.12563 w_k1_MSFT_s110
                                 + 1.09432 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.824112 w_k1_ADM_s110
                                 + 0.822905 w_k1_AXP_s110
                                 + 0.715595 w_k1_DIS_s110
                                 + 0.800187 w_k1_FDX_s110
                                 + 0.909909 w_k1_IBM_s110
                                 + 0.961233 w_k1_INTC_s110
                                 + 0.786127 w_k1_KO_s110
                                 + 0.899071 w_k1_MAT_s110
                                 + 0.8076 w_k1_MSFT_s110 + 0.89417 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.802525 w_k1_ADM_s110
                                 + 0.701751 w_k1_AXP_s110
                                 + 0.57261 w_k1_DIS_s110
                                 + 0.737488 w_k1_FDX_s110
                                 + 0.789859 w_k1_IBM_s110
                                 + 0.719181 w_k1_INTC_s110
                                 + 0.795272 w_k1_KO_s110
                                 + 0.850121 w_k1_MAT_s110
                                 + 0.735121 w_k1_MSFT_s110
                                 + 0.867838 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.23859 w_k1_ADM_s110
                                 + 1.54751 w_k1_AXP_s110
                                 + 0.981764 w_k1_DIS_s110
                                 + 1.13988 w_k1_FDX_s110 + 1.4927 w_k1_IBM_s110
                                 + 1.18751 w_k1_INTC_s110
                                 + 1.22495 w_k1_KO_s110 + 1.55206 w_k1_MAT_s110
                                 + 1.10988 w_k1_MSFT_s110
                                 + 1.04004 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.14949 w_k1_ADM_s110
                                 + 1.31753 w_k1_AXP_s110
                                 + 0.816699 w_k1_DIS_s110
                                 + 1.04653 w_k1_FDX_s110
                                 + 0.890825 w_k1_IBM_s110
                                 + 0.86065 w_k1_INTC_s110
                                 + 0.809389 w_k1_KO_s110
                                 + 1.25485 w_k1_MAT_s110
                                 + 0.856339 w_k1_MSFT_s110
                                 + 0.860707 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.952903 w_k1_ADM_s111
                                 + 0.904178 w_k1_AXP_s111
                                 + 0.666943 w_k1_DIS_s111
                                 + 0.870559 w_k1_FDX_s111
                                 + 0.957467 w_k1_IBM_s111
                                 + 1.11708 w_k1_INTC_s111
                                 + 1.07432 w_k1_KO_s111 + 1.1768 w_k1_MAT_s111
                                 + 0.942316 w_k1_MSFT_s111
                                 + 0.827318 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.30791 w_k1_ADM_s111
                                 + 1.15708 w_k1_AXP_s111
                                 + 1.35844 w_k1_DIS_s111
                                 + 1.41996 w_k1_FDX_s111
                                 + 0.990526 w_k1_IBM_s111
                                 + 1.41993 w_k1_INTC_s111
                                 + 1.28194 w_k1_KO_s111 + 1.34119 w_k1_MAT_s111
                                 + 1.3945 w_k1_MSFT_s111 + 1.10875 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.859083 w_k1_ADM_s111
                                 + 0.8558 w_k1_AXP_s111
                                 + 0.844308 w_k1_DIS_s111
                                 + 0.821679 w_k1_FDX_s111
                                 + 1.56301 w_k1_IBM_s111
                                 + 1.26222 w_k1_INTC_s111
                                 + 1.26652 w_k1_KO_s111 + 1.51805 w_k1_MAT_s111
                                 + 1.00129 w_k1_MSFT_s111
                                 + 1.04995 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.4284 w_k1_ADM_s111
                                 + 1.21115 w_k1_AXP_s111
                                 + 1.13584 w_k1_DIS_s111
                                 + 1.08812 w_k1_FDX_s111
                                 + 1.14331 w_k1_IBM_s111
                                 + 1.06672 w_k1_INTC_s111
                                 + 1.33895 w_k1_KO_s111 + 1.65 w_k1_MAT_s111
                                 + 1.10962 w_k1_MSFT_s111
                                 + 1.13032 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.06805 w_k1_ADM_s111
                                 + 1.01294 w_k1_AXP_s111
                                 + 0.940813 w_k1_DIS_s111
                                 + 1.12422 w_k1_FDX_s111
                                 + 0.731123 w_k1_IBM_s111
                                 + 1.16715 w_k1_INTC_s111
                                 + 0.968021 w_k1_KO_s111
                                 + 0.74461 w_k1_MAT_s111
                                 + 1.04885 w_k1_MSFT_s111
                                 + 0.945502 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.2753 w_k1_ADM_s111
                                 + 1.00731 w_k1_AXP_s111
                                 + 1.15065 w_k1_DIS_s111
                                 + 1.12853 w_k1_FDX_s111
                                 + 1.33162 w_k1_IBM_s111
                                 + 1.20324 w_k1_INTC_s111
                                 + 1.26041 w_k1_KO_s111 + 1.36728 w_k1_MAT_s111
                                 + 1.09211 w_k1_MSFT_s111
                                 + 1.02061 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.09727 w_k1_ADM_s111
                                 + 1.53436 w_k1_AXP_s111
                                 + 1.42877 w_k1_DIS_s111
                                 + 1.33894 w_k1_FDX_s111
                                 + 1.50074 w_k1_IBM_s111
                                 + 1.38371 w_k1_INTC_s111
                                 + 1.39446 w_k1_KO_s111 + 2.07012 w_k1_MAT_s111
                                 + 1.33249 w_k1_MSFT_s111
                                 + 1.32883 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.11058 w_k1_ADM_s111
                                 + 0.554257 w_k1_AXP_s111
                                 + 0.792093 w_k1_DIS_s111
                                 + 0.822926 w_k1_FDX_s111
                                 + 0.78463 w_k1_IBM_s111
                                 + 0.938425 w_k1_INTC_s111
                                 + 0.975796 w_k1_KO_s111
                                 + 0.618338 w_k1_MAT_s111
                                 + 0.975018 w_k1_MSFT_s111
                                 + 0.92325 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.976622 w_k1_ADM_s112
                                 + 1.1415 w_k1_AXP_s112 + 1.59736 w_k1_DIS_s112
                                 + 1.43852 w_k1_FDX_s112
                                 + 1.15434 w_k1_IBM_s112
                                 + 1.4775 w_k1_INTC_s112 + 1.43713 w_k1_KO_s112
                                 + 1.54489 w_k1_MAT_s112
                                 + 1.51822 w_k1_MSFT_s112
                                 + 1.17037 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.913215 w_k1_ADM_s112
                                 + 0.709373 w_k1_AXP_s112
                                 + 0.860658 w_k1_DIS_s112
                                 + 0.89504 w_k1_FDX_s112
                                 + 0.981107 w_k1_IBM_s112
                                 + 0.926176 w_k1_INTC_s112
                                 + 0.952439 w_k1_KO_s112
                                 + 0.644964 w_k1_MAT_s112
                                 + 0.97265 w_k1_MSFT_s112
                                 + 0.922301 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.18759 w_k1_ADM_s112
                                 + 1.26345 w_k1_AXP_s112
                                 + 1.11571 w_k1_DIS_s112
                                 + 1.10278 w_k1_FDX_s112
                                 + 1.48838 w_k1_IBM_s112
                                 + 1.17633 w_k1_INTC_s112
                                 + 1.21079 w_k1_KO_s112 + 1.49064 w_k1_MAT_s112
                                 + 0.99903 w_k1_MSFT_s112
                                 + 1.00023 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.00659 w_k1_ADM_s112
                                 + 1.2225 w_k1_AXP_s112 + 1.11171 w_k1_DIS_s112
                                 + 1.18607 w_k1_FDX_s112
                                 + 0.757583 w_k1_IBM_s112
                                 + 1.19806 w_k1_INTC_s112
                                 + 0.890543 w_k1_KO_s112
                                 + 1.01552 w_k1_MAT_s112
                                 + 1.12686 w_k1_MSFT_s112
                                 + 0.954403 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.004 w_k1_ADM_s112
                                 + 0.936211 w_k1_AXP_s112
                                 + 0.703057 w_k1_DIS_s112
                                 + 0.998164 w_k1_FDX_s112
                                 + 0.677528 w_k1_IBM_s112
                                 + 0.963878 w_k1_INTC_s112
                                 + 0.815489 w_k1_KO_s112 + 0.8676 w_k1_MAT_s112
                                 + 0.949789 w_k1_MSFT_s112
                                 + 0.898254 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.932204 w_k1_ADM_s112
                                 + 1.00501 w_k1_AXP_s112
                                 + 0.842514 w_k1_DIS_s112
                                 + 0.932144 w_k1_FDX_s112
                                 + 1.39546 w_k1_IBM_s112
                                 + 1.08455 w_k1_INTC_s112
                                 + 1.19882 w_k1_KO_s112 + 1.27936 w_k1_MAT_s112
                                 + 1.01347 w_k1_MSFT_s112
                                 + 1.08156 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.10201 w_k1_ADM_s112
                                 + 1.23291 w_k1_AXP_s112
                                 + 1.34312 w_k1_DIS_s112
                                 + 1.18081 w_k1_FDX_s112
                                 + 1.12961 w_k1_IBM_s112
                                 + 1.11642 w_k1_INTC_s112
                                 + 1.47235 w_k1_KO_s112 + 1.278 w_k1_MAT_s112
                                 + 1.24485 w_k1_MSFT_s112
                                 + 1.17556 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.854652 w_k1_ADM_s112
                                 + 0.638877 w_k1_AXP_s112
                                 + 0.895904 w_k1_DIS_s112
                                 + 0.882639 w_k1_FDX_s112
                                 + 0.655765 w_k1_IBM_s112
                                 + 1.07798 w_k1_INTC_s112
                                 + 0.847915 w_k1_KO_s112
                                 + 0.60305 w_k1_MAT_s112
                                 + 1.01069 w_k1_MSFT_s112
                                 + 0.911084 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.911309 w_k1_ADM_s113
                                 + 0.884292 w_k1_AXP_s113
                                 + 0.933897 w_k1_DIS_s113
                                 + 0.833344 w_k1_FDX_s113
                                 + 0.931846 w_k1_IBM_s113
                                 + 0.740417 w_k1_INTC_s113
                                 + 0.982642 w_k1_KO_s113 + 1.2372 w_k1_MAT_s113
                                 + 0.876659 w_k1_MSFT_s113
                                 + 1.0193 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.0186 w_k1_ADM_s113
                                 + 0.844624 w_k1_AXP_s113 + 1.009 w_k1_DIS_s113
                                 + 1.02601 w_k1_FDX_s113
                                 + 0.795038 w_k1_IBM_s113
                                 + 0.757784 w_k1_INTC_s113
                                 + 1.06482 w_k1_KO_s113
                                 + 0.972551 w_k1_MAT_s113
                                 + 0.946043 w_k1_MSFT_s113
                                 + 1.06471 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.3872 w_k1_ADM_s113
                                 + 1.08122 w_k1_AXP_s113
                                 + 1.11999 w_k1_DIS_s113
                                 + 1.02279 w_k1_FDX_s113
                                 + 1.09427 w_k1_IBM_s113
                                 + 0.886006 w_k1_INTC_s113
                                 + 1.2397 w_k1_KO_s113 + 1.40251 w_k1_MAT_s113
                                 + 1.05945 w_k1_MSFT_s113
                                 + 1.13906 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.856603 w_k1_ADM_s113
                                 + 0.652584 w_k1_AXP_s113
                                 + 0.824087 w_k1_DIS_s113
                                 + 0.868588 w_k1_FDX_s113
                                 + 0.717236 w_k1_IBM_s113
                                 + 0.69452 w_k1_INTC_s113
                                 + 0.801377 w_k1_KO_s113
                                 + 0.612209 w_k1_MAT_s113
                                 + 0.772295 w_k1_MSFT_s113
                                 + 0.962203 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.50907 w_k1_ADM_s113
                                 + 1.40589 w_k1_AXP_s113
                                 + 1.32819 w_k1_DIS_s113
                                 + 1.16414 w_k1_FDX_s113
                                 + 1.47988 w_k1_IBM_s113
                                 + 0.926981 w_k1_INTC_s113
                                 + 1.26723 w_k1_KO_s113 + 1.65571 w_k1_MAT_s113
                                 + 0.943641 w_k1_MSFT_s113
                                 + 1.30929 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.48901 w_k1_ADM_s113
                                 + 1.07527 w_k1_AXP_s113
                                 + 1.21831 w_k1_DIS_s113
                                 + 1.18126 w_k1_FDX_s113
                                 + 0.909826 w_k1_IBM_s113
                                 + 0.896196 w_k1_INTC_s113
                                 + 1.13193 w_k1_KO_s113 + 1.01279 w_k1_MAT_s113
                                 + 1.15353 w_k1_MSFT_s113 + 1.1556 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.20378 w_k1_ADM_s113
                                 + 0.919528 w_k1_AXP_s113
                                 + 1.31044 w_k1_DIS_s113
                                 + 1.05637 w_k1_FDX_s113
                                 + 0.736934 w_k1_IBM_s113
                                 + 0.9407 w_k1_INTC_s113 + 1.10834 w_k1_KO_s113
                                 + 0.619131 w_k1_MAT_s113
                                 + 1.06495 w_k1_MSFT_s113
                                 + 1.09828 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.853305 w_k1_ADM_s113
                                 + 0.512682 w_k1_AXP_s113
                                 + 0.74103 w_k1_DIS_s113
                                 + 0.645328 w_k1_FDX_s113
                                 + 0.898576 w_k1_IBM_s113
                                 + 0.728021 w_k1_INTC_s113
                                 + 0.945924 w_k1_KO_s113
                                 + 0.707476 w_k1_MAT_s113
                                 + 0.821486 w_k1_MSFT_s113
                                 + 0.921356 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.913864 w_k1_ADM_s114
                                 + 1.02962 w_k1_AXP_s114 + 1.0211 w_k1_DIS_s114
                                 + 1.14205 w_k1_FDX_s114
                                 + 0.941107 w_k1_IBM_s114
                                 + 0.97567 w_k1_INTC_s114
                                 + 1.14302 w_k1_KO_s114 + 1.19622 w_k1_MAT_s114
                                 + 0.907586 w_k1_MSFT_s114
                                 + 1.02029 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.07645 w_k1_ADM_s114
                                 + 1.16201 w_k1_AXP_s114
                                 + 1.18581 w_k1_DIS_s114
                                 + 1.09944 w_k1_FDX_s114
                                 + 1.26477 w_k1_IBM_s114
                                 + 1.12065 w_k1_INTC_s114
                                 + 1.39482 w_k1_KO_s114 + 1.40055 w_k1_MAT_s114
                                 + 1.06964 w_k1_MSFT_s114
                                 + 0.964052 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.970668 w_k1_ADM_s114
                                 + 0.776101 w_k1_AXP_s114
                                 + 0.700333 w_k1_DIS_s114
                                 + 0.840538 w_k1_FDX_s114
                                 + 0.708986 w_k1_IBM_s114
                                 + 0.829103 w_k1_INTC_s114
                                 + 0.815119 w_k1_KO_s114
                                 + 0.574782 w_k1_MAT_s114
                                 + 0.772291 w_k1_MSFT_s114
                                 + 0.845499 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.997658 w_k1_ADM_s114
                                 + 1.15167 w_k1_AXP_s114
                                 + 0.750538 w_k1_DIS_s114
                                 + 0.882423 w_k1_FDX_s114
                                 + 1.11461 w_k1_IBM_s114
                                 + 1.04395 w_k1_INTC_s114
                                 + 0.938006 w_k1_KO_s114
                                 + 1.32241 w_k1_MAT_s114
                                 + 0.772384 w_k1_MSFT_s114
                                 + 0.901688 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.31897 w_k1_ADM_s114
                                 + 1.56976 w_k1_AXP_s114 + 1.8189 w_k1_DIS_s114
                                 + 1.36728 w_k1_FDX_s114
                                 + 1.56631 w_k1_IBM_s114
                                 + 1.35085 w_k1_INTC_s114
                                 + 1.83946 w_k1_KO_s114 + 1.74868 w_k1_MAT_s114
                                 + 1.46374 w_k1_MSFT_s114
                                 + 1.32039 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.727733 w_k1_ADM_s114
                                 + 0.708993 w_k1_AXP_s114
                                 + 0.996407 w_k1_DIS_s114
                                 + 0.882794 w_k1_FDX_s114
                                 + 1.07037 w_k1_IBM_s114
                                 + 0.989127 w_k1_INTC_s114
                                 + 1.22163 w_k1_KO_s114 + 1.02051 w_k1_MAT_s114
                                 + 1.00393 w_k1_MSFT_s114
                                 + 1.01409 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.612592 w_k1_ADM_s114
                                 + 0.688222 w_k1_AXP_s114
                                 + 0.697976 w_k1_DIS_s114
                                 + 0.753677 w_k1_FDX_s114
                                 + 1.02085 w_k1_IBM_s114
                                 + 0.904015 w_k1_INTC_s114
                                 + 0.90526 w_k1_KO_s114 + 0.87312 w_k1_MAT_s114
                                 + 0.754402 w_k1_MSFT_s114
                                 + 0.817081 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.986993 w_k1_ADM_s114
                                 + 1.12577 w_k1_AXP_s114 + 0.8676 w_k1_DIS_s114
                                 + 0.989846 w_k1_FDX_s114
                                 + 0.836823 w_k1_IBM_s114
                                 + 0.955816 w_k1_INTC_s114
                                 + 0.896535 w_k1_KO_s114
                                 + 0.702439 w_k1_MAT_s114
                                 + 0.857269 w_k1_MSFT_s114
                                 + 0.845061 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.822661 w_k1_ADM_s115
                                 + 0.772396 w_k1_AXP_s115
                                 + 0.79759 w_k1_DIS_s115
                                 + 0.815347 w_k1_FDX_s115
                                 + 1.18056 w_k1_IBM_s115
                                 + 1.16962 w_k1_INTC_s115
                                 + 0.92622 w_k1_KO_s115
                                 + 0.935497 w_k1_MAT_s115
                                 + 0.893481 w_k1_MSFT_s115
                                 + 0.751189 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.952501 w_k1_ADM_s115
                                 + 0.956345 w_k1_AXP_s115
                                 + 1.04001 w_k1_DIS_s115 + 1.0495 w_k1_FDX_s115
                                 + 0.859473 w_k1_IBM_s115
                                 + 1.20803 w_k1_INTC_s115
                                 + 0.978678 w_k1_KO_s115
                                 + 0.933313 w_k1_MAT_s115
                                 + 1.17284 w_k1_MSFT_s115
                                 + 1.01898 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1413 w_k1_ADM_s115
                                 + 1.14224 w_k1_AXP_s115
                                 + 0.941529 w_k1_DIS_s115
                                 + 0.955268 w_k1_FDX_s115
                                 + 1.21088 w_k1_IBM_s115
                                 + 1.13273 w_k1_INTC_s115
                                 + 1.09179 w_k1_KO_s115 + 1.49619 w_k1_MAT_s115
                                 + 1.11493 w_k1_MSFT_s115
                                 + 0.910196 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.875413 w_k1_ADM_s115
                                 + 0.75527 w_k1_AXP_s115
                                 + 0.714537 w_k1_DIS_s115
                                 + 0.782279 w_k1_FDX_s115
                                 + 0.742768 w_k1_IBM_s115
                                 + 0.755906 w_k1_INTC_s115
                                 + 0.702576 w_k1_KO_s115
                                 + 0.907461 w_k1_MAT_s115
                                 + 0.747638 w_k1_MSFT_s115
                                 + 0.753362 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.69005 w_k1_ADM_s115
                                 + 1.50623 w_k1_AXP_s115
                                 + 1.50214 w_k1_DIS_s115
                                 + 1.33219 w_k1_FDX_s115
                                 + 1.09305 w_k1_IBM_s115
                                 + 1.27463 w_k1_INTC_s115
                                 + 1.23502 w_k1_KO_s115 + 1.28493 w_k1_MAT_s115
                                 + 1.36756 w_k1_MSFT_s115 + 1.1703 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.824657 w_k1_ADM_s115
                                 + 0.866139 w_k1_AXP_s115
                                 + 0.64373 w_k1_DIS_s115
                                 + 0.792503 w_k1_FDX_s115
                                 + 0.999633 w_k1_IBM_s115
                                 + 0.894313 w_k1_INTC_s115
                                 + 0.845172 w_k1_KO_s115
                                 + 1.25982 w_k1_MAT_s115
                                 + 0.928922 w_k1_MSFT_s115
                                 + 0.834203 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.996198 w_k1_ADM_s115
                                 + 1.29087 w_k1_AXP_s115
                                 + 1.39045 w_k1_DIS_s115
                                 + 1.05006 w_k1_FDX_s115
                                 + 1.62781 w_k1_IBM_s115
                                 + 0.948717 w_k1_INTC_s115
                                 + 1.21723 w_k1_KO_s115 + 2.0891 w_k1_MAT_s115
                                 + 1.02783 w_k1_MSFT_s115
                                 + 1.13555 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.860944 w_k1_ADM_s115
                                 + 1.39989 w_k1_AXP_s115
                                 + 0.864943 w_k1_DIS_s115
                                 + 1.0779 w_k1_FDX_s115
                                 + 0.814146 w_k1_IBM_s115
                                 + 1.29485 w_k1_INTC_s115
                                 + 0.832054 w_k1_KO_s115
                                 + 1.03139 w_k1_MAT_s115
                                 + 1.02882 w_k1_MSFT_s115
                                 + 0.919704 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.957072 w_k1_ADM_s116
                                 + 0.79817 w_k1_AXP_s116
                                 + 0.764078 w_k1_DIS_s116
                                 + 0.841857 w_k1_FDX_s116
                                 + 0.631309 w_k1_IBM_s116
                                 + 0.919849 w_k1_INTC_s116
                                 + 0.703367 w_k1_KO_s116
                                 + 0.412983 w_k1_MAT_s116
                                 + 0.844625 w_k1_MSFT_s116
                                 + 0.879506 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.903612 w_k1_ADM_s116
                                 + 0.782663 w_k1_AXP_s116
                                 + 0.825777 w_k1_DIS_s116
                                 + 0.847133 w_k1_FDX_s116
                                 + 1.09951 w_k1_IBM_s116
                                 + 0.867127 w_k1_INTC_s116
                                 + 0.890281 w_k1_KO_s116
                                 + 1.11881 w_k1_MAT_s116
                                 + 0.97416 w_k1_MSFT_s116
                                 + 0.894034 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.07472 w_k1_ADM_s116
                                 + 1.27133 w_k1_AXP_s116 + 1.0563 w_k1_DIS_s116
                                 + 1.149 w_k1_FDX_s116 + 0.767644 w_k1_IBM_s116
                                 + 1.29398 w_k1_INTC_s116
                                 + 0.893091 w_k1_KO_s116
                                 + 0.866805 w_k1_MAT_s116
                                 + 1.06279 w_k1_MSFT_s116
                                 + 1.03246 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.708441 w_k1_ADM_s116
                                 + 0.906534 w_k1_AXP_s116
                                 + 0.897728 w_k1_DIS_s116
                                 + 0.904432 w_k1_FDX_s116
                                 + 0.826051 w_k1_IBM_s116
                                 + 0.92221 w_k1_INTC_s116
                                 + 0.805246 w_k1_KO_s116
                                 + 0.572526 w_k1_MAT_s116
                                 + 0.870606 w_k1_MSFT_s116
                                 + 0.88956 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.17331 w_k1_ADM_s116
                                 + 1.37462 w_k1_AXP_s116 + 1.4898 w_k1_DIS_s116
                                 + 1.3536 w_k1_FDX_s116 + 1.05712 w_k1_IBM_s116
                                 + 0.941928 w_k1_INTC_s116
                                 + 0.965309 w_k1_KO_s116
                                 + 1.00179 w_k1_MAT_s116
                                 + 1.11957 w_k1_MSFT_s116 + 1.103 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.790097 w_k1_ADM_s116
                                 + 1.54108 w_k1_AXP_s116
                                 + 1.53715 w_k1_DIS_s116
                                 + 1.24447 w_k1_FDX_s116
                                 + 1.62915 w_k1_IBM_s116
                                 + 0.988534 w_k1_INTC_s116
                                 + 1.47094 w_k1_KO_s116 + 1.49977 w_k1_MAT_s116
                                 + 1.06352 w_k1_MSFT_s116
                                 + 1.21748 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.956675 w_k1_ADM_s116
                                 + 1.0097 w_k1_AXP_s116 + 1.16492 w_k1_DIS_s116
                                 + 1.02794 w_k1_FDX_s116
                                 + 0.996278 w_k1_IBM_s116
                                 + 1.07961 w_k1_INTC_s116
                                 + 1.02039 w_k1_KO_s116 + 1.00918 w_k1_MAT_s116
                                 + 1.05244 w_k1_MSFT_s116
                                 + 0.913933 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.492092 w_k1_ADM_s116
                                 + 0.858069 w_k1_AXP_s116
                                 + 0.677625 w_k1_DIS_s116
                                 + 0.785099 w_k1_FDX_s116
                                 + 0.954504 w_k1_IBM_s116
                                 + 0.810804 w_k1_INTC_s116
                                 + 0.758629 w_k1_KO_s116
                                 + 0.830332 w_k1_MAT_s116
                                 + 0.693915 w_k1_MSFT_s116
                                 + 0.791153 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10582 w_k1_ADM_s117
                                 + 1.05389 w_k1_AXP_s117
                                 + 1.22606 w_k1_DIS_s117
                                 + 1.15639 w_k1_FDX_s117
                                 + 0.863248 w_k1_IBM_s117
                                 + 1.19008 w_k1_INTC_s117
                                 + 1.11337 w_k1_KO_s117 + 0.88895 w_k1_MAT_s117
                                 + 1.13321 w_k1_MSFT_s117
                                 + 1.05968 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.975651 w_k1_ADM_s117
                                 + 0.97571 w_k1_AXP_s117
                                 + 1.05012 w_k1_DIS_s117
                                 + 0.954101 w_k1_FDX_s117
                                 + 1.31809 w_k1_IBM_s117
                                 + 0.912523 w_k1_INTC_s117
                                 + 1.28519 w_k1_KO_s117 + 1.4205 w_k1_MAT_s117
                                 + 1.22071 w_k1_MSFT_s117
                                 + 1.04688 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.761899 w_k1_ADM_s117
                                 + 0.658475 w_k1_AXP_s117
                                 + 0.555099 w_k1_DIS_s117
                                 + 0.734291 w_k1_FDX_s117
                                 + 0.767404 w_k1_IBM_s117
                                 + 0.704536 w_k1_INTC_s117
                                 + 0.787281 w_k1_KO_s117
                                 + 0.993001 w_k1_MAT_s117
                                 + 0.906442 w_k1_MSFT_s117
                                 + 0.890327 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.790251 w_k1_ADM_s117
                                 + 0.691514 w_k1_AXP_s117
                                 + 0.969573 w_k1_DIS_s117
                                 + 0.929226 w_k1_FDX_s117
                                 + 0.884281 w_k1_IBM_s117
                                 + 0.861652 w_k1_INTC_s117
                                 + 1.02137 w_k1_KO_s117
                                 + 0.911712 w_k1_MAT_s117
                                 + 1.07444 w_k1_MSFT_s117
                                 + 0.874717 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.23557 w_k1_ADM_s117
                                 + 1.53022 w_k1_AXP_s117
                                 + 1.73657 w_k1_DIS_s117 + 1.4321 w_k1_FDX_s117
                                 + 1.33405 w_k1_IBM_s117
                                 + 1.17414 w_k1_INTC_s117 + 1.4817 w_k1_KO_s117
                                 + 1.47867 w_k1_MAT_s117
                                 + 1.32283 w_k1_MSFT_s117
                                 + 1.11995 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.679841 w_k1_ADM_s117
                                 + 0.783565 w_k1_AXP_s117
                                 + 0.844349 w_k1_DIS_s117
                                 + 0.912923 w_k1_FDX_s117
                                 + 0.685349 w_k1_IBM_s117
                                 + 0.963923 w_k1_INTC_s117
                                 + 0.946525 w_k1_KO_s117
                                 + 1.10952 w_k1_MAT_s117
                                 + 1.10569 w_k1_MSFT_s117
                                 + 0.936906 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.761235 w_k1_ADM_s117
                                 + 0.816689 w_k1_AXP_s117
                                 + 0.739944 w_k1_DIS_s117
                                 + 0.852294 w_k1_FDX_s117
                                 + 0.98191 w_k1_IBM_s117
                                 + 0.893292 w_k1_INTC_s117
                                 + 0.920211 w_k1_KO_s117
                                 + 0.992514 w_k1_MAT_s117
                                 + 0.867104 w_k1_MSFT_s117
                                 + 0.943731 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.71676 w_k1_ADM_s117
                                 + 1.42626 w_k1_AXP_s117
                                 + 1.14018 w_k1_DIS_s117
                                 + 1.18591 w_k1_FDX_s117 + 1.0523 w_k1_IBM_s117
                                 + 1.01916 w_k1_INTC_s117
                                 + 1.18522 w_k1_KO_s117 + 1.65662 w_k1_MAT_s117
                                 + 1.09061 w_k1_MSFT_s117
                                 + 0.96511 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0805111448509454 w_k1_ADM_s000
                                 + 0.0409198759541548 w_k1_ADM_s001
                                 + 0.0330816723358001 w_k1_ADM_s002
                                 + 0.137998382404334 w_k1_ADM_s003
                                 + 0.1006008938893 w_k1_ADM_s004
                                 + 0.04377030334391 w_k1_ADM_s005
                                 + 0.0380144584877031 w_k1_ADM_s006
                                 + 0.0481046466709048 w_k1_ADM_s007
                                 + 0.0142865468826712 w_k1_ADM_s010
                                 + 0.0119525783937233 w_k1_ADM_s011
                                 + 0.0335140960328322 w_k1_ADM_s012
                                 + 0.0211663109588112 w_k1_ADM_s013
                                 + 0.0224770853363487 w_k1_ADM_s014
                                 + 0.0318498908831659 w_k1_ADM_s015
                                 + 0.0113584433577767 w_k1_ADM_s016
                                 + 0.0105901387776117 w_k1_ADM_s017
                                 + 0.0214304378081925 w_k1_ADM_s100
                                 + 0.0308493739063443 w_k1_ADM_s101
                                 + 0.0170168757773295 w_k1_ADM_s102
                                 + 0.00818773794963588 w_k1_ADM_s103
                                 + 0.00843227463263425 w_k1_ADM_s104
                                 + 0.00872345097846227 w_k1_ADM_s105
                                 + 0.0182867651047951 w_k1_ADM_s106
                                 + 0.00995414617087183 w_k1_ADM_s107
                                 + 0.0371394956241801 w_k1_ADM_s110
                                 + 0.034994542343599 w_k1_ADM_s111
                                 + 0.020184474706583 w_k1_ADM_s112
                                 + 0.014123102007051 w_k1_ADM_s113
                                 + 0.0184462091943391 w_k1_ADM_s114
                                 + 0.0319920625841695 w_k1_ADM_s115
                                 + 0.0192125793242074 w_k1_ADM_s116
                                 + 0.0305810429716841 w_k1_ADM_s117
                                 + 0.0824458390658142 w_k1_AXP_s000
                                 + 0.0447146595020684 w_k1_AXP_s001
                                 + 0.0414900858894832 w_k1_AXP_s002
                                 + 0.159087094086656 w_k1_AXP_s003
                                 + 0.0918009903548093 w_k1_AXP_s004
                                 + 0.0390415538597728 w_k1_AXP_s005
                                 + 0.0380353736393923 w_k1_AXP_s006
                                 + 0.0493983571248773 w_k1_AXP_s007
                                 + 0.0148176909141437 w_k1_AXP_s010
                                 + 0.0101559032169605 w_k1_AXP_s011
                                 + 0.0371594182176429 w_k1_AXP_s012
                                 + 0.0197272911021821 w_k1_AXP_s013
                                 + 0.0229683045609175 w_k1_AXP_s014
                                 + 0.0323485316178734 w_k1_AXP_s015
                                 + 0.00982660068794611 w_k1_AXP_s016
                                 + 0.0135014102429567 w_k1_AXP_s017
                                 + 0.022323404222459 w_k1_AXP_s100
                                 + 0.0307161687213343 w_k1_AXP_s101
                                 + 0.0156613065320014 w_k1_AXP_s102
                                 + 0.0067063757468769 w_k1_AXP_s103
                                 + 0.00874790257211235 w_k1_AXP_s104
                                 + 0.00940262062383893 w_k1_AXP_s105
                                 + 0.0195382726748994 w_k1_AXP_s106
                                 + 0.00998776983380676 w_k1_AXP_s107
                                 + 0.0377278211285034 w_k1_AXP_s110
                                 + 0.0309066538767706 w_k1_AXP_s111
                                 + 0.0197769237226174 w_k1_AXP_s112
                                 + 0.011387457071256 w_k1_AXP_s113
                                 + 0.0195033475627789 w_k1_AXP_s114
                                 + 0.0323147798949713 w_k1_AXP_s115
                                 + 0.0224784783051109 w_k1_AXP_s116
                                 + 0.0336741992267242 w_k1_AXP_s117
                                 + 0.0773989305387751 w_k1_DIS_s000
                                 + 0.0393915847887656 w_k1_DIS_s001
                                 + 0.0383904538885973 w_k1_DIS_s002
                                 + 0.148379594345072 w_k1_DIS_s003
                                 + 0.0918738641931904 w_k1_DIS_s004
                                 + 0.0397033227068849 w_k1_DIS_s005
                                 + 0.0363093254785703 w_k1_DIS_s006
                                 + 0.052620559016629 w_k1_DIS_s007
                                 + 0.0152556057334243 w_k1_DIS_s010
                                 + 0.00995005216010049 w_k1_DIS_s011
                                 + 0.0376136432146377 w_k1_DIS_s012
                                 + 0.0201972576340838 w_k1_DIS_s013
                                 + 0.0247907561533964 w_k1_DIS_s014
                                 + 0.0339425246585005 w_k1_DIS_s015
                                 + 0.0109223884663328 w_k1_DIS_s016
                                 + 0.011649572151085 w_k1_DIS_s017
                                 + 0.0221886994967417 w_k1_DIS_s100
                                 + 0.0300024976582964 w_k1_DIS_s101
                                 + 0.0154330627294651 w_k1_DIS_s102
                                 + 0.00799391235333352 w_k1_DIS_s103
                                 + 0.00847442039887146 w_k1_DIS_s104
                                 + 0.00872803864687868 w_k1_DIS_s105
                                 + 0.0202854391269475 w_k1_DIS_s106
                                 + 0.0100962932767402 w_k1_DIS_s107
                                 + 0.0349127887662821 w_k1_DIS_s110
                                 + 0.0306367638010742 w_k1_DIS_s111
                                 + 0.0202855781372183 w_k1_DIS_s112
                                 + 0.0124737407186562 w_k1_DIS_s113
                                 + 0.0188663224146879 w_k1_DIS_s114
                                 + 0.0300013029329516 w_k1_DIS_s115
                                 + 0.0227702430122471 w_k1_DIS_s116
                                 + 0.0351914262421974 w_k1_DIS_s117
                                 + 0.083388244766136 w_k1_FDX_s000
                                 + 0.0401289781140652 w_k1_FDX_s001
                                 + 0.0383839641708409 w_k1_FDX_s002
                                 + 0.145306646647799 w_k1_FDX_s003
                                 + 0.0991427165356671 w_k1_FDX_s004
                                 + 0.0394537074008555 w_k1_FDX_s005
                                 + 0.0400011777987809 w_k1_FDX_s006
                                 + 0.0500288483749889 w_k1_FDX_s007
                                 + 0.0154765458712044 w_k1_FDX_s010
                                 + 0.00964755776908571 w_k1_FDX_s011
                                 + 0.0372137146880005 w_k1_FDX_s012
                                 + 0.0195575135046357 w_k1_FDX_s013
                                 + 0.0236817762752101 w_k1_FDX_s014
                                 + 0.0314107024880907 w_k1_FDX_s015
                                 + 0.0105793106746575 w_k1_FDX_s016
                                 + 0.0115004176613794 w_k1_FDX_s017
                                 + 0.0217895052764944 w_k1_FDX_s100
                                 + 0.0285706019020799 w_k1_FDX_s101
                                 + 0.0151550283337742 w_k1_FDX_s102
                                 + 0.00745469695494803 w_k1_FDX_s103
                                 + 0.00836278680427594 w_k1_FDX_s104
                                 + 0.00852585589554217 w_k1_FDX_s105
                                 + 0.0197578535427549 w_k1_FDX_s106
                                 + 0.010260562272158 w_k1_FDX_s107
                                 + 0.0369580283804763 w_k1_FDX_s110
                                 + 0.0329938308703609 w_k1_FDX_s111
                                 + 0.0210718252409306 w_k1_FDX_s112
                                 + 0.0114733268598721 w_k1_FDX_s113
                                 + 0.0192031776469393 w_k1_FDX_s114
                                 + 0.0299024763022663 w_k1_FDX_s115
                                 + 0.0218106517440058 w_k1_FDX_s116
                                 + 0.0343026464959695 w_k1_FDX_s117
                                 + 0.0843212372189104 w_k1_IBM_s000
                                 + 0.0392614430088611 w_k1_IBM_s001
                                 + 0.0378898966455557 w_k1_IBM_s002
                                 + 0.148821860937743 w_k1_IBM_s003
                                 + 0.0977264516837759 w_k1_IBM_s004
                                 + 0.040826880243868 w_k1_IBM_s005
                                 + 0.0410480634705292 w_k1_IBM_s006
                                 + 0.0519451070181537 w_k1_IBM_s007
                                 + 0.0152412760228702 w_k1_IBM_s010
                                 + 0.0101562591550176 w_k1_IBM_s011
                                 + 0.0385924572283771 w_k1_IBM_s012
                                 + 0.0197497053396676 w_k1_IBM_s013
                                 + 0.0237788325859117 w_k1_IBM_s014
                                 + 0.0328585595570878 w_k1_IBM_s015
                                 + 0.0100639655893613 w_k1_IBM_s016
                                 + 0.0121332364617792 w_k1_IBM_s017
                                 + 0.0208724867921213 w_k1_IBM_s100
                                 + 0.0286712606652831 w_k1_IBM_s101
                                 + 0.0155813304705148 w_k1_IBM_s102
                                 + 0.00728670982568728 w_k1_IBM_s103
                                 + 0.00852731147224738 w_k1_IBM_s104
                                 + 0.00882816332954177 w_k1_IBM_s105
                                 + 0.0207984216524645 w_k1_IBM_s106
                                 + 0.0105929565286531 w_k1_IBM_s107
                                 + 0.0356648235507162 w_k1_IBM_s110
                                 + 0.0323852617678203 w_k1_IBM_s111
                                 + 0.0212295898581267 w_k1_IBM_s112
                                 + 0.0119791207035501 w_k1_IBM_s113
                                 + 0.0197050854170426 w_k1_IBM_s114
                                 + 0.0304374875285826 w_k1_IBM_s115
                                 + 0.0209902390894167 w_k1_IBM_s116
                                 + 0.0343351809858412 w_k1_IBM_s117
                                 + 0.0788007305727334 w_k1_INTC_s000
                                 + 0.0390779874704065 w_k1_INTC_s001
                                 + 0.0410217970044594 w_k1_INTC_s002
                                 + 0.16167876879027 w_k1_INTC_s003
                                 + 0.101836050770597 w_k1_INTC_s004
                                 + 0.0393179791936843 w_k1_INTC_s005
                                 + 0.0383459773589039 w_k1_INTC_s006
                                 + 0.0540008329838509 w_k1_INTC_s007
                                 + 0.0128773388062871 w_k1_INTC_s010
                                 + 0.00891104088627226 w_k1_INTC_s011
                                 + 0.0387058064912469 w_k1_INTC_s012
                                 + 0.0194187774512229 w_k1_INTC_s013
                                 + 0.0255195125517473 w_k1_INTC_s014
                                 + 0.0328549730012955 w_k1_INTC_s015
                                 + 0.00950787439737409 w_k1_INTC_s016
                                 + 0.0114361998217242 w_k1_INTC_s017
                                 + 0.0230700847881984 w_k1_INTC_s100
                                 + 0.0269876668076118 w_k1_INTC_s101
                                 + 0.0170524949902573 w_k1_INTC_s102
                                 + 0.00748703848838019 w_k1_INTC_s103
                                 + 0.00803685549380222 w_k1_INTC_s104
                                 + 0.00817011488830924 w_k1_INTC_s105
                                 + 0.0222068333810153 w_k1_INTC_s106
                                 + 0.0113620168191695 w_k1_INTC_s107
                                 + 0.0370012568072118 w_k1_INTC_s110
                                 + 0.0368502393553039 w_k1_INTC_s111
                                 + 0.0219902116479291 w_k1_INTC_s112
                                 + 0.00967922381890111 w_k1_INTC_s113
                                 + 0.0191945093884252 w_k1_INTC_s114
                                 + 0.0328433985287776 w_k1_INTC_s115
                                 + 0.0207596632393402 w_k1_INTC_s116
                                 + 0.0328176185766148 w_k1_INTC_s117
                                 + 0.0784620498400639 w_k1_KO_s000
                                 + 0.0341320909228884 w_k1_KO_s001
                                 + 0.0401419751281611 w_k1_KO_s002
                                 + 0.142199276598643 w_k1_KO_s003
                                 + 0.0968146032950253 w_k1_KO_s004
                                 + 0.0407237890716489 w_k1_KO_s005
                                 + 0.0378585390566899 w_k1_KO_s006
                                 + 0.0570545184670268 w_k1_KO_s007
                                 + 0.0130997305768738 w_k1_KO_s010
                                 + 0.00956283637133382 w_k1_KO_s011
                                 + 0.0388109062980896 w_k1_KO_s012
                                 + 0.0200772084933097 w_k1_KO_s013
                                 + 0.0236711612242942 w_k1_KO_s014
                                 + 0.0307818519084214 w_k1_KO_s015
                                 + 0.010862935270933 w_k1_KO_s016
                                 + 0.0120662713428785 w_k1_KO_s017
                                 + 0.020837230293612 w_k1_KO_s100
                                 + 0.0296767927980594 w_k1_KO_s101
                                 + 0.0162223398876814 w_k1_KO_s102
                                 + 0.00713480365180896 w_k1_KO_s103
                                 + 0.00749609263128933 w_k1_KO_s104
                                 + 0.00848804345048454 w_k1_KO_s105
                                 + 0.0198248009179772 w_k1_KO_s106
                                 + 0.0096504109128894 w_k1_KO_s107
                                 + 0.0360436773760747 w_k1_KO_s110
                                 + 0.0359278184339151 w_k1_KO_s111
                                 + 0.0218004988748434 w_k1_KO_s112
                                 + 0.0127652613104297 w_k1_KO_s113
                                 + 0.0215926764039111 w_k1_KO_s114
                                 + 0.029455387270078 w_k1_KO_s115
                                 + 0.019932764061619 w_k1_KO_s116
                                 + 0.0374280717476633 w_k1_KO_s117
                                 + 0.090840401299518 w_k1_MAT_s000
                                 + 0.0365609827455314 w_k1_MAT_s001
                                 + 0.0342112986425779 w_k1_MAT_s002
                                 + 0.146632524824649 w_k1_MAT_s003
                                 + 0.114726771990697 w_k1_MAT_s004
                                 + 0.0377106270465229 w_k1_MAT_s005
                                 + 0.0451099533559037 w_k1_MAT_s006
                                 + 0.0460437897867969 w_k1_MAT_s007
                                 + 0.0169212561868193 w_k1_MAT_s010
                                 + 0.00798800296606475 w_k1_MAT_s011
                                 + 0.0453554755527983 w_k1_MAT_s012
                                 + 0.01952426908561 w_k1_MAT_s013
                                 + 0.0223557156784694 w_k1_MAT_s014
                                 + 0.0264529797128086 w_k1_MAT_s015
                                 + 0.00902479693129236 w_k1_MAT_s016
                                 + 0.0134144026798628 w_k1_MAT_s017
                                 + 0.0244908468245886 w_k1_MAT_s100
                                 + 0.0255280113844873 w_k1_MAT_s101
                                 + 0.012383656048963 w_k1_MAT_s102
                                 + 0.00702006006272394 w_k1_MAT_s103
                                 + 0.00950113855298947 w_k1_MAT_s104
                                 + 0.0120640406917436 w_k1_MAT_s105
                                 + 0.0194419958142268 w_k1_MAT_s106
                                 + 0.0080438369473224 w_k1_MAT_s107
                                 + 0.0413803079298332 w_k1_MAT_s110
                                 + 0.0369793568301236 w_k1_MAT_s111
                                 + 0.0205531842961194 w_k1_MAT_s112
                                 + 0.013200303508971 w_k1_MAT_s113
                                 + 0.0207128749991408 w_k1_MAT_s114
                                 + 0.0354141513698978 w_k1_MAT_s115
                                 + 0.019122673132163 w_k1_MAT_s116
                                 + 0.0398731076977056 w_k1_MAT_s117
                                 + 0.082374942823202 w_k1_MSFT_s000
                                 + 0.038720378094234 w_k1_MSFT_s001
                                 + 0.0386340650999381 w_k1_MSFT_s002
                                 + 0.154522492180313 w_k1_MSFT_s003
                                 + 0.0917046171852749 w_k1_MSFT_s004
                                 + 0.0407467614492542 w_k1_MSFT_s005
                                 + 0.0380381587879308 w_k1_MSFT_s006
                                 + 0.0562314904758094 w_k1_MSFT_s007
                                 + 0.0130383591890539 w_k1_MSFT_s010
                                 + 0.00929335512473886 w_k1_MSFT_s011
                                 + 0.0392311085319198 w_k1_MSFT_s012
                                 + 0.0198910163969161 w_k1_MSFT_s013
                                 + 0.0249971382593966 w_k1_MSFT_s014
                                 + 0.0312860535250102 w_k1_MSFT_s015
                                 + 0.010005059491226 w_k1_MSFT_s016
                                 + 0.0128121701900977 w_k1_MSFT_s017
                                 + 0.0233596581064289 w_k1_MSFT_s100
                                 + 0.0302685095849169 w_k1_MSFT_s101
                                 + 0.0140394498945976 w_k1_MSFT_s102
                                 + 0.00716595345591968 w_k1_MSFT_s103
                                 + 0.00764086163726444 w_k1_MSFT_s104
                                 + 0.00901933440110981 w_k1_MSFT_s105
                                 + 0.0192107950861125 w_k1_MSFT_s106
                                 + 0.00965007743824514 w_k1_MSFT_s107
                                 + 0.0363116731135346 w_k1_MSFT_s110
                                 + 0.0341044895118843 w_k1_MSFT_s111
                                 + 0.0217064178164699 w_k1_MSFT_s112
                                 + 0.0110483884745872 w_k1_MSFT_s113
                                 + 0.0178585961065727 w_k1_MSFT_s114
                                 + 0.0319644937742076 w_k1_MSFT_s115
                                 + 0.0207089767222441 w_k1_MSFT_s116
                                 + 0.0369204674127313 w_k1_MSFT_s117
                                 + 0.0818205785392567 w_k1_PG_s000
                                 + 0.0375358970291329 w_k1_PG_s001
                                 + 0.0369234246939545 w_k1_PG_s002
                                 + 0.141010682721192 w_k1_PG_s003
                                 + 0.0934831254826822 w_k1_PG_s004
                                 + 0.0422536096493809 w_k1_PG_s005
                                 + 0.03897020624523 w_k1_PG_s006
                                 + 0.0487214957921934 w_k1_PG_s007
                                 + 0.0144972275303896 w_k1_PG_s010
                                 + 0.00946799795640815 w_k1_PG_s011
                                 + 0.0349348827269286 w_k1_PG_s012
                                 + 0.0206102424669617 w_k1_PG_s013
                                 + 0.0231285371914947 w_k1_PG_s014
                                 + 0.0306565663124884 w_k1_PG_s015
                                 + 0.00983533320521007 w_k1_PG_s016
                                 + 0.012267050463912 w_k1_PG_s017
                                 + 0.0213779402898681 w_k1_PG_s100
                                 + 0.0277926543449624 w_k1_PG_s101
                                 + 0.0155476734804389 w_k1_PG_s102
                                 + 0.00692828417600072 w_k1_PG_s103
                                 + 0.00777087095792506 w_k1_PG_s104
                                 + 0.00860426986772025 w_k1_PG_s105
                                 + 0.0193729084266645 w_k1_PG_s106
                                 + 0.00940707711673081 w_k1_PG_s107
                                 + 0.0351130193574731 w_k1_PG_s110
                                 + 0.0315403268380644 w_k1_PG_s111
                                 + 0.0203442994525406 w_k1_PG_s112
                                 + 0.0128773017181939 w_k1_PG_s113
                                 + 0.0181132240073046 w_k1_PG_s114
                                 + 0.0282799168049493 w_k1_PG_s115
                                 + 0.0205115753944051 w_k1_PG_s116
                                 + 0.0334473107688997 w_k1_PG_s117 >= 1.1
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
