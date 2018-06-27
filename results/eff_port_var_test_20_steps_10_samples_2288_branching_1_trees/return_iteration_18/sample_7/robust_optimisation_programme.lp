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
 w_k1_ADM_s0_return_constraint:  1.17738 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.895588 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.956445 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06329 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.12632 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.951602 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.21471 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.842628 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.47116 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.2816 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.839461 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.968863 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.02903 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.33765 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.00697 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.86106 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.955647 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.932932 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.39681 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.703433 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.827596 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.16482 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.09872 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.46667 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.38115 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.994969 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.789813 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.801321 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.31279 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.81087 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.874215 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.840208 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.1644 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.992124 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.697602 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.860364 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.702568 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.18586 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26646 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.90491 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.960936 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28243 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.22976 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.745061 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.971928 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.599534 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.43597 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.58243 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.969824 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.938274 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.37448 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.14369 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.701362 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.850436 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.29389 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.04855 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.21598 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.675556 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.644753 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.13372 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.901378 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.38127 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.44012 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.0094 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.57878 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.720439 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.09755 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.758348 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.860162 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.01139 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.23377 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.01789 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.770115 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.36691 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.739423 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.34218 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.44932 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.921986 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.855522 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.17504 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.19617 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.690206 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.94095 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.607026 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.57498 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.64054 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.841061 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.992831 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.32142 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.17879 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.947915 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.772163 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.71624 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.931632 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.54523 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.642594 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.58047 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.33034 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.995622 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.63239 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.4794 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.06799 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.54095 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.742064 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.1904 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.71935 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.835634 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.98868 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.54473 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.1363 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.648915 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.19291 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.669737 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.43326 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24759 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.941766 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.945437 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07239 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.09908 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.758951 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.12537 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.737805 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.51158 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.37708 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.848798 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.996314 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.19912 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.06611 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.97236 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.803979 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.27533 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.02335 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.42846 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.717746 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.705707 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.23353 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.998876 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.28547 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.46501 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.914428 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.722936 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.940582 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.28363 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.786923 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.959828 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.942021 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.34035 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.1391 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.732361 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11262 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.891752 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.09634 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.2343 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.968824 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.720804 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.1961 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.10707 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.943165 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.730328 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.859859 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.979607 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.62151 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.32984 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.893041 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.3242 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.25051 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.752444 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.39223 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.78936 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.16541 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.896826 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.0056 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.656027 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.24156 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.956916 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.54843 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.975103 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.76301 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.537882 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.571229 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.678208 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.928065 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.782244 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.895736 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.1275 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.741003 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.98039 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.4378 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.519694 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 2.03219 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12185 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.981319 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.02312 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.11202 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.12787 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.75374 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.06464 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.671616 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.30919 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.23335 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.839971 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.867164 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.04883 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.961354 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.0119 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05269 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.37925 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.00774 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.34492 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.697991 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.722723 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.20787 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.937864 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07924 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.14146 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.934178 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.633338 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.912758 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.13423 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.795219 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.975381 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.13117 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.35798 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.05176 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.93879 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.32924 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.04573 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.01265 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.28029 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.958586 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.850455 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.13303 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.08077 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.892203 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.835061 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.696979 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.0209 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.3629 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.914661 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.934479 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.15141 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.19523 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.903699 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.01939 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.70912 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.00059 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.08341 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.813585 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.727505 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.27401 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.0076 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.57501 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.29847 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.38866 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.627326 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.799865 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.948567 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.801573 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.845705 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.885262 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.22944 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.899938 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.793544 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.10999 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.620423 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.51038 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.27865 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.989489 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.848885 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.2376 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.19101 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.80763 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.484893 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.772432 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.820621 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.51185 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.24717 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.97069 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.32593 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.06412 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.693357 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.2528 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 2.21489 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.3212 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.027 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.982459 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.56615 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.21412 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.925939 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.6176 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.11074 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.54977 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.487409 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.586283 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.607863 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.735735 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.713773 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.809608 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.06309 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.916629 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.851238 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.59827 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.377132 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.18924 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21676 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.961807 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.956753 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.00799 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.09305 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.814506 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.955301 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.720112 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.20025 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.18095 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.843503 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.985999 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08785 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.04118 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.08447 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.910394 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.44596 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.02627 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.43973 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.853255 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.773818 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2682 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.06966 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.27988 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.21509 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.964895 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.709344 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.968167 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.26028 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.819835 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.959638 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.05588 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.24767 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.11079 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.756677 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.06983 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.807739 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.06922 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16364 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.942068 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.914757 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12247 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11642 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.902544 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0019 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.788717 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.13694 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.21517 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.960833 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.971684 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.06736 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.11802 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.930699 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.942256 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.18641 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.936163 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.08805 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.776877 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.712057 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.15065 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.02136 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.23388 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.22653 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08879 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.741499 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.892112 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.03529 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.868671 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.894179 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.979563 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.14121 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.02582 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.821215 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.09602 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.756597 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.16913 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0710002962717023 z_k1_s0000
                                 - 0.0920003131405492 z_k1_s0001
                                 - 0.105000335545115 z_k1_s0002
                                 - 0.0650002376232715 z_k1_s0003
                                 - 0.305000573953453 z_k1_s0004
                                 - 0.153999956237623 z_k1_s0005
                                 - 0.251001176521542 z_k1_s0006
                                 - 0.213000763214666 z_k1_s0007
                                 - 0.334001131233139 z_k1_s0010
                                 - 0.125000032230082 z_k1_s0011
                                 - 0.19099990509386 z_k1_s0012
                                 - 0.194999795006742 z_k1_s0013
                                 - 0.266001125712784 z_k1_s0014
                                 - 0.337000579417459 z_k1_s0015
                                 - 0.12400015360193 z_k1_s0016
                                 - 0.304999583113039 z_k1_s0017
                                 - 0.0590000537312859 z_k1_s0020
                                 - 0.0550000189698213 z_k1_s0021
                                 - 0.185000077716681 z_k1_s0022
                                 - 0.101999815416608 z_k1_s0023
                                 - 0.0550000189698213 z_k1_s0024
                                 - 0.0430000676855331 z_k1_s0025
                                 - 0.0799999302287638 z_k1_s0026
                                 - 0.185999933406942 z_k1_s0027
                                 - 0.0990000420118328 z_k1_s0030
                                 - 0.0839999817070086 z_k1_s0031
                                 - 0.024000007487717 z_k1_s0032
                                 - 0.0360000112315755 z_k1_s0033
                                 - 0.098000126991513 z_k1_s0034
                                 - 0.0340000031909323 z_k1_s0035
                                 - 0.0380000192722187 z_k1_s0036
                                 - 0.031999995150289 z_k1_s0037
                                 - 0.100000086423408 z_k1_s0040
                                 - 0.15900048747364 z_k1_s0041
                                 - 0.154000711452744 z_k1_s0042
                                 - 0.239999903015811 z_k1_s0043
                                 - 0.271000036347196 z_k1_s0044
                                 - 0.279999633184808 z_k1_s0045
                                 - 0.138000210584412 z_k1_s0046
                                 - 0.1799995467614 z_k1_s0047
                                 - 0.167999654111297 z_k1_s0050
                                 - 0.250000053569359 z_k1_s0051
                                 - 0.144999783550674 z_k1_s0052
                                 - 0.210999811836959 z_k1_s0053
                                 - 0.276999132924518 z_k1_s0054
                                 - 0.282999714113138 z_k1_s0055
                                 - 0.136999774764467 z_k1_s0056
                                 - 0.296998712890832 z_k1_s0057
                                 - 0.0630003062746344 z_k1_s0060
                                 - 0.112000156376763 z_k1_s0061
                                 - 0.0480000527614563 z_k1_s0062
                                 - 0.137000079898257 z_k1_s0063
                                 - 0.033000048748513 z_k1_s0064
                                 - 0.0400000273345313 z_k1_s0065
                                 - 0.029000011085027 z_k1_s0066
                                 - 0.037000036511952 z_k1_s0067
                                 - 0.0879997238676191 z_k1_s0070
                                 - 0.0819999612176507 z_k1_s0071
                                 - 0.0960000484011935 z_k1_s0072
                                 - 0.0370000106421231 z_k1_s0073
                                 - 0.0450000146755559 z_k1_s0074
                                 - 0.0689998344757693 z_k1_s0075
                                 - 0.122000301884956 z_k1_s0076
                                 - 0.101999811051162 z_k1_s0077
                                 - 0.310000029652959 z_k1_s0100
                                 - 0.798000070136968 z_k1_s0101
                                 - 0.803000034474903 z_k1_s0102
                                 - 0.182999975168985 z_k1_s0103
                                 - 0.165000103552422 z_k1_s0104
                                 - 0.298000115241917 z_k1_s0105
                                 - 0.435998810868747 z_k1_s0106
                                 - 0.208000116958805 z_k1_s0107
                                 - 0.157000077261341 z_k1_s0110
                                 - 0.342998829687675 z_k1_s0111
                                 - 0.118999832013723 z_k1_s0112
                                 - 0.102999872751346 z_k1_s0113
                                 - 0.115999884651958 z_k1_s0114
                                 - 0.228999965941928 z_k1_s0115
                                 - 0.298999697714968 z_k1_s0116
                                 - 0.0739999015883564 z_k1_s0117
                                 - 0.0949999997094205 z_k1_s0120
                                 - 0.0730000473262078 z_k1_s0121
                                 - 0.13099984144555 z_k1_s0122
                                 - 0.293000454958735 z_k1_s0123
                                 - 0.121999807361484 z_k1_s0124
                                 - 0.0910000172942949 z_k1_s0125
                                 - 0.0600000308270165 z_k1_s0126
                                 - 0.117000443092856 z_k1_s0127
                                 - 0.404000212115385 z_k1_s0130
                                 - 0.151999781196259 z_k1_s0131
                                 - 0.124999800704842 z_k1_s0132
                                 - 0.172999847756148 z_k1_s0133
                                 - 0.235000000480637 z_k1_s0134
                                 - 0.373000227703026 z_k1_s0135
                                 - 0.114999882852373 z_k1_s0136
                                 - 0.262000164871825 z_k1_s0137
                                 - 0.143000459373281 z_k1_s0140
                                 - 0.265001357769994 z_k1_s0141
                                 - 0.434001016208829 z_k1_s0142
                                 - 0.260000638678179 z_k1_s0143
                                 - 0.391000885635065 z_k1_s0144
                                 - 0.161000453697025 z_k1_s0145
                                 - 0.262000493913773 z_k1_s0146
                                 - 0.24600165202902 z_k1_s0147
                                 - 0.103000215888492 z_k1_s0150
                                 - 0.360000191142818 z_k1_s0151
                                 - 0.123000325707788 z_k1_s0152
                                 - 0.211000422031714 z_k1_s0153
                                 - 0.123000325707788 z_k1_s0154
                                 - 0.300000903287565 z_k1_s0155
                                 - 0.123000325707788 z_k1_s0156
                                 - 0.145000312588676 z_k1_s0157
                                 - 0.189999608718942 z_k1_s0160
                                 - 0.219999546937723 z_k1_s0161
                                 - 0.105000052365769 z_k1_s0162
                                 - 0.349000087078592 z_k1_s0163
                                 - 0.136999449265726 z_k1_s0164
                                 - 0.119000023534533 z_k1_s0165
                                 - 0.141999438968856 z_k1_s0166
                                 - 0.0809999674907259 z_k1_s0167
                                 - 0.325999161742266 z_k1_s0170
                                 - 0.187999962077715 z_k1_s0171
                                 - 0.243000337232869 z_k1_s0172
                                 - 0.239999951588573 z_k1_s0173
                                 - 0.149999815243082 z_k1_s0174
                                 - 0.0879998277493677 z_k1_s0175
                                 - 0.132999741422337 z_k1_s0176
                                 - 0.176999578047133 z_k1_s0177
                                 - 1.46199822024621 z_k1_s1000
                                 - 1.37799854735251 z_k1_s1001
                                 - 0.817999206162607 z_k1_s1002
                                 - 2.60699713139409 z_k1_s1003
                                 - 2.5940001171079 z_k1_s1004
                                 - 0.963999507277049 z_k1_s1005
                                 - 3.81299526146526 z_k1_s1006
                                 - 1.58300285828847 z_k1_s1007
                                 - 1.63700458777304 z_k1_s1010
                                 - 0.688001412534378 z_k1_s1011
                                 - 1.3090021823396 z_k1_s1012
                                 - 0.484001265875435 z_k1_s1013
                                 - 0.806000850650216 z_k1_s1014
                                 - 0.478001282423753 z_k1_s1015
                                 - 0.7249985914964 z_k1_s1016
                                 - 0.966005138040037 z_k1_s1017
                                 - 0.471002089887541 z_k1_s1020
                                 - 0.717001995943329 z_k1_s1021
                                 - 0.430998975650496 z_k1_s1022
                                 - 0.77200046176865 z_k1_s1023
                                 - 0.428998185438577 z_k1_s1024
                                 - 0.295000231245798 z_k1_s1025
                                 - 0.838001158759303 z_k1_s1026
                                 - 0.27699988733924 z_k1_s1027
                                 - 0.382999901411354 z_k1_s1030
                                 - 0.627002004756242 z_k1_s1031
                                 - 0.469000171685416 z_k1_s1032
                                 - 0.548002316220713 z_k1_s1033
                                 - 1.03499759820582 z_k1_s1034
                                 - 0.326000076397117 z_k1_s1035
                                 - 1.22400151634734 z_k1_s1036
                                 - 0.299999880453806 z_k1_s1037
                                 - 0.149000054154373 z_k1_s1040
                                 - 0.635999991215899 z_k1_s1041
                                 - 0.352999185006348 z_k1_s1042
                                 - 0.463999067742102 z_k1_s1043
                                 - 0.406001813025233 z_k1_s1044
                                 - 0.344000094446998 z_k1_s1045
                                 - 0.482000753861993 z_k1_s1046
                                 - 0.671000933114893 z_k1_s1047
                                 - 0.386000002758158 z_k1_s1050
                                 - 0.36299993285136 z_k1_s1051
                                 - 0.368999927200972 z_k1_s1052
                                 - 0.481000004860295 z_k1_s1053
                                 - 0.696001542153825 z_k1_s1054
                                 - 0.728998489378225 z_k1_s1055
                                 - 1.14499846388444 z_k1_s1056
                                 - 1.32499829437279 z_k1_s1057
                                 - 1.12799779211259 z_k1_s1060
                                 - 0.669998012157988 z_k1_s1061
                                 - 0.536999710611873 z_k1_s1062
                                 - 0.509999419368569 z_k1_s1063
                                 - 1.08899626562894 z_k1_s1064
                                 - 0.769000185848982 z_k1_s1065
                                 - 0.312999690268702 z_k1_s1066
                                 - 0.512999451728936 z_k1_s1067
                                 - 1.68700438789735 z_k1_s1070
                                 - 1.67000207347169 z_k1_s1071
                                 - 1.43100538829736 z_k1_s1072
                                 - 0.441000373231527 z_k1_s1073
                                 - 0.922000305868943 z_k1_s1074
                                 - 1.0730015991549 z_k1_s1075
                                 - 0.699000715353528 z_k1_s1076
                                 - 1.17400571743637 z_k1_s1077
                                 - 0.546999478549286 z_k1_s1100
                                 - 0.113000037902194 z_k1_s1101
                                 - 0.254999122619623 z_k1_s1102
                                 - 0.160000000979192 z_k1_s1103
                                 - 0.422000472206552 z_k1_s1104
                                 - 0.228000109177891 z_k1_s1105
                                 - 0.168000134473204 z_k1_s1106
                                 - 0.160000000979192 z_k1_s1107
                                 - 0.581000011218596 z_k1_s1110
                                 - 0.250999464502046 z_k1_s1111
                                 - 0.141999798601786 z_k1_s1112
                                 - 0.166999788787973 z_k1_s1113
                                 - 0.245999570504636 z_k1_s1114
                                 - 0.491000358667804 z_k1_s1115
                                 - 0.399000540438392 z_k1_s1116
                                 - 0.323998228983717 z_k1_s1117
                                 - 0.274998370083423 z_k1_s1120
                                 - 0.306999466195416 z_k1_s1121
                                 - 0.189999621419432 z_k1_s1122
                                 - 0.260999295293464 z_k1_s1123
                                 - 0.585999686413615 z_k1_s1124
                                 - 0.370998917426158 z_k1_s1125
                                 - 0.26499922673297 z_k1_s1126
                                 - 0.486000029929348 z_k1_s1127
                                 - 0.197000347482527 z_k1_s1130
                                 - 0.776999373421272 z_k1_s1131
                                 - 0.737998672422674 z_k1_s1132
                                 - 0.482001155822482 z_k1_s1133
                                 - 0.361000419043431 z_k1_s1134
                                 - 0.193000064097876 z_k1_s1135
                                 - 0.91900118556939 z_k1_s1136
                                 - 0.457001446670161 z_k1_s1137
                                 - 0.465999486966513 z_k1_s1140
                                 - 0.551001094659345 z_k1_s1141
                                 - 0.401000649085889 z_k1_s1142
                                 - 0.217000238035904 z_k1_s1143
                                 - 0.335000317513448 z_k1_s1144
                                 - 0.867000136052402 z_k1_s1145
                                 - 0.268000186250698 z_k1_s1146
                                 - 0.283000230808043 z_k1_s1147
                                 - 0.306000503879229 z_k1_s1150
                                 - 0.506999520900757 z_k1_s1151
                                 - 0.390998951627088 z_k1_s1152
                                 - 0.520998906883525 z_k1_s1153
                                 - 0.574998279099043 z_k1_s1154
                                 - 0.401998034043054 z_k1_s1155
                                 - 0.163999638954927 z_k1_s1156
                                 - 0.179999633448027 z_k1_s1157
                                 - 0.17700068858846 z_k1_s1160
                                 - 0.325001521036245 z_k1_s1161
                                 - 0.632001808435522 z_k1_s1162
                                 - 0.197000685113414 z_k1_s1163
                                 - 0.433001350830441 z_k1_s1164
                                 - 0.270000836490092 z_k1_s1165
                                 - 0.206000797130057 z_k1_s1166
                                 - 0.284001515540044 z_k1_s1167
                                 - 0.182000394307865 z_k1_s1170
                                 - 0.131000113042392 z_k1_s1171
                                 - 0.12000008186554 z_k1_s1172
                                 - 0.164000039472869 z_k1_s1173
                                 - 0.145000091958357 z_k1_s1174
                                 - 0.322000183467514 z_k1_s1175
                                 - 0.416000662560719 z_k1_s1176
                                 - 0.191000404625282 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.07182 w_k1_ADM_s000
                                 + 1.766 w_k1_AXP_s000 + 1.52833 w_k1_DIS_s000
                                 + 1.30193 w_k1_FDX_s000
                                 + 1.43626 w_k1_IBM_s000
                                 + 1.25859 w_k1_INTC_s000
                                 + 1.21535 w_k1_KO_s000 + 1.81831 w_k1_MAT_s000
                                 + 1.28717 w_k1_MSFT_s000
                                 + 1.25299 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.15968 w_k1_ADM_s000
                                 + 1.21086 w_k1_AXP_s000
                                 + 1.34437 w_k1_DIS_s000
                                 + 1.21232 w_k1_FDX_s000
                                 + 1.28798 w_k1_IBM_s000
                                 + 1.18712 w_k1_INTC_s000
                                 + 1.41517 w_k1_KO_s000 + 2.02038 w_k1_MAT_s000
                                 + 1.18917 w_k1_MSFT_s000
                                 + 1.16632 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.17337 w_k1_ADM_s000
                                 + 1.60128 w_k1_AXP_s000
                                 + 1.71273 w_k1_DIS_s000
                                 + 1.42503 w_k1_FDX_s000
                                 + 1.23544 w_k1_IBM_s000
                                 + 1.32346 w_k1_INTC_s000
                                 + 1.35123 w_k1_KO_s000 + 1.91315 w_k1_MAT_s000
                                 + 1.43455 w_k1_MSFT_s000
                                 + 1.20406 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.2165 w_k1_ADM_s000
                                 + 1.17206 w_k1_AXP_s000
                                 + 0.860732 w_k1_DIS_s000
                                 + 1.17648 w_k1_FDX_s000
                                 + 0.814489 w_k1_IBM_s000
                                 + 0.894265 w_k1_INTC_s000
                                 + 0.785988 w_k1_KO_s000
                                 + 0.956647 w_k1_MAT_s000
                                 + 1.08246 w_k1_MSFT_s000
                                 + 0.953895 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.940068 w_k1_ADM_s000
                                 + 0.878604 w_k1_AXP_s000
                                 + 1.0289 w_k1_DIS_s000 + 1.04366 w_k1_FDX_s000
                                 + 0.924959 w_k1_IBM_s000
                                 + 0.904017 w_k1_INTC_s000
                                 + 1.05422 w_k1_KO_s000 + 1.42412 w_k1_MAT_s000
                                 + 1.12513 w_k1_MSFT_s000
                                 + 0.926389 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.882044 w_k1_ADM_s000
                                 + 1.28419 w_k1_AXP_s000
                                 + 1.00696 w_k1_DIS_s000
                                 + 1.02507 w_k1_FDX_s000
                                 + 1.08321 w_k1_IBM_s000
                                 + 0.921784 w_k1_INTC_s000
                                 + 1.0784 w_k1_KO_s000 + 2.18725 w_k1_MAT_s000
                                 + 1.0979 w_k1_MSFT_s000 + 1.01892 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.747524 w_k1_ADM_s000
                                 + 0.7244 w_k1_AXP_s000
                                 + 0.631791 w_k1_DIS_s000
                                 + 0.775566 w_k1_FDX_s000
                                 + 0.716402 w_k1_IBM_s000
                                 + 0.826179 w_k1_INTC_s000
                                 + 0.709147 w_k1_KO_s000
                                 + 0.811689 w_k1_MAT_s000
                                 + 0.870182 w_k1_MSFT_s000
                                 + 0.732352 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.902676 w_k1_ADM_s000
                                 + 1.02066 w_k1_AXP_s000
                                 + 0.903752 w_k1_DIS_s000
                                 + 0.890648 w_k1_FDX_s000
                                 + 1.37807 w_k1_IBM_s000
                                 + 0.686348 w_k1_INTC_s000
                                 + 1.11487 w_k1_KO_s000 + 2.10411 w_k1_MAT_s000
                                 + 0.873709 w_k1_MSFT_s000
                                 + 0.962002 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.967082 w_k1_ADM_s001
                                 + 0.875823 w_k1_AXP_s001
                                 + 0.76392 w_k1_DIS_s001
                                 + 0.924216 w_k1_FDX_s001
                                 + 0.650174 w_k1_IBM_s001
                                 + 1.1322 w_k1_INTC_s001
                                 + 0.751873 w_k1_KO_s001
                                 + 0.519964 w_k1_MAT_s001
                                 + 0.951307 w_k1_MSFT_s001
                                 + 0.977796 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.15953 w_k1_ADM_s001
                                 + 1.12925 w_k1_AXP_s001
                                 + 1.41579 w_k1_DIS_s001
                                 + 1.26723 w_k1_FDX_s001
                                 + 0.990607 w_k1_IBM_s001
                                 + 1.15458 w_k1_INTC_s001
                                 + 1.21212 w_k1_KO_s001
                                 + 0.777332 w_k1_MAT_s001
                                 + 1.13164 w_k1_MSFT_s001
                                 + 1.22333 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.25762 w_k1_ADM_s001
                                 + 1.44258 w_k1_AXP_s001 + 1.4444 w_k1_DIS_s001
                                 + 1.15445 w_k1_FDX_s001
                                 + 1.63321 w_k1_IBM_s001
                                 + 1.16298 w_k1_INTC_s001
                                 + 1.41278 w_k1_KO_s001 + 1.35928 w_k1_MAT_s001
                                 + 1.12607 w_k1_MSFT_s001
                                 + 1.28684 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.12064 w_k1_ADM_s001
                                 + 1.07403 w_k1_AXP_s001
                                 + 1.08402 w_k1_DIS_s001
                                 + 1.10488 w_k1_FDX_s001
                                 + 0.782813 w_k1_IBM_s001
                                 + 1.26816 w_k1_INTC_s001
                                 + 1.15726 w_k1_KO_s001
                                 + 0.990887 w_k1_MAT_s001
                                 + 1.19468 w_k1_MSFT_s001 + 1.1655 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.633544 w_k1_ADM_s001
                                 + 1.08984 w_k1_AXP_s001
                                 + 0.881843 w_k1_DIS_s001
                                 + 0.7761 w_k1_FDX_s001 + 1.35517 w_k1_IBM_s001
                                 + 0.956206 w_k1_INTC_s001
                                 + 1.09867 w_k1_KO_s001 + 1.27393 w_k1_MAT_s001
                                 + 0.945106 w_k1_MSFT_s001
                                 + 1.04104 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.680001 w_k1_ADM_s001
                                 + 0.794956 w_k1_AXP_s001
                                 + 0.67508 w_k1_DIS_s001
                                 + 0.731797 w_k1_FDX_s001
                                 + 0.76765 w_k1_IBM_s001
                                 + 0.813328 w_k1_INTC_s001
                                 + 0.847578 w_k1_KO_s001
                                 + 0.730457 w_k1_MAT_s001
                                 + 0.841834 w_k1_MSFT_s001
                                 + 0.914034 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.98152 w_k1_ADM_s001
                                 + 1.74996 w_k1_AXP_s001
                                 + 1.38563 w_k1_DIS_s001
                                 + 1.32614 w_k1_FDX_s001
                                 + 1.08639 w_k1_IBM_s001
                                 + 1.45529 w_k1_INTC_s001
                                 + 1.19346 w_k1_KO_s001 + 1.36 w_k1_MAT_s001
                                 + 1.15532 w_k1_MSFT_s001
                                 + 1.25786 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.07806 w_k1_ADM_s001
                                 + 1.24422 w_k1_AXP_s001
                                 + 1.10368 w_k1_DIS_s001
                                 + 0.977873 w_k1_FDX_s001
                                 + 1.1702 w_k1_IBM_s001
                                 + 1.15103 w_k1_INTC_s001
                                 + 1.13845 w_k1_KO_s001 + 1.04442 w_k1_MAT_s001
                                 + 1.03358 w_k1_MSFT_s001
                                 + 1.07777 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.89497 w_k1_ADM_s002
                                 + 0.981064 w_k1_AXP_s002
                                 + 1.0603 w_k1_DIS_s002 + 1.09186 w_k1_FDX_s002
                                 + 0.556347 w_k1_IBM_s002
                                 + 1.08783 w_k1_INTC_s002
                                 + 0.979147 w_k1_KO_s002
                                 + 0.546606 w_k1_MAT_s002
                                 + 1.33001 w_k1_MSFT_s002
                                 + 0.870282 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.924625 w_k1_ADM_s002
                                 + 0.599765 w_k1_AXP_s002
                                 + 0.788829 w_k1_DIS_s002
                                 + 0.86191 w_k1_FDX_s002
                                 + 0.587284 w_k1_IBM_s002
                                 + 0.707279 w_k1_INTC_s002
                                 + 0.740799 w_k1_KO_s002
                                 + 0.618344 w_k1_MAT_s002
                                 + 1.00133 w_k1_MSFT_s002
                                 + 0.833887 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15169 w_k1_ADM_s002
                                 + 1.24308 w_k1_AXP_s002
                                 + 1.66486 w_k1_DIS_s002
                                 + 1.26113 w_k1_FDX_s002
                                 + 1.09108 w_k1_IBM_s002
                                 + 1.05758 w_k1_INTC_s002
                                 + 1.42688 w_k1_KO_s002 + 1.60118 w_k1_MAT_s002
                                 + 1.36261 w_k1_MSFT_s002
                                 + 1.06756 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.00843 w_k1_ADM_s002
                                 + 1.07483 w_k1_AXP_s002
                                 + 1.06295 w_k1_DIS_s002
                                 + 0.988184 w_k1_FDX_s002
                                 + 0.944651 w_k1_IBM_s002
                                 + 0.866141 w_k1_INTC_s002
                                 + 1.06422 w_k1_KO_s002 + 1.23534 w_k1_MAT_s002
                                 + 1.14133 w_k1_MSFT_s002
                                 + 1.12776 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.75914 w_k1_ADM_s002
                                 + 1.19498 w_k1_AXP_s002
                                 + 0.716483 w_k1_DIS_s002
                                 + 0.942162 w_k1_FDX_s002
                                 + 0.730056 w_k1_IBM_s002
                                 + 0.996857 w_k1_INTC_s002
                                 + 0.827492 w_k1_KO_s002
                                 + 1.23533 w_k1_MAT_s002
                                 + 0.887647 w_k1_MSFT_s002
                                 + 0.934212 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.699842 w_k1_ADM_s002
                                 + 0.730958 w_k1_AXP_s002
                                 + 1.18039 w_k1_DIS_s002
                                 + 0.85752 w_k1_FDX_s002
                                 + 1.26804 w_k1_IBM_s002
                                 + 1.10502 w_k1_INTC_s002
                                 + 1.45492 w_k1_KO_s002 + 1.17597 w_k1_MAT_s002
                                 + 1.30626 w_k1_MSFT_s002 + 1.0612 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.08133 w_k1_ADM_s002
                                 + 1.40659 w_k1_AXP_s002
                                 + 0.952234 w_k1_DIS_s002
                                 + 1.08099 w_k1_FDX_s002
                                 + 1.29292 w_k1_IBM_s002
                                 + 0.929474 w_k1_INTC_s002
                                 + 1.33412 w_k1_KO_s002 + 1.64967 w_k1_MAT_s002
                                 + 1.03872 w_k1_MSFT_s002
                                 + 1.16793 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.72592 w_k1_ADM_s002
                                 + 0.740701 w_k1_AXP_s002
                                 + 0.723949 w_k1_DIS_s002
                                 + 0.77774 w_k1_FDX_s002
                                 + 0.889004 w_k1_IBM_s002
                                 + 1.0838 w_k1_INTC_s002
                                 + 0.982586 w_k1_KO_s002
                                 + 0.807822 w_k1_MAT_s002
                                 + 0.946039 w_k1_MSFT_s002
                                 + 0.868058 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.984999 w_k1_ADM_s003
                                 + 0.844623 w_k1_AXP_s003
                                 + 0.72105 w_k1_DIS_s003
                                 + 0.98199 w_k1_FDX_s003
                                 + 0.759213 w_k1_IBM_s003
                                 + 0.848417 w_k1_INTC_s003
                                 + 0.751862 w_k1_KO_s003
                                 + 0.614504 w_k1_MAT_s003
                                 + 1.03165 w_k1_MSFT_s003
                                 + 0.873694 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.19418 w_k1_ADM_s003
                                 + 1.12592 w_k1_AXP_s003
                                 + 1.36514 w_k1_DIS_s003
                                 + 1.25461 w_k1_FDX_s003 + 1.1268 w_k1_IBM_s003
                                 + 1.12615 w_k1_INTC_s003
                                 + 1.12783 w_k1_KO_s003
                                 + 0.808223 w_k1_MAT_s003
                                 + 1.25749 w_k1_MSFT_s003
                                 + 1.12289 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.664194 w_k1_ADM_s003
                                 + 0.990092 w_k1_AXP_s003
                                 + 1.05675 w_k1_DIS_s003
                                 + 0.839767 w_k1_FDX_s003
                                 + 1.57001 w_k1_IBM_s003
                                 + 1.12403 w_k1_INTC_s003 + 1.1439 w_k1_KO_s003
                                 + 1.50474 w_k1_MAT_s003
                                 + 1.08754 w_k1_MSFT_s003
                                 + 1.03894 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.17877 w_k1_ADM_s003
                                 + 0.763808 w_k1_AXP_s003
                                 + 1.03129 w_k1_DIS_s003
                                 + 0.850776 w_k1_FDX_s003
                                 + 1.15951 w_k1_IBM_s003
                                 + 0.92891 w_k1_INTC_s003
                                 + 1.02167 w_k1_KO_s003
                                 + 0.684573 w_k1_MAT_s003
                                 + 0.949843 w_k1_MSFT_s003
                                 + 1.1256 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.22554 w_k1_ADM_s003
                                 + 0.851829 w_k1_AXP_s003
                                 + 1.0502 w_k1_DIS_s003
                                 + 0.974901 w_k1_FDX_s003
                                 + 0.791607 w_k1_IBM_s003
                                 + 0.852627 w_k1_INTC_s003
                                 + 0.894996 w_k1_KO_s003
                                 + 0.678289 w_k1_MAT_s003
                                 + 1.00998 w_k1_MSFT_s003
                                 + 0.897328 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.18148 w_k1_ADM_s003
                                 + 1.54092 w_k1_AXP_s003
                                 + 1.44988 w_k1_DIS_s003
                                 + 1.33495 w_k1_FDX_s003
                                 + 1.12808 w_k1_IBM_s003
                                 + 1.57988 w_k1_INTC_s003
                                 + 1.20201 w_k1_KO_s003 + 1.31247 w_k1_MAT_s003
                                 + 1.27551 w_k1_MSFT_s003
                                 + 1.18262 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.855715 w_k1_ADM_s003
                                 + 0.864265 w_k1_AXP_s003
                                 + 0.744609 w_k1_DIS_s003
                                 + 0.836612 w_k1_FDX_s003
                                 + 1.08356 w_k1_IBM_s003
                                 + 0.89428 w_k1_INTC_s003
                                 + 0.928443 w_k1_KO_s003
                                 + 0.931365 w_k1_MAT_s003
                                 + 0.846713 w_k1_MSFT_s003
                                 + 0.927971 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.24192 w_k1_ADM_s003
                                 + 1.66589 w_k1_AXP_s003
                                 + 1.82235 w_k1_DIS_s003
                                 + 1.29097 w_k1_FDX_s003
                                 + 1.57182 w_k1_IBM_s003
                                 + 1.28712 w_k1_INTC_s003
                                 + 1.74851 w_k1_KO_s003 + 1.55621 w_k1_MAT_s003
                                 + 1.46512 w_k1_MSFT_s003 + 1.2592 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.961927 w_k1_ADM_s004
                                 + 0.837431 w_k1_AXP_s004
                                 + 1.10271 w_k1_DIS_s004
                                 + 0.951174 w_k1_FDX_s004
                                 + 1.51239 w_k1_IBM_s004
                                 + 1.09563 w_k1_INTC_s004
                                 + 1.20632 w_k1_KO_s004 + 1.00736 w_k1_MAT_s004
                                 + 1.02429 w_k1_MSFT_s004 + 0.9472 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.4611 w_k1_ADM_s004
                                 + 1.51926 w_k1_AXP_s004
                                 + 1.67696 w_k1_DIS_s004
                                 + 1.43261 w_k1_FDX_s004
                                 + 1.50119 w_k1_IBM_s004
                                 + 1.56394 w_k1_INTC_s004
                                 + 1.62348 w_k1_KO_s004 + 1.77328 w_k1_MAT_s004
                                 + 1.25582 w_k1_MSFT_s004
                                 + 1.25064 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.16318 w_k1_ADM_s004
                                 + 1.19625 w_k1_AXP_s004
                                 + 1.10171 w_k1_DIS_s004
                                 + 1.20371 w_k1_FDX_s004
                                 + 0.727309 w_k1_IBM_s004
                                 + 1.11843 w_k1_INTC_s004
                                 + 0.910235 w_k1_KO_s004
                                 + 0.978446 w_k1_MAT_s004
                                 + 1.19434 w_k1_MSFT_s004 + 1.0114 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.32243 w_k1_ADM_s004
                                 + 1.16091 w_k1_AXP_s004
                                 + 1.13058 w_k1_DIS_s004
                                 + 1.07258 w_k1_FDX_s004
                                 + 1.43296 w_k1_IBM_s004
                                 + 0.967437 w_k1_INTC_s004
                                 + 1.31996 w_k1_KO_s004 + 1.48719 w_k1_MAT_s004
                                 + 0.939659 w_k1_MSFT_s004
                                 + 1.16015 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.861243 w_k1_ADM_s004
                                 + 0.736126 w_k1_AXP_s004
                                 + 0.856265 w_k1_DIS_s004
                                 + 0.806025 w_k1_FDX_s004
                                 + 0.918396 w_k1_IBM_s004
                                 + 0.896543 w_k1_INTC_s004
                                 + 1.06982 w_k1_KO_s004
                                 + 0.939039 w_k1_MAT_s004
                                 + 0.914316 w_k1_MSFT_s004
                                 + 0.908339 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.27553 w_k1_ADM_s004
                                 + 1.19613 w_k1_AXP_s004
                                 + 1.05044 w_k1_DIS_s004
                                 + 1.23593 w_k1_FDX_s004
                                 + 0.973386 w_k1_IBM_s004
                                 + 1.27634 w_k1_INTC_s004
                                 + 1.11846 w_k1_KO_s004 + 0.89888 w_k1_MAT_s004
                                 + 1.12397 w_k1_MSFT_s004
                                 + 1.10031 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.711494 w_k1_ADM_s004
                                 + 1.02316 w_k1_AXP_s004
                                 + 0.784661 w_k1_DIS_s004
                                 + 0.915307 w_k1_FDX_s004
                                 + 0.975811 w_k1_IBM_s004
                                 + 0.91795 w_k1_INTC_s004
                                 + 0.845778 w_k1_KO_s004
                                 + 1.01599 w_k1_MAT_s004
                                 + 0.861979 w_k1_MSFT_s004
                                 + 0.973384 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.884723 w_k1_ADM_s004
                                 + 0.813489 w_k1_AXP_s004
                                 + 0.676867 w_k1_DIS_s004
                                 + 0.938532 w_k1_FDX_s004
                                 + 0.681621 w_k1_IBM_s004
                                 + 0.891667 w_k1_INTC_s004
                                 + 0.665965 w_k1_KO_s004
                                 + 0.592958 w_k1_MAT_s004
                                 + 0.912816 w_k1_MSFT_s004
                                 + 0.816938 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.662401 w_k1_ADM_s005
                                 + 0.895623 w_k1_AXP_s005
                                 + 0.983244 w_k1_DIS_s005
                                 + 0.883075 w_k1_FDX_s005
                                 + 1.07463 w_k1_IBM_s005
                                 + 1.07591 w_k1_INTC_s005
                                 + 1.12004 w_k1_KO_s005 + 1.11645 w_k1_MAT_s005
                                 + 0.993659 w_k1_MSFT_s005
                                 + 0.970987 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.884402 w_k1_ADM_s005
                                 + 0.717088 w_k1_AXP_s005
                                 + 0.613911 w_k1_DIS_s005
                                 + 0.73301 w_k1_FDX_s005
                                 + 0.80207 w_k1_IBM_s005
                                 + 0.891703 w_k1_INTC_s005
                                 + 0.861544 w_k1_KO_s005
                                 + 0.583231 w_k1_MAT_s005
                                 + 0.794932 w_k1_MSFT_s005
                                 + 0.783648 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.00237 w_k1_ADM_s005
                                 + 0.674817 w_k1_AXP_s005
                                 + 0.827115 w_k1_DIS_s005
                                 + 0.829347 w_k1_FDX_s005
                                 + 1.05009 w_k1_IBM_s005
                                 + 0.89509 w_k1_INTC_s005
                                 + 0.920391 w_k1_KO_s005
                                 + 0.532651 w_k1_MAT_s005
                                 + 0.88394 w_k1_MSFT_s005
                                 + 0.798028 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.14109 w_k1_ADM_s005
                                 + 1.46884 w_k1_AXP_s005
                                 + 1.15324 w_k1_DIS_s005
                                 + 1.14416 w_k1_FDX_s005 + 1.0844 w_k1_IBM_s005
                                 + 1.42329 w_k1_INTC_s005
                                 + 1.16156 w_k1_KO_s005 + 1.20905 w_k1_MAT_s005
                                 + 1.14339 w_k1_MSFT_s005
                                 + 1.14602 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.06574 w_k1_ADM_s005
                                 + 0.857831 w_k1_AXP_s005
                                 + 1.25457 w_k1_DIS_s005
                                 + 1.12722 w_k1_FDX_s005
                                 + 0.923561 w_k1_IBM_s005
                                 + 1.42083 w_k1_INTC_s005
                                 + 1.15133 w_k1_KO_s005
                                 + 0.889409 w_k1_MAT_s005
                                 + 1.30941 w_k1_MSFT_s005 + 1.1902 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.25903 w_k1_ADM_s005
                                 + 0.897344 w_k1_AXP_s005
                                 + 1.13358 w_k1_DIS_s005
                                 + 1.00925 w_k1_FDX_s005
                                 + 1.40508 w_k1_IBM_s005
                                 + 0.973307 w_k1_INTC_s005
                                 + 1.16277 w_k1_KO_s005
                                 + 0.949197 w_k1_MAT_s005
                                 + 1.1367 w_k1_MSFT_s005 + 1.05385 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.29996 w_k1_ADM_s005
                                 + 1.28975 w_k1_AXP_s005
                                 + 1.49355 w_k1_DIS_s005
                                 + 1.27078 w_k1_FDX_s005
                                 + 1.58331 w_k1_IBM_s005
                                 + 1.5867 w_k1_INTC_s005 + 1.48086 w_k1_KO_s005
                                 + 0.99477 w_k1_MAT_s005
                                 + 1.2747 w_k1_MSFT_s005 + 1.33595 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.898654 w_k1_ADM_s005
                                 + 0.778215 w_k1_AXP_s005
                                 + 0.791903 w_k1_DIS_s005
                                 + 0.942182 w_k1_FDX_s005
                                 + 0.85697 w_k1_IBM_s005
                                 + 1.06652 w_k1_INTC_s005
                                 + 0.824978 w_k1_KO_s005
                                 + 0.696368 w_k1_MAT_s005
                                 + 0.863364 w_k1_MSFT_s005
                                 + 0.892603 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.888845 w_k1_ADM_s006
                                 + 1.01628 w_k1_AXP_s006
                                 + 1.24708 w_k1_DIS_s006
                                 + 0.961036 w_k1_FDX_s006
                                 + 1.47352 w_k1_IBM_s006
                                 + 1.02043 w_k1_INTC_s006
                                 + 1.18184 w_k1_KO_s006 + 1.13772 w_k1_MAT_s006
                                 + 0.980623 w_k1_MSFT_s006
                                 + 1.09681 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.09277 w_k1_ADM_s006
                                 + 0.804688 w_k1_AXP_s006
                                 + 0.849093 w_k1_DIS_s006
                                 + 0.906567 w_k1_FDX_s006
                                 + 0.837043 w_k1_IBM_s006
                                 + 0.972386 w_k1_INTC_s006
                                 + 0.900938 w_k1_KO_s006
                                 + 0.704022 w_k1_MAT_s006
                                 + 0.967998 w_k1_MSFT_s006
                                 + 1.07347 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.03194 w_k1_ADM_s006
                                 + 0.502128 w_k1_AXP_s006
                                 + 0.686269 w_k1_DIS_s006
                                 + 0.768284 w_k1_FDX_s006
                                 + 0.91525 w_k1_IBM_s006
                                 + 0.965203 w_k1_INTC_s006
                                 + 0.812161 w_k1_KO_s006
                                 + 0.638631 w_k1_MAT_s006
                                 + 0.854637 w_k1_MSFT_s006
                                 + 0.959891 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.2005 w_k1_ADM_s006
                                 + 0.920131 w_k1_AXP_s006
                                 + 1.1575 w_k1_DIS_s006 + 1.09931 w_k1_FDX_s006
                                 + 1.03642 w_k1_IBM_s006
                                 + 1.19337 w_k1_INTC_s006
                                 + 1.03917 w_k1_KO_s006 + 1.02606 w_k1_MAT_s006
                                 + 1.13171 w_k1_MSFT_s006
                                 + 1.16931 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.04957 w_k1_ADM_s006
                                 + 1.14788 w_k1_AXP_s006
                                 + 0.832492 w_k1_DIS_s006
                                 + 0.879082 w_k1_FDX_s006
                                 + 1.08929 w_k1_IBM_s006
                                 + 0.824189 w_k1_INTC_s006
                                 + 0.983653 w_k1_KO_s006 + 1.6613 w_k1_MAT_s006
                                 + 0.851411 w_k1_MSFT_s006
                                 + 1.15411 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.13701 w_k1_ADM_s006
                                 + 1.12342 w_k1_AXP_s006
                                 + 1.13118 w_k1_DIS_s006
                                 + 1.17239 w_k1_FDX_s006
                                 + 0.83126 w_k1_IBM_s006
                                 + 1.03912 w_k1_INTC_s006
                                 + 1.08142 w_k1_KO_s006
                                 + 0.799118 w_k1_MAT_s006
                                 + 1.1967 w_k1_MSFT_s006 + 1.15581 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.96269 w_k1_ADM_s006
                                 + 1.38949 w_k1_AXP_s006
                                 + 1.24213 w_k1_DIS_s006 + 1.1909 w_k1_FDX_s006
                                 + 1.15688 w_k1_IBM_s006
                                 + 1.32792 w_k1_INTC_s006
                                 + 1.06435 w_k1_KO_s006 + 1.60273 w_k1_MAT_s006
                                 + 1.02599 w_k1_MSFT_s006
                                 + 1.10175 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.14093 w_k1_ADM_s006
                                 + 1.25642 w_k1_AXP_s006
                                 + 1.02352 w_k1_DIS_s006
                                 + 1.21704 w_k1_FDX_s006
                                 + 1.03756 w_k1_IBM_s006
                                 + 1.24502 w_k1_INTC_s006
                                 + 1.08659 w_k1_KO_s006 + 1.28763 w_k1_MAT_s006
                                 + 1.20075 w_k1_MSFT_s006
                                 + 1.33107 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.73498 w_k1_ADM_s007
                                 + 0.872438 w_k1_AXP_s007
                                 + 1.07197 w_k1_DIS_s007
                                 + 0.996215 w_k1_FDX_s007
                                 + 0.956034 w_k1_IBM_s007
                                 + 1.34917 w_k1_INTC_s007
                                 + 0.945773 w_k1_KO_s007
                                 + 0.883319 w_k1_MAT_s007
                                 + 0.985733 w_k1_MSFT_s007
                                 + 1.0018 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.852813 w_k1_ADM_s007
                                 + 1.45704 w_k1_AXP_s007
                                 + 1.14334 w_k1_DIS_s007
                                 + 1.21358 w_k1_FDX_s007
                                 + 1.44469 w_k1_IBM_s007
                                 + 1.4093 w_k1_INTC_s007 + 1.08113 w_k1_KO_s007
                                 + 1.85449 w_k1_MAT_s007
                                 + 1.05931 w_k1_MSFT_s007 + 1.0833 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.927967 w_k1_ADM_s007
                                 + 0.798744 w_k1_AXP_s007
                                 + 0.754894 w_k1_DIS_s007
                                 + 0.856085 w_k1_FDX_s007
                                 + 1.15397 w_k1_IBM_s007
                                 + 1.06428 w_k1_INTC_s007
                                 + 0.969994 w_k1_KO_s007
                                 + 1.13413 w_k1_MAT_s007
                                 + 0.977432 w_k1_MSFT_s007
                                 + 0.962442 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.10724 w_k1_ADM_s007
                                 + 1.40412 w_k1_AXP_s007
                                 + 1.22515 w_k1_DIS_s007
                                 + 1.34286 w_k1_FDX_s007
                                 + 0.856448 w_k1_IBM_s007
                                 + 1.24545 w_k1_INTC_s007 + 0.827 w_k1_KO_s007
                                 + 0.66866 w_k1_MAT_s007
                                 + 1.04214 w_k1_MSFT_s007
                                 + 0.943089 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.802647 w_k1_ADM_s007
                                 + 0.88671 w_k1_AXP_s007
                                 + 1.11205 w_k1_DIS_s007
                                 + 1.08917 w_k1_FDX_s007
                                 + 1.28145 w_k1_IBM_s007
                                 + 1.05549 w_k1_INTC_s007
                                 + 1.20921 w_k1_KO_s007 + 1.32491 w_k1_MAT_s007
                                 + 1.10258 w_k1_MSFT_s007
                                 + 1.03973 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.23514 w_k1_ADM_s007
                                 + 1.1554 w_k1_AXP_s007 + 1.33284 w_k1_DIS_s007
                                 + 1.27245 w_k1_FDX_s007
                                 + 1.09553 w_k1_IBM_s007
                                 + 1.17358 w_k1_INTC_s007
                                 + 1.12524 w_k1_KO_s007
                                 + 0.903385 w_k1_MAT_s007
                                 + 1.08651 w_k1_MSFT_s007
                                 + 1.06272 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.907441 w_k1_ADM_s007
                                 + 0.980779 w_k1_AXP_s007
                                 + 0.907147 w_k1_DIS_s007
                                 + 1.12731 w_k1_FDX_s007
                                 + 0.846005 w_k1_IBM_s007
                                 + 1.012 w_k1_INTC_s007 + 0.78492 w_k1_KO_s007
                                 + 0.732998 w_k1_MAT_s007
                                 + 0.93534 w_k1_MSFT_s007
                                 + 0.961641 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.858579 w_k1_ADM_s007
                                 + 0.822783 w_k1_AXP_s007
                                 + 0.678184 w_k1_DIS_s007
                                 + 0.830742 w_k1_FDX_s007
                                 + 0.773542 w_k1_IBM_s007
                                 + 0.937819 w_k1_INTC_s007
                                 + 0.719522 w_k1_KO_s007
                                 + 0.706112 w_k1_MAT_s007
                                 + 0.715455 w_k1_MSFT_s007
                                 + 0.784705 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.717769 w_k1_ADM_s010
                                 + 0.802014 w_k1_AXP_s010
                                 + 0.636716 w_k1_DIS_s010
                                 + 0.783322 w_k1_FDX_s010
                                 + 0.968812 w_k1_IBM_s010
                                 + 0.759546 w_k1_INTC_s010
                                 + 0.787019 w_k1_KO_s010
                                 + 0.817996 w_k1_MAT_s010
                                 + 0.767192 w_k1_MSFT_s010
                                 + 0.886733 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.845119 w_k1_ADM_s010
                                 + 1.05217 w_k1_AXP_s010 + 0.7876 w_k1_DIS_s010
                                 + 0.931239 w_k1_FDX_s010
                                 + 0.747954 w_k1_IBM_s010
                                 + 0.69365 w_k1_INTC_s010
                                 + 0.72494 w_k1_KO_s010
                                 + 0.771096 w_k1_MAT_s010
                                 + 0.820749 w_k1_MSFT_s010
                                 + 0.864912 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.08396 w_k1_ADM_s010
                                 + 1.15308 w_k1_AXP_s010
                                 + 1.01334 w_k1_DIS_s010
                                 + 1.00948 w_k1_FDX_s010
                                 + 1.08722 w_k1_IBM_s010
                                 + 0.978484 w_k1_INTC_s010
                                 + 1.09518 w_k1_KO_s010 + 1.03789 w_k1_MAT_s010
                                 + 1.13958 w_k1_MSFT_s010
                                 + 0.964099 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.36883 w_k1_ADM_s010
                                 + 1.49371 w_k1_AXP_s010
                                 + 1.79774 w_k1_DIS_s010
                                 + 1.44384 w_k1_FDX_s010
                                 + 1.16058 w_k1_IBM_s010
                                 + 1.20783 w_k1_INTC_s010
                                 + 1.21154 w_k1_KO_s010 + 1.21867 w_k1_MAT_s010
                                 + 1.37277 w_k1_MSFT_s010
                                 + 1.07885 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.89642 w_k1_ADM_s010
                                 + 1.06178 w_k1_AXP_s010
                                 + 0.771219 w_k1_DIS_s010
                                 + 0.881607 w_k1_FDX_s010
                                 + 1.52525 w_k1_IBM_s010
                                 + 0.838511 w_k1_INTC_s010
                                 + 1.06597 w_k1_KO_s010 + 1.69287 w_k1_MAT_s010
                                 + 0.829399 w_k1_MSFT_s010
                                 + 0.967977 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.1833 w_k1_ADM_s010
                                 + 1.31431 w_k1_AXP_s010
                                 + 1.27084 w_k1_DIS_s010
                                 + 1.15824 w_k1_FDX_s010
                                 + 1.61556 w_k1_IBM_s010
                                 + 1.00341 w_k1_INTC_s010
                                 + 1.26007 w_k1_KO_s010 + 1.6931 w_k1_MAT_s010
                                 + 1.13997 w_k1_MSFT_s010
                                 + 1.16071 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.3654 w_k1_ADM_s010
                                 + 1.33325 w_k1_AXP_s010
                                 + 1.18142 w_k1_DIS_s010
                                 + 1.37717 w_k1_FDX_s010
                                 + 0.885207 w_k1_IBM_s010
                                 + 1.03698 w_k1_INTC_s010
                                 + 1.03662 w_k1_KO_s010 + 1.1292 w_k1_MAT_s010
                                 + 1.21244 w_k1_MSFT_s010
                                 + 1.04656 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.867566 w_k1_ADM_s010
                                 + 0.888306 w_k1_AXP_s010
                                 + 0.670303 w_k1_DIS_s010
                                 + 0.868504 w_k1_FDX_s010
                                 + 1.21214 w_k1_IBM_s010
                                 + 1.00775 w_k1_INTC_s010
                                 + 1.02172 w_k1_KO_s010 + 1.05369 w_k1_MAT_s010
                                 + 0.85216 w_k1_MSFT_s010
                                 + 0.903175 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.980837 w_k1_ADM_s011
                                 + 1.13162 w_k1_AXP_s011
                                 + 1.22458 w_k1_DIS_s011
                                 + 0.944905 w_k1_FDX_s011
                                 + 1.6149 w_k1_IBM_s011
                                 + 0.804376 w_k1_INTC_s011
                                 + 1.32726 w_k1_KO_s011 + 1.78777 w_k1_MAT_s011
                                 + 0.944723 w_k1_MSFT_s011
                                 + 0.970973 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.860423 w_k1_ADM_s011
                                 + 0.76509 w_k1_AXP_s011
                                 + 1.03192 w_k1_DIS_s011
                                 + 0.887343 w_k1_FDX_s011
                                 + 0.87374 w_k1_IBM_s011
                                 + 0.687723 w_k1_INTC_s011
                                 + 0.958585 w_k1_KO_s011
                                 + 0.736102 w_k1_MAT_s011
                                 + 0.75391 w_k1_MSFT_s011
                                 + 0.895852 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.34738 w_k1_ADM_s011
                                 + 1.18725 w_k1_AXP_s011
                                 + 1.65766 w_k1_DIS_s011
                                 + 1.25298 w_k1_FDX_s011
                                 + 1.29581 w_k1_IBM_s011
                                 + 1.21669 w_k1_INTC_s011
                                 + 1.48724 w_k1_KO_s011 + 1.29444 w_k1_MAT_s011
                                 + 1.40217 w_k1_MSFT_s011 + 1.3017 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.27209 w_k1_ADM_s011
                                 + 1.19464 w_k1_AXP_s011
                                 + 1.09104 w_k1_DIS_s011
                                 + 1.12891 w_k1_FDX_s011
                                 + 0.791258 w_k1_IBM_s011
                                 + 0.94916 w_k1_INTC_s011
                                 + 1.03561 w_k1_KO_s011 + 1.07311 w_k1_MAT_s011
                                 + 1.13777 w_k1_MSFT_s011
                                 + 0.984303 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.02123 w_k1_ADM_s011
                                 + 1.04425 w_k1_AXP_s011 + 1.6095 w_k1_DIS_s011
                                 + 1.20528 w_k1_FDX_s011
                                 + 0.867714 w_k1_IBM_s011
                                 + 1.17073 w_k1_INTC_s011
                                 + 1.20779 w_k1_KO_s011
                                 + 0.890249 w_k1_MAT_s011
                                 + 1.28255 w_k1_MSFT_s011
                                 + 1.03061 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.825625 w_k1_ADM_s011
                                 + 0.667814 w_k1_AXP_s011
                                 + 0.61905 w_k1_DIS_s011
                                 + 0.738066 w_k1_FDX_s011
                                 + 0.70703 w_k1_IBM_s011
                                 + 0.6723 w_k1_INTC_s011
                                 + 0.724564 w_k1_KO_s011
                                 + 0.806398 w_k1_MAT_s011
                                 + 0.766494 w_k1_MSFT_s011
                                 + 0.834239 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.96152 w_k1_ADM_s011
                                 + 0.892194 w_k1_AXP_s011
                                 + 0.995488 w_k1_DIS_s011
                                 + 0.940309 w_k1_FDX_s011
                                 + 1.16901 w_k1_IBM_s011
                                 + 0.819752 w_k1_INTC_s011
                                 + 1.04693 w_k1_KO_s011 + 1.03574 w_k1_MAT_s011
                                 + 0.902045 w_k1_MSFT_s011
                                 + 0.961732 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.28536 w_k1_ADM_s011
                                 + 1.76515 w_k1_AXP_s011
                                 + 1.32727 w_k1_DIS_s011
                                 + 1.24085 w_k1_FDX_s011
                                 + 1.03107 w_k1_IBM_s011
                                 + 1.05595 w_k1_INTC_s011
                                 + 1.17041 w_k1_KO_s011 + 1.37453 w_k1_MAT_s011
                                 + 1.15002 w_k1_MSFT_s011
                                 + 1.02375 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.18957 w_k1_ADM_s012
                                 + 0.833884 w_k1_AXP_s012
                                 + 0.657427 w_k1_DIS_s012
                                 + 0.749499 w_k1_FDX_s012
                                 + 1.24574 w_k1_IBM_s012
                                 + 0.530626 w_k1_INTC_s012
                                 + 0.753338 w_k1_KO_s012
                                 + 1.05298 w_k1_MAT_s012
                                 + 0.578342 w_k1_MSFT_s012
                                 + 0.903167 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.63207 w_k1_ADM_s012
                                 + 1.429 w_k1_AXP_s012 + 1.38071 w_k1_DIS_s012
                                 + 1.31672 w_k1_FDX_s012
                                 + 1.37642 w_k1_IBM_s012
                                 + 0.963984 w_k1_INTC_s012
                                 + 1.16402 w_k1_KO_s012 + 1.36588 w_k1_MAT_s012
                                 + 1.08714 w_k1_MSFT_s012
                                 + 1.30459 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.28584 w_k1_ADM_s012
                                 + 0.829124 w_k1_AXP_s012
                                 + 0.689404 w_k1_DIS_s012
                                 + 0.860407 w_k1_FDX_s012
                                 + 0.665265 w_k1_IBM_s012
                                 + 0.774953 w_k1_INTC_s012
                                 + 0.67948 w_k1_KO_s012
                                 + 0.555399 w_k1_MAT_s012
                                 + 0.790012 w_k1_MSFT_s012
                                 + 0.818211 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.26224 w_k1_ADM_s012
                                 + 1.05767 w_k1_AXP_s012
                                 + 0.952776 w_k1_DIS_s012
                                 + 1.0182 w_k1_FDX_s012 + 1.04622 w_k1_IBM_s012
                                 + 0.981953 w_k1_INTC_s012
                                 + 0.997389 w_k1_KO_s012
                                 + 0.818844 w_k1_MAT_s012
                                 + 0.954732 w_k1_MSFT_s012
                                 + 1.06095 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.788337 w_k1_ADM_s012
                                 + 0.735635 w_k1_AXP_s012
                                 + 0.662751 w_k1_DIS_s012
                                 + 0.735145 w_k1_FDX_s012
                                 + 0.907947 w_k1_IBM_s012
                                 + 0.829176 w_k1_INTC_s012
                                 + 0.833724 w_k1_KO_s012
                                 + 0.756705 w_k1_MAT_s012
                                 + 0.792068 w_k1_MSFT_s012
                                 + 0.834502 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.01766 w_k1_ADM_s012
                                 + 1.13763 w_k1_AXP_s012
                                 + 0.801329 w_k1_DIS_s012
                                 + 0.831518 w_k1_FDX_s012
                                 + 1.73053 w_k1_IBM_s012
                                 + 0.86701 w_k1_INTC_s012
                                 + 1.02867 w_k1_KO_s012 + 1.29785 w_k1_MAT_s012
                                 + 0.809264 w_k1_MSFT_s012
                                 + 0.855682 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 2.21036 w_k1_ADM_s012
                                 + 2.27749 w_k1_AXP_s012
                                 + 2.00561 w_k1_DIS_s012
                                 + 1.73357 w_k1_FDX_s012
                                 + 1.31285 w_k1_IBM_s012
                                 + 1.22998 w_k1_INTC_s012 + 1.2736 w_k1_KO_s012
                                 + 1.65283 w_k1_MAT_s012
                                 + 1.2743 w_k1_MSFT_s012 + 1.35721 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.75523 w_k1_ADM_s012
                                 + 1.25535 w_k1_AXP_s012
                                 + 1.51552 w_k1_DIS_s012 + 1.3489 w_k1_FDX_s012
                                 + 0.886402 w_k1_IBM_s012
                                 + 1.13243 w_k1_INTC_s012
                                 + 1.08324 w_k1_KO_s012
                                 + 0.699031 w_k1_MAT_s012
                                 + 1.17418 w_k1_MSFT_s012
                                 + 1.08971 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.21502 w_k1_ADM_s013
                                 + 1.12684 w_k1_AXP_s013
                                 + 1.06351 w_k1_DIS_s013
                                 + 0.979487 w_k1_FDX_s013
                                 + 1.43003 w_k1_IBM_s013
                                 + 1.03144 w_k1_INTC_s013
                                 + 1.16956 w_k1_KO_s013 + 1.43688 w_k1_MAT_s013
                                 + 0.97045 w_k1_MSFT_s013
                                 + 1.11246 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.22002 w_k1_ADM_s013
                                 + 1.24679 w_k1_AXP_s013
                                 + 1.15169 w_k1_DIS_s013
                                 + 1.17983 w_k1_FDX_s013
                                 + 0.656577 w_k1_IBM_s013
                                 + 1.27708 w_k1_INTC_s013
                                 + 1.00423 w_k1_KO_s013
                                 + 0.644505 w_k1_MAT_s013
                                 + 1.103 w_k1_MSFT_s013 + 1.05617 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.12106 w_k1_ADM_s013
                                 + 0.788298 w_k1_AXP_s013
                                 + 0.879757 w_k1_DIS_s013
                                 + 0.987649 w_k1_FDX_s013
                                 + 0.660324 w_k1_IBM_s013
                                 + 0.905759 w_k1_INTC_s013
                                 + 0.901672 w_k1_KO_s013
                                 + 0.570031 w_k1_MAT_s013
                                 + 1.0582 w_k1_MSFT_s013
                                 + 0.903678 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.00828 w_k1_ADM_s013
                                 + 0.761727 w_k1_AXP_s013
                                 + 0.711183 w_k1_DIS_s013
                                 + 0.785958 w_k1_FDX_s013
                                 + 0.914353 w_k1_IBM_s013
                                 + 0.892616 w_k1_INTC_s013
                                 + 0.818254 w_k1_KO_s013
                                 + 0.831788 w_k1_MAT_s013
                                 + 0.865268 w_k1_MSFT_s013
                                 + 0.93373 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.950452 w_k1_ADM_s013
                                 + 0.670872 w_k1_AXP_s013
                                 + 0.80415 w_k1_DIS_s013
                                 + 0.850928 w_k1_FDX_s013
                                 + 0.652269 w_k1_IBM_s013
                                 + 0.868246 w_k1_INTC_s013
                                 + 0.834274 w_k1_KO_s013
                                 + 0.601246 w_k1_MAT_s013
                                 + 0.906865 w_k1_MSFT_s013
                                 + 0.867615 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.53326 w_k1_ADM_s013
                                 + 1.36704 w_k1_AXP_s013
                                 + 1.46179 w_k1_DIS_s013
                                 + 1.19796 w_k1_FDX_s013
                                 + 1.08255 w_k1_IBM_s013
                                 + 1.18597 w_k1_INTC_s013
                                 + 1.51061 w_k1_KO_s013 + 1.28468 w_k1_MAT_s013
                                 + 1.19889 w_k1_MSFT_s013
                                 + 1.21738 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.17128 w_k1_ADM_s013
                                 + 1.03452 w_k1_AXP_s013 + 1.0018 w_k1_DIS_s013
                                 + 0.910757 w_k1_FDX_s013
                                 + 1.07331 w_k1_IBM_s013
                                 + 1.00363 w_k1_INTC_s013
                                 + 1.20616 w_k1_KO_s013 + 1.39137 w_k1_MAT_s013
                                 + 1.10627 w_k1_MSFT_s013
                                 + 1.06774 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.2584 w_k1_ADM_s013
                                 + 1.15901 w_k1_AXP_s013
                                 + 0.953031 w_k1_DIS_s013
                                 + 1.00951 w_k1_FDX_s013
                                 + 0.814809 w_k1_IBM_s013
                                 + 0.993571 w_k1_INTC_s013
                                 + 1.01349 w_k1_KO_s013
                                 + 0.728404 w_k1_MAT_s013
                                 + 0.901981 w_k1_MSFT_s013
                                 + 1.02457 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.05817 w_k1_ADM_s014
                                 + 1.06781 w_k1_AXP_s014
                                 + 1.12677 w_k1_DIS_s014
                                 + 0.93444 w_k1_FDX_s014
                                 + 1.50586 w_k1_IBM_s014
                                 + 0.874859 w_k1_INTC_s014
                                 + 1.35949 w_k1_KO_s014 + 1.68846 w_k1_MAT_s014
                                 + 0.961222 w_k1_MSFT_s014
                                 + 1.24226 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.640169 w_k1_ADM_s014
                                 + 0.719855 w_k1_AXP_s014
                                 + 0.580488 w_k1_DIS_s014
                                 + 0.676209 w_k1_FDX_s014
                                 + 0.901923 w_k1_IBM_s014
                                 + 0.607276 w_k1_INTC_s014
                                 + 0.757985 w_k1_KO_s014
                                 + 0.983258 w_k1_MAT_s014
                                 + 0.626854 w_k1_MSFT_s014
                                 + 0.769698 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.945901 w_k1_ADM_s014
                                 + 0.960109 w_k1_AXP_s014
                                 + 0.734613 w_k1_DIS_s014
                                 + 0.877515 w_k1_FDX_s014
                                 + 0.658531 w_k1_IBM_s014
                                 + 0.751095 w_k1_INTC_s014
                                 + 0.771648 w_k1_KO_s014
                                 + 0.865516 w_k1_MAT_s014
                                 + 0.869634 w_k1_MSFT_s014
                                 + 0.857765 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20845 w_k1_ADM_s014
                                 + 1.14481 w_k1_AXP_s014
                                 + 1.26212 w_k1_DIS_s014
                                 + 1.29148 w_k1_FDX_s014
                                 + 0.773988 w_k1_IBM_s014
                                 + 1.15519 w_k1_INTC_s014
                                 + 0.983646 w_k1_KO_s014
                                 + 0.956464 w_k1_MAT_s014
                                 + 1.25735 w_k1_MSFT_s014 + 1.0455 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.838982 w_k1_ADM_s014
                                 + 0.838156 w_k1_AXP_s014
                                 + 1.11623 w_k1_DIS_s014
                                 + 0.889838 w_k1_FDX_s014
                                 + 1.00479 w_k1_IBM_s014
                                 + 0.859606 w_k1_INTC_s014
                                 + 1.17656 w_k1_KO_s014
                                 + 0.737798 w_k1_MAT_s014
                                 + 0.987867 w_k1_MSFT_s014
                                 + 0.952707 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.03227 w_k1_ADM_s014
                                 + 1.7398 w_k1_AXP_s014 + 1.64991 w_k1_DIS_s014
                                 + 1.30851 w_k1_FDX_s014 + 1.5621 w_k1_IBM_s014
                                 + 1.11626 w_k1_INTC_s014
                                 + 1.68055 w_k1_KO_s014 + 1.81088 w_k1_MAT_s014
                                 + 1.2547 w_k1_MSFT_s014 + 1.37028 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.05721 w_k1_ADM_s014
                                 + 1.21796 w_k1_AXP_s014
                                 + 0.728357 w_k1_DIS_s014
                                 + 0.90875 w_k1_FDX_s014
                                 + 1.18499 w_k1_IBM_s014
                                 + 0.869711 w_k1_INTC_s014
                                 + 0.910478 w_k1_KO_s014
                                 + 1.41027 w_k1_MAT_s014
                                 + 0.76654 w_k1_MSFT_s014
                                 + 0.962017 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.35844 w_k1_ADM_s014
                                 + 1.6325 w_k1_AXP_s014 + 1.28261 w_k1_DIS_s014
                                 + 1.16725 w_k1_FDX_s014
                                 + 1.18583 w_k1_IBM_s014
                                 + 0.960868 w_k1_INTC_s014 + 1.117 w_k1_KO_s014
                                 + 1.27527 w_k1_MAT_s014
                                 + 1.04457 w_k1_MSFT_s014
                                 + 1.05247 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.3109 w_k1_ADM_s015
                                 + 1.36338 w_k1_AXP_s015
                                 + 1.11716 w_k1_DIS_s015
                                 + 1.17508 w_k1_FDX_s015
                                 + 1.31666 w_k1_IBM_s015
                                 + 0.902143 w_k1_INTC_s015
                                 + 1.30278 w_k1_KO_s015 + 1.83467 w_k1_MAT_s015
                                 + 0.89211 w_k1_MSFT_s015
                                 + 1.24345 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.869125 w_k1_ADM_s015
                                 + 0.740649 w_k1_AXP_s015
                                 + 0.685572 w_k1_DIS_s015
                                 + 0.81487 w_k1_FDX_s015
                                 + 0.870081 w_k1_IBM_s015
                                 + 1.00211 w_k1_INTC_s015
                                 + 0.892423 w_k1_KO_s015
                                 + 0.668748 w_k1_MAT_s015
                                 + 0.799027 w_k1_MSFT_s015
                                 + 0.908448 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.47537 w_k1_ADM_s015
                                 + 0.910346 w_k1_AXP_s015
                                 + 1.0482 w_k1_DIS_s015 + 1.09462 w_k1_FDX_s015
                                 + 1.02001 w_k1_IBM_s015
                                 + 1.06212 w_k1_INTC_s015
                                 + 1.17947 w_k1_KO_s015
                                 + 0.844959 w_k1_MAT_s015
                                 + 1.01839 w_k1_MSFT_s015
                                 + 1.10459 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.52141 w_k1_ADM_s015
                                 + 1.41633 w_k1_AXP_s015
                                 + 1.62121 w_k1_DIS_s015
                                 + 1.39792 w_k1_FDX_s015
                                 + 1.17957 w_k1_IBM_s015
                                 + 1.47181 w_k1_INTC_s015
                                 + 1.24045 w_k1_KO_s015 + 1.08934 w_k1_MAT_s015
                                 + 1.14652 w_k1_MSFT_s015
                                 + 1.16258 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.1548 w_k1_ADM_s015
                                 + 0.965565 w_k1_AXP_s015
                                 + 1.16152 w_k1_DIS_s015 + 1.0465 w_k1_FDX_s015
                                 + 1.50206 w_k1_IBM_s015
                                 + 1.29366 w_k1_INTC_s015
                                 + 1.26272 w_k1_KO_s015 + 1.09385 w_k1_MAT_s015
                                 + 1.05242 w_k1_MSFT_s015
                                 + 1.11498 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.01566 w_k1_ADM_s015
                                 + 0.997825 w_k1_AXP_s015
                                 + 0.981102 w_k1_DIS_s015
                                 + 1.08095 w_k1_FDX_s015
                                 + 0.772025 w_k1_IBM_s015
                                 + 1.31675 w_k1_INTC_s015
                                 + 0.942323 w_k1_KO_s015
                                 + 0.593912 w_k1_MAT_s015
                                 + 0.955558 w_k1_MSFT_s015
                                 + 1.03085 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.729375 w_k1_ADM_s015
                                 + 0.771857 w_k1_AXP_s015
                                 + 0.741812 w_k1_DIS_s015
                                 + 0.788244 w_k1_FDX_s015
                                 + 1.2101 w_k1_IBM_s015
                                 + 0.792959 w_k1_INTC_s015
                                 + 0.97319 w_k1_KO_s015 + 1.22476 w_k1_MAT_s015
                                 + 0.735191 w_k1_MSFT_s015
                                 + 1.04697 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.47778 w_k1_ADM_s015
                                 + 1.20325 w_k1_AXP_s015
                                 + 1.41902 w_k1_DIS_s015
                                 + 1.31388 w_k1_FDX_s015
                                 + 0.814139 w_k1_IBM_s015
                                 + 1.28109 w_k1_INTC_s015
                                 + 1.09736 w_k1_KO_s015
                                 + 0.985915 w_k1_MAT_s015
                                 + 1.167 w_k1_MSFT_s015 + 1.25247 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.876389 w_k1_ADM_s016
                                 + 1.04683 w_k1_AXP_s016
                                 + 0.994963 w_k1_DIS_s016
                                 + 1.19925 w_k1_FDX_s016
                                 + 0.807259 w_k1_IBM_s016
                                 + 1.03611 w_k1_INTC_s016
                                 + 0.853553 w_k1_KO_s016
                                 + 1.19878 w_k1_MAT_s016
                                 + 1.13869 w_k1_MSFT_s016
                                 + 1.02594 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.23235 w_k1_ADM_s016
                                 + 1.25395 w_k1_AXP_s016
                                 + 1.63189 w_k1_DIS_s016
                                 + 1.43348 w_k1_FDX_s016
                                 + 1.17343 w_k1_IBM_s016
                                 + 1.19272 w_k1_INTC_s016
                                 + 1.16687 w_k1_KO_s016 + 2.08408 w_k1_MAT_s016
                                 + 1.38674 w_k1_MSFT_s016
                                 + 1.36909 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.72206 w_k1_ADM_s016
                                 + 0.764379 w_k1_AXP_s016
                                 + 0.614543 w_k1_DIS_s016
                                 + 0.699657 w_k1_FDX_s016
                                 + 1.08036 w_k1_IBM_s016
                                 + 0.869821 w_k1_INTC_s016
                                 + 0.674785 w_k1_KO_s016
                                 + 0.961156 w_k1_MAT_s016
                                 + 0.711781 w_k1_MSFT_s016
                                 + 0.835862 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.909175 w_k1_ADM_s016
                                 + 0.923111 w_k1_AXP_s016
                                 + 0.900831 w_k1_DIS_s016
                                 + 0.908882 w_k1_FDX_s016
                                 + 1.08729 w_k1_IBM_s016
                                 + 1.03663 w_k1_INTC_s016
                                 + 0.859675 w_k1_KO_s016
                                 + 1.28359 w_k1_MAT_s016
                                 + 0.976461 w_k1_MSFT_s016
                                 + 0.963363 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.911083 w_k1_ADM_s016
                                 + 0.995758 w_k1_AXP_s016
                                 + 0.874491 w_k1_DIS_s016
                                 + 0.861296 w_k1_FDX_s016
                                 + 1.44308 w_k1_IBM_s016
                                 + 1.00978 w_k1_INTC_s016
                                 + 1.02414 w_k1_KO_s016 + 1.41021 w_k1_MAT_s016
                                 + 0.877097 w_k1_MSFT_s016
                                 + 1.15841 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.72446 w_k1_ADM_s016
                                 + 0.724925 w_k1_AXP_s016
                                 + 0.57366 w_k1_DIS_s016
                                 + 0.731575 w_k1_FDX_s016
                                 + 0.652615 w_k1_IBM_s016
                                 + 0.878104 w_k1_INTC_s016
                                 + 0.548248 w_k1_KO_s016
                                 + 0.696901 w_k1_MAT_s016
                                 + 0.668538 w_k1_MSFT_s016
                                 + 0.805633 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.664703 w_k1_ADM_s016
                                 + 0.921573 w_k1_AXP_s016
                                 + 1.33217 w_k1_DIS_s016
                                 + 0.994732 w_k1_FDX_s016
                                 + 1.29482 w_k1_IBM_s016
                                 + 1.25052 w_k1_INTC_s016
                                 + 1.09093 w_k1_KO_s016 + 1.67315 w_k1_MAT_s016
                                 + 1.14748 w_k1_MSFT_s016
                                 + 1.14888 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.989304 w_k1_ADM_s016
                                 + 0.823559 w_k1_AXP_s016
                                 + 0.897562 w_k1_DIS_s016
                                 + 1.04758 w_k1_FDX_s016
                                 + 0.589171 w_k1_IBM_s016
                                 + 0.815692 w_k1_INTC_s016
                                 + 0.712171 w_k1_KO_s016
                                 + 1.02806 w_k1_MAT_s016
                                 + 0.984868 w_k1_MSFT_s016
                                 + 1.00593 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.02643 w_k1_ADM_s017
                                 + 1.56015 w_k1_AXP_s017
                                 + 1.22098 w_k1_DIS_s017
                                 + 1.18474 w_k1_FDX_s017 + 1.1388 w_k1_IBM_s017
                                 + 1.17207 w_k1_INTC_s017
                                 + 1.11168 w_k1_KO_s017 + 1.56441 w_k1_MAT_s017
                                 + 1.15475 w_k1_MSFT_s017
                                 + 1.04553 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.05638 w_k1_ADM_s017
                                 + 1.07402 w_k1_AXP_s017
                                 + 0.842157 w_k1_DIS_s017
                                 + 0.858982 w_k1_FDX_s017
                                 + 1.22638 w_k1_IBM_s017
                                 + 0.967213 w_k1_INTC_s017
                                 + 0.994586 w_k1_KO_s017
                                 + 1.27276 w_k1_MAT_s017
                                 + 0.940751 w_k1_MSFT_s017
                                 + 0.956468 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.0359 w_k1_ADM_s017
                                 + 0.937442 w_k1_AXP_s017
                                 + 1.11706 w_k1_DIS_s017
                                 + 1.00713 w_k1_FDX_s017
                                 + 1.07899 w_k1_IBM_s017
                                 + 0.841299 w_k1_INTC_s017
                                 + 0.936577 w_k1_KO_s017
                                 + 1.08994 w_k1_MAT_s017
                                 + 0.972282 w_k1_MSFT_s017
                                 + 0.975806 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.849005 w_k1_ADM_s017
                                 + 0.690729 w_k1_AXP_s017
                                 + 0.598187 w_k1_DIS_s017
                                 + 0.85769 w_k1_FDX_s017
                                 + 0.662994 w_k1_IBM_s017
                                 + 1.03628 w_k1_INTC_s017
                                 + 0.633484 w_k1_KO_s017
                                 + 0.470483 w_k1_MAT_s017
                                 + 0.831818 w_k1_MSFT_s017
                                 + 0.863322 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.15929 w_k1_ADM_s017
                                 + 0.928916 w_k1_AXP_s017
                                 + 0.978774 w_k1_DIS_s017
                                 + 1.05199 w_k1_FDX_s017
                                 + 0.631835 w_k1_IBM_s017
                                 + 1.05594 w_k1_INTC_s017
                                 + 0.791198 w_k1_KO_s017
                                 + 0.918373 w_k1_MAT_s017
                                 + 1.03954 w_k1_MSFT_s017
                                 + 0.930673 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.29457 w_k1_ADM_s017
                                 + 1.54334 w_k1_AXP_s017 + 1.8729 w_k1_DIS_s017
                                 + 1.30045 w_k1_FDX_s017
                                 + 1.42761 w_k1_IBM_s017
                                 + 1.02652 w_k1_INTC_s017
                                 + 1.43407 w_k1_KO_s017 + 1.42094 w_k1_MAT_s017
                                 + 1.1963 w_k1_MSFT_s017 + 1.17841 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.881096 w_k1_ADM_s017
                                 + 1.15927 w_k1_AXP_s017
                                 + 0.789453 w_k1_DIS_s017
                                 + 0.914747 w_k1_FDX_s017
                                 + 0.891108 w_k1_IBM_s017
                                 + 0.774219 w_k1_INTC_s017
                                 + 0.821585 w_k1_KO_s017
                                 + 1.49188 w_k1_MAT_s017
                                 + 0.854858 w_k1_MSFT_s017
                                 + 1.01026 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.871834 w_k1_ADM_s017
                                 + 0.797787 w_k1_AXP_s017
                                 + 0.90021 w_k1_DIS_s017
                                 + 0.799825 w_k1_FDX_s017
                                 + 1.00325 w_k1_IBM_s017
                                 + 0.947771 w_k1_INTC_s017
                                 + 0.915736 w_k1_KO_s017
                                 + 0.823581 w_k1_MAT_s017
                                 + 1.01053 w_k1_MSFT_s017
                                 + 0.823534 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.18205 w_k1_ADM_s100
                                 + 1.43814 w_k1_AXP_s100
                                 + 1.34928 w_k1_DIS_s100
                                 + 1.24723 w_k1_FDX_s100
                                 + 1.20523 w_k1_IBM_s100
                                 + 1.10454 w_k1_INTC_s100
                                 + 1.26589 w_k1_KO_s100 + 1.21023 w_k1_MAT_s100
                                 + 1.15348 w_k1_MSFT_s100
                                 + 1.10992 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.923597 w_k1_ADM_s100
                                 + 1.16651 w_k1_AXP_s100
                                 + 0.811757 w_k1_DIS_s100
                                 + 0.989356 w_k1_FDX_s100
                                 + 0.770866 w_k1_IBM_s100
                                 + 0.891812 w_k1_INTC_s100
                                 + 0.786458 w_k1_KO_s100
                                 + 0.877226 w_k1_MAT_s100
                                 + 0.779614 w_k1_MSFT_s100
                                 + 0.866976 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.14494 w_k1_ADM_s100
                                 + 1.63187 w_k1_AXP_s100
                                 + 1.54394 w_k1_DIS_s100
                                 + 1.42601 w_k1_FDX_s100
                                 + 0.851814 w_k1_IBM_s100
                                 + 1.27881 w_k1_INTC_s100 + 1.0305 w_k1_KO_s100
                                 + 1.06551 w_k1_MAT_s100
                                 + 1.21807 w_k1_MSFT_s100
                                 + 1.10644 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.91869 w_k1_ADM_s100
                                 + 0.763964 w_k1_AXP_s100
                                 + 0.863578 w_k1_DIS_s100
                                 + 0.888788 w_k1_FDX_s100
                                 + 1.22419 w_k1_IBM_s100
                                 + 0.953629 w_k1_INTC_s100
                                 + 1.07481 w_k1_KO_s100 + 1.10288 w_k1_MAT_s100
                                 + 0.932102 w_k1_MSFT_s100
                                 + 0.905485 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.966269 w_k1_ADM_s100
                                 + 1.33428 w_k1_AXP_s100
                                 + 1.29004 w_k1_DIS_s100
                                 + 1.06968 w_k1_FDX_s100
                                 + 1.47568 w_k1_IBM_s100
                                 + 1.11532 w_k1_INTC_s100
                                 + 1.25499 w_k1_KO_s100 + 1.27316 w_k1_MAT_s100
                                 + 1.06024 w_k1_MSFT_s100
                                 + 1.04434 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.1316 w_k1_ADM_s100
                                 + 1.12427 w_k1_AXP_s100
                                 + 1.34183 w_k1_DIS_s100
                                 + 1.19495 w_k1_FDX_s100 + 0.9405 w_k1_IBM_s100
                                 + 1.08293 w_k1_INTC_s100
                                 + 1.15866 w_k1_KO_s100 + 0.97177 w_k1_MAT_s100
                                 + 1.09095 w_k1_MSFT_s100
                                 + 1.06285 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.939235 w_k1_ADM_s100
                                 + 0.76405 w_k1_AXP_s100
                                 + 0.854085 w_k1_DIS_s100
                                 + 0.93531 w_k1_FDX_s100
                                 + 0.735345 w_k1_IBM_s100
                                 + 0.948695 w_k1_INTC_s100
                                 + 0.84781 w_k1_KO_s100
                                 + 0.731351 w_k1_MAT_s100
                                 + 0.917799 w_k1_MSFT_s100
                                 + 0.814382 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.864833 w_k1_ADM_s100
                                 + 1.15819 w_k1_AXP_s100
                                 + 0.829563 w_k1_DIS_s100
                                 + 0.969356 w_k1_FDX_s100
                                 + 0.981534 w_k1_IBM_s100
                                 + 0.869465 w_k1_INTC_s100
                                 + 0.834503 w_k1_KO_s100
                                 + 1.35035 w_k1_MAT_s100
                                 + 0.816521 w_k1_MSFT_s100
                                 + 0.887276 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.804843 w_k1_ADM_s101
                                 + 0.912955 w_k1_AXP_s101
                                 + 0.863611 w_k1_DIS_s101
                                 + 0.921503 w_k1_FDX_s101
                                 + 0.750533 w_k1_IBM_s101
                                 + 0.905405 w_k1_INTC_s101
                                 + 0.836873 w_k1_KO_s101
                                 + 0.744267 w_k1_MAT_s101
                                 + 0.906771 w_k1_MSFT_s101
                                 + 0.960914 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.13866 w_k1_ADM_s101
                                 + 1.64376 w_k1_AXP_s101
                                 + 1.34465 w_k1_DIS_s101
                                 + 1.28114 w_k1_FDX_s101 + 1.331 w_k1_IBM_s101
                                 + 1.43676 w_k1_INTC_s101
                                 + 1.27793 w_k1_KO_s101 + 1.40246 w_k1_MAT_s101
                                 + 1.15091 w_k1_MSFT_s101
                                 + 1.10839 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.812519 w_k1_ADM_s101
                                 + 0.912311 w_k1_AXP_s101
                                 + 0.897555 w_k1_DIS_s101
                                 + 0.941039 w_k1_FDX_s101
                                 + 1.2322 w_k1_IBM_s101
                                 + 1.02251 w_k1_INTC_s101
                                 + 1.13503 w_k1_KO_s101 + 1.28836 w_k1_MAT_s101
                                 + 0.919016 w_k1_MSFT_s101
                                 + 1.02789 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.1574 w_k1_ADM_s101
                                 + 1.11928 w_k1_AXP_s101
                                 + 1.06291 w_k1_DIS_s101
                                 + 1.19772 w_k1_FDX_s101
                                 + 0.65368 w_k1_IBM_s101
                                 + 1.15892 w_k1_INTC_s101
                                 + 0.823892 w_k1_KO_s101
                                 + 0.637357 w_k1_MAT_s101
                                 + 0.995438 w_k1_MSFT_s101
                                 + 1.07719 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.993197 w_k1_ADM_s101
                                 + 1.39257 w_k1_AXP_s101
                                 + 1.06217 w_k1_DIS_s101
                                 + 1.04339 w_k1_FDX_s101
                                 + 1.34413 w_k1_IBM_s101
                                 + 0.859856 w_k1_INTC_s101
                                 + 1.02031 w_k1_KO_s101 + 1.37172 w_k1_MAT_s101
                                 + 0.896084 w_k1_MSFT_s101
                                 + 1.12049 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.682916 w_k1_ADM_s101
                                 + 0.673245 w_k1_AXP_s101
                                 + 0.523322 w_k1_DIS_s101
                                 + 0.728421 w_k1_FDX_s101
                                 + 0.670498 w_k1_IBM_s101
                                 + 0.815906 w_k1_INTC_s101
                                 + 0.636957 w_k1_KO_s101
                                 + 0.68683 w_k1_MAT_s101
                                 + 0.749609 w_k1_MSFT_s101
                                 + 0.740841 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.831187 w_k1_ADM_s101
                                 + 0.696368 w_k1_AXP_s101
                                 + 0.721521 w_k1_DIS_s101
                                 + 0.755616 w_k1_FDX_s101
                                 + 0.97343 w_k1_IBM_s101
                                 + 0.950907 w_k1_INTC_s101
                                 + 0.900221 w_k1_KO_s101
                                 + 0.782427 w_k1_MAT_s101
                                 + 0.869613 w_k1_MSFT_s101
                                 + 0.941459 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.08236 w_k1_ADM_s101
                                 + 1.59255 w_k1_AXP_s101
                                 + 1.77779 w_k1_DIS_s101 + 1.4707 w_k1_FDX_s101
                                 + 1.17402 w_k1_IBM_s101
                                 + 1.44316 w_k1_INTC_s101
                                 + 1.15332 w_k1_KO_s101 + 1.40857 w_k1_MAT_s101
                                 + 1.2551 w_k1_MSFT_s101 + 1.18342 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.14782 w_k1_ADM_s102
                                 + 1.12745 w_k1_AXP_s102
                                 + 1.88329 w_k1_DIS_s102
                                 + 1.31852 w_k1_FDX_s102 + 1.1852 w_k1_IBM_s102
                                 + 1.28112 w_k1_INTC_s102 + 1.3633 w_k1_KO_s102
                                 + 1.26745 w_k1_MAT_s102
                                 + 1.71647 w_k1_MSFT_s102
                                 + 1.27473 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.842264 w_k1_ADM_s102
                                 + 0.744134 w_k1_AXP_s102
                                 + 0.738324 w_k1_DIS_s102
                                 + 0.760181 w_k1_FDX_s102
                                 + 0.760192 w_k1_IBM_s102
                                 + 0.864349 w_k1_INTC_s102
                                 + 0.771309 w_k1_KO_s102
                                 + 0.838761 w_k1_MAT_s102
                                 + 0.953963 w_k1_MSFT_s102
                                 + 0.836758 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.02588 w_k1_ADM_s102
                                 + 1.19346 w_k1_AXP_s102
                                 + 0.84964 w_k1_DIS_s102
                                 + 1.01363 w_k1_FDX_s102
                                 + 0.855351 w_k1_IBM_s102
                                 + 1.08389 w_k1_INTC_s102
                                 + 0.680292 w_k1_KO_s102
                                 + 0.690207 w_k1_MAT_s102
                                 + 1.03485 w_k1_MSFT_s102
                                 + 0.809696 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.15393 w_k1_ADM_s102
                                 + 0.852875 w_k1_AXP_s102
                                 + 1.13549 w_k1_DIS_s102 + 1.0449 w_k1_FDX_s102
                                 + 0.895575 w_k1_IBM_s102
                                 + 1.07231 w_k1_INTC_s102
                                 + 0.961958 w_k1_KO_s102
                                 + 0.629514 w_k1_MAT_s102
                                 + 1.21584 w_k1_MSFT_s102
                                 + 1.00473 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.06009 w_k1_ADM_s102
                                 + 0.9516 w_k1_AXP_s102 + 1.21026 w_k1_DIS_s102
                                 + 0.976062 w_k1_FDX_s102
                                 + 1.3653 w_k1_IBM_s102 + 1.1043 w_k1_INTC_s102
                                 + 1.15281 w_k1_KO_s102 + 1.08502 w_k1_MAT_s102
                                 + 1.16207 w_k1_MSFT_s102
                                 + 1.02581 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.07303 w_k1_ADM_s102
                                 + 1.64047 w_k1_AXP_s102
                                 + 1.35099 w_k1_DIS_s102
                                 + 1.18446 w_k1_FDX_s102
                                 + 1.48531 w_k1_IBM_s102
                                 + 1.23283 w_k1_INTC_s102
                                 + 1.12837 w_k1_KO_s102 + 1.91816 w_k1_MAT_s102
                                 + 1.23897 w_k1_MSFT_s102
                                 + 1.01607 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.03265 w_k1_ADM_s102
                                 + 0.842596 w_k1_AXP_s102
                                 + 0.778199 w_k1_DIS_s102
                                 + 0.776854 w_k1_FDX_s102
                                 + 1.16711 w_k1_IBM_s102
                                 + 0.827515 w_k1_INTC_s102
                                 + 0.819781 w_k1_KO_s102
                                 + 1.12314 w_k1_MAT_s102
                                 + 0.924602 w_k1_MSFT_s102
                                 + 0.883072 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.18691 w_k1_ADM_s102
                                 + 1.52392 w_k1_AXP_s102
                                 + 1.27922 w_k1_DIS_s102
                                 + 1.35349 w_k1_FDX_s102
                                 + 0.866695 w_k1_IBM_s102
                                 + 1.34807 w_k1_INTC_s102
                                 + 1.06767 w_k1_KO_s102
                                 + 0.972509 w_k1_MAT_s102
                                 + 1.45648 w_k1_MSFT_s102
                                 + 1.00525 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.799199 w_k1_ADM_s103
                                 + 0.762515 w_k1_AXP_s103
                                 + 0.692402 w_k1_DIS_s103
                                 + 0.868702 w_k1_FDX_s103
                                 + 0.624613 w_k1_IBM_s103
                                 + 0.870695 w_k1_INTC_s103
                                 + 0.655334 w_k1_KO_s103
                                 + 0.623413 w_k1_MAT_s103
                                 + 0.943767 w_k1_MSFT_s103
                                 + 0.794598 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.937746 w_k1_ADM_s103
                                 + 0.806061 w_k1_AXP_s103
                                 + 0.730718 w_k1_DIS_s103
                                 + 0.849976 w_k1_FDX_s103
                                 + 0.906094 w_k1_IBM_s103
                                 + 0.795784 w_k1_INTC_s103
                                 + 0.822976 w_k1_KO_s103
                                 + 0.946256 w_k1_MAT_s103
                                 + 0.929875 w_k1_MSFT_s103
                                 + 0.864213 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02735 w_k1_ADM_s103
                                 + 0.918042 w_k1_AXP_s103
                                 + 0.842499 w_k1_DIS_s103
                                 + 0.994359 w_k1_FDX_s103
                                 + 0.713424 w_k1_IBM_s103
                                 + 0.965363 w_k1_INTC_s103
                                 + 0.873011 w_k1_KO_s103
                                 + 0.597634 w_k1_MAT_s103
                                 + 1.08724 w_k1_MSFT_s103
                                 + 0.962902 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.13625 w_k1_ADM_s103
                                 + 1.36773 w_k1_AXP_s103
                                 + 0.969921 w_k1_DIS_s103
                                 + 1.18036 w_k1_FDX_s103
                                 + 0.906294 w_k1_IBM_s103
                                 + 0.917698 w_k1_INTC_s103
                                 + 0.881116 w_k1_KO_s103
                                 + 0.92571 w_k1_MAT_s103
                                 + 0.92668 w_k1_MSFT_s103
                                 + 0.899626 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.31427 w_k1_ADM_s103
                                 + 1.53345 w_k1_AXP_s103
                                 + 1.54211 w_k1_DIS_s103
                                 + 1.36171 w_k1_FDX_s103
                                 + 1.53712 w_k1_IBM_s103
                                 + 1.1606 w_k1_INTC_s103 + 1.26958 w_k1_KO_s103
                                 + 1.60841 w_k1_MAT_s103
                                 + 1.31259 w_k1_MSFT_s103
                                 + 1.28321 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.649383 w_k1_ADM_s103
                                 + 0.608997 w_k1_AXP_s103
                                 + 0.598664 w_k1_DIS_s103
                                 + 0.670371 w_k1_FDX_s103
                                 + 0.712391 w_k1_IBM_s103
                                 + 0.636082 w_k1_INTC_s103
                                 + 0.673864 w_k1_KO_s103
                                 + 0.711685 w_k1_MAT_s103
                                 + 0.632734 w_k1_MSFT_s103
                                 + 0.744417 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.904714 w_k1_ADM_s103
                                 + 0.991756 w_k1_AXP_s103
                                 + 0.939152 w_k1_DIS_s103
                                 + 0.925525 w_k1_FDX_s103
                                 + 1.17034 w_k1_IBM_s103
                                 + 1.05468 w_k1_INTC_s103
                                 + 1.00096 w_k1_KO_s103 + 1.31943 w_k1_MAT_s103
                                 + 0.950587 w_k1_MSFT_s103
                                 + 0.950886 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.984488 w_k1_ADM_s103
                                 + 0.928313 w_k1_AXP_s103
                                 + 1.38625 w_k1_DIS_s103
                                 + 1.27085 w_k1_FDX_s103
                                 + 0.851587 w_k1_IBM_s103
                                 + 1.0209 w_k1_INTC_s103 + 1.00423 w_k1_KO_s103
                                 + 0.692462 w_k1_MAT_s103
                                 + 1.19686 w_k1_MSFT_s103
                                 + 0.997458 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.828928 w_k1_ADM_s104
                                 + 0.582517 w_k1_AXP_s104
                                 + 0.443366 w_k1_DIS_s104
                                 + 0.711668 w_k1_FDX_s104
                                 + 0.618385 w_k1_IBM_s104
                                 + 0.852002 w_k1_INTC_s104
                                 + 0.560414 w_k1_KO_s104
                                 + 0.460499 w_k1_MAT_s104
                                 + 0.752939 w_k1_MSFT_s104
                                 + 0.779904 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.06184 w_k1_ADM_s104
                                 + 0.896661 w_k1_AXP_s104
                                 + 0.712536 w_k1_DIS_s104
                                 + 0.859708 w_k1_FDX_s104
                                 + 0.862403 w_k1_IBM_s104
                                 + 0.988934 w_k1_INTC_s104
                                 + 0.852301 w_k1_KO_s104
                                 + 0.788511 w_k1_MAT_s104
                                 + 0.851895 w_k1_MSFT_s104
                                 + 0.831451 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.54482 w_k1_ADM_s104
                                 + 1.72044 w_k1_AXP_s104
                                 + 1.60153 w_k1_DIS_s104
                                 + 1.39612 w_k1_FDX_s104
                                 + 1.20918 w_k1_IBM_s104
                                 + 1.32365 w_k1_INTC_s104
                                 + 1.28084 w_k1_KO_s104 + 1.26628 w_k1_MAT_s104
                                 + 1.27837 w_k1_MSFT_s104 + 1.1444 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.875025 w_k1_ADM_s104
                                 + 0.896183 w_k1_AXP_s104
                                 + 1.09782 w_k1_DIS_s104
                                 + 1.06031 w_k1_FDX_s104
                                 + 0.913282 w_k1_IBM_s104
                                 + 1.20756 w_k1_INTC_s104
                                 + 1.01134 w_k1_KO_s104 + 1.00611 w_k1_MAT_s104
                                 + 1.12 w_k1_MSFT_s104 + 1.0343 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.27963 w_k1_ADM_s104
                                 + 1.30936 w_k1_AXP_s104
                                 + 1.26104 w_k1_DIS_s104
                                 + 1.18451 w_k1_FDX_s104 + 1.4009 w_k1_IBM_s104
                                 + 1.28976 w_k1_INTC_s104
                                 + 1.27107 w_k1_KO_s104 + 1.36827 w_k1_MAT_s104
                                 + 1.08081 w_k1_MSFT_s104
                                 + 1.08948 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.01773 w_k1_ADM_s104
                                 + 0.846503 w_k1_AXP_s104
                                 + 0.78379 w_k1_DIS_s104
                                 + 0.937969 w_k1_FDX_s104
                                 + 0.598856 w_k1_IBM_s104
                                 + 0.906045 w_k1_INTC_s104
                                 + 0.846554 w_k1_KO_s104
                                 + 0.683199 w_k1_MAT_s104
                                 + 0.920441 w_k1_MSFT_s104
                                 + 0.906008 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.990206 w_k1_ADM_s104
                                 + 1.15237 w_k1_AXP_s104
                                 + 1.03692 w_k1_DIS_s104
                                 + 1.24224 w_k1_FDX_s104
                                 + 0.72375 w_k1_IBM_s104
                                 + 1.31061 w_k1_INTC_s104
                                 + 0.918799 w_k1_KO_s104
                                 + 0.737279 w_k1_MAT_s104
                                 + 1.0705 w_k1_MSFT_s104
                                 + 0.996047 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.994567 w_k1_ADM_s104
                                 + 0.92639 w_k1_AXP_s104
                                 + 0.815997 w_k1_DIS_s104
                                 + 0.80653 w_k1_FDX_s104
                                 + 1.37233 w_k1_IBM_s104
                                 + 1.01206 w_k1_INTC_s104
                                 + 1.12596 w_k1_KO_s104 + 1.49261 w_k1_MAT_s104
                                 + 0.844005 w_k1_MSFT_s104
                                 + 1.01676 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.997059 w_k1_ADM_s105
                                 + 0.96237 w_k1_AXP_s105
                                 + 1.23242 w_k1_DIS_s105 + 0.9189 w_k1_FDX_s105
                                 + 1.35543 w_k1_IBM_s105
                                 + 0.99568 w_k1_INTC_s105 + 1.2027 w_k1_KO_s105
                                 + 1.18269 w_k1_MAT_s105
                                 + 1.05385 w_k1_MSFT_s105
                                 + 1.04031 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.978566 w_k1_ADM_s105
                                 + 0.576823 w_k1_AXP_s105
                                 + 0.67352 w_k1_DIS_s105
                                 + 0.681751 w_k1_FDX_s105
                                 + 0.878545 w_k1_IBM_s105
                                 + 0.684915 w_k1_INTC_s105
                                 + 0.805794 w_k1_KO_s105
                                 + 0.393164 w_k1_MAT_s105
                                 + 0.72414 w_k1_MSFT_s105
                                 + 0.820231 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.883031 w_k1_ADM_s105
                                 + 0.774815 w_k1_AXP_s105
                                 + 0.844764 w_k1_DIS_s105
                                 + 0.834846 w_k1_FDX_s105
                                 + 1.14031 w_k1_IBM_s105
                                 + 0.916807 w_k1_INTC_s105
                                 + 0.935419 w_k1_KO_s105
                                 + 0.855776 w_k1_MAT_s105
                                 + 0.77472 w_k1_MSFT_s105
                                 + 0.89885 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.38005 w_k1_ADM_s105
                                 + 1.28624 w_k1_AXP_s105
                                 + 1.17635 w_k1_DIS_s105
                                 + 1.30776 w_k1_FDX_s105
                                 + 0.834648 w_k1_IBM_s105
                                 + 1.14403 w_k1_INTC_s105
                                 + 1.00705 w_k1_KO_s105 + 0.83422 w_k1_MAT_s105
                                 + 1.02783 w_k1_MSFT_s105
                                 + 1.08477 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.09658 w_k1_ADM_s105
                                 + 1.35046 w_k1_AXP_s105
                                 + 1.11436 w_k1_DIS_s105
                                 + 1.01959 w_k1_FDX_s105 + 1.3657 w_k1_IBM_s105
                                 + 1.0407 w_k1_INTC_s105 + 1.01294 w_k1_KO_s105
                                 + 1.0652 w_k1_MAT_s105
                                 + 0.807385 w_k1_MSFT_s105
                                 + 1.00877 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.10898 w_k1_ADM_s105
                                 + 0.933396 w_k1_AXP_s105
                                 + 0.778447 w_k1_DIS_s105
                                 + 0.941769 w_k1_FDX_s105
                                 + 0.586871 w_k1_IBM_s105
                                 + 0.845559 w_k1_INTC_s105
                                 + 0.647992 w_k1_KO_s105
                                 + 0.609277 w_k1_MAT_s105
                                 + 0.777298 w_k1_MSFT_s105
                                 + 0.796627 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.42077 w_k1_ADM_s105
                                 + 1.28853 w_k1_AXP_s105
                                 + 1.53518 w_k1_DIS_s105
                                 + 1.22336 w_k1_FDX_s105
                                 + 1.20488 w_k1_IBM_s105
                                 + 1.18796 w_k1_INTC_s105
                                 + 1.15677 w_k1_KO_s105 + 1.01505 w_k1_MAT_s105
                                 + 1.22703 w_k1_MSFT_s105 + 1.089 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.25419 w_k1_ADM_s105
                                 + 0.9729 w_k1_AXP_s105
                                 + 0.926112 w_k1_DIS_s105
                                 + 0.948976 w_k1_FDX_s105
                                 + 0.902413 w_k1_IBM_s105
                                 + 0.802591 w_k1_INTC_s105
                                 + 0.894571 w_k1_KO_s105
                                 + 0.693508 w_k1_MAT_s105
                                 + 0.882233 w_k1_MSFT_s105
                                 + 0.884634 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.900722 w_k1_ADM_s106
                                 + 0.838887 w_k1_AXP_s106
                                 + 1.0067 w_k1_DIS_s106 + 1.01088 w_k1_FDX_s106
                                 + 0.868575 w_k1_IBM_s106
                                 + 1.17307 w_k1_INTC_s106
                                 + 0.94506 w_k1_KO_s106
                                 + 0.766484 w_k1_MAT_s106
                                 + 0.953696 w_k1_MSFT_s106
                                 + 0.965422 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.19062 w_k1_ADM_s106
                                 + 1.32451 w_k1_AXP_s106
                                 + 1.07567 w_k1_DIS_s106
                                 + 1.08413 w_k1_FDX_s106 + 1.4568 w_k1_IBM_s106
                                 + 1.09224 w_k1_INTC_s106
                                 + 1.08529 w_k1_KO_s106 + 1.46872 w_k1_MAT_s106
                                 + 1.04029 w_k1_MSFT_s106
                                 + 1.03703 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.781377 w_k1_ADM_s106
                                 + 0.835312 w_k1_AXP_s106
                                 + 0.887466 w_k1_DIS_s106
                                 + 0.844225 w_k1_FDX_s106
                                 + 1.37527 w_k1_IBM_s106
                                 + 0.906138 w_k1_INTC_s106
                                 + 1.03014 w_k1_KO_s106 + 1.72801 w_k1_MAT_s106
                                 + 0.888933 w_k1_MSFT_s106
                                 + 1.01381 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.24567 w_k1_ADM_s106
                                 + 1.35892 w_k1_AXP_s106
                                 + 1.02405 w_k1_DIS_s106
                                 + 1.19049 w_k1_FDX_s106
                                 + 0.922582 w_k1_IBM_s106
                                 + 1.02318 w_k1_INTC_s106
                                 + 1.13718 w_k1_KO_s106 + 1.62077 w_k1_MAT_s106
                                 + 1.09929 w_k1_MSFT_s106
                                 + 1.09543 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.08291 w_k1_ADM_s106
                                 + 1.04121 w_k1_AXP_s106
                                 + 1.61019 w_k1_DIS_s106
                                 + 1.24551 w_k1_FDX_s106
                                 + 1.20217 w_k1_IBM_s106
                                 + 1.08376 w_k1_INTC_s106
                                 + 1.19499 w_k1_KO_s106 + 1.33271 w_k1_MAT_s106
                                 + 1.17017 w_k1_MSFT_s106
                                 + 1.18033 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.764137 w_k1_ADM_s106
                                 + 0.730448 w_k1_AXP_s106
                                 + 0.624818 w_k1_DIS_s106
                                 + 0.829457 w_k1_FDX_s106
                                 + 0.629299 w_k1_IBM_s106
                                 + 0.818439 w_k1_INTC_s106
                                 + 0.670939 w_k1_KO_s106
                                 + 0.830974 w_k1_MAT_s106
                                 + 0.775492 w_k1_MSFT_s106
                                 + 0.81145 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.763792 w_k1_ADM_s106
                                 + 0.635467 w_k1_AXP_s106
                                 + 0.545982 w_k1_DIS_s106
                                 + 0.667546 w_k1_FDX_s106
                                 + 0.865415 w_k1_IBM_s106
                                 + 0.71214 w_k1_INTC_s106
                                 + 0.837093 w_k1_KO_s106
                                 + 0.943604 w_k1_MAT_s106
                                 + 0.732821 w_k1_MSFT_s106
                                 + 0.803696 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.988188 w_k1_ADM_s106
                                 + 0.904332 w_k1_AXP_s106
                                 + 0.830314 w_k1_DIS_s106
                                 + 0.873343 w_k1_FDX_s106
                                 + 0.978011 w_k1_IBM_s106
                                 + 0.890288 w_k1_INTC_s106
                                 + 0.935093 w_k1_KO_s106
                                 + 1.01892 w_k1_MAT_s106
                                 + 0.790988 w_k1_MSFT_s106
                                 + 1.00056 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.0858 w_k1_ADM_s107
                                 + 1.26931 w_k1_AXP_s107
                                 + 0.998854 w_k1_DIS_s107
                                 + 1.1233 w_k1_FDX_s107 + 1.10921 w_k1_IBM_s107
                                 + 1.1383 w_k1_INTC_s107 + 1.09668 w_k1_KO_s107
                                 + 1.1099 w_k1_MAT_s107
                                 + 1.06107 w_k1_MSFT_s107 + 1.0091 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.02086 w_k1_ADM_s107
                                 + 1.14481 w_k1_AXP_s107
                                 + 0.88105 w_k1_DIS_s107
                                 + 1.02732 w_k1_FDX_s107
                                 + 0.695722 w_k1_IBM_s107
                                 + 0.968996 w_k1_INTC_s107
                                 + 0.880436 w_k1_KO_s107
                                 + 0.699341 w_k1_MAT_s107
                                 + 1.00516 w_k1_MSFT_s107
                                 + 0.896318 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.01217 w_k1_ADM_s107
                                 + 0.938139 w_k1_AXP_s107
                                 + 0.821936 w_k1_DIS_s107
                                 + 0.848275 w_k1_FDX_s107
                                 + 0.949321 w_k1_IBM_s107
                                 + 0.967867 w_k1_INTC_s107
                                 + 0.942921 w_k1_KO_s107 + 1.058 w_k1_MAT_s107
                                 + 0.963218 w_k1_MSFT_s107
                                 + 0.8818 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.677425 w_k1_ADM_s107
                                 + 0.734995 w_k1_AXP_s107
                                 + 0.618046 w_k1_DIS_s107
                                 + 0.723977 w_k1_FDX_s107
                                 + 1.15778 w_k1_IBM_s107
                                 + 0.811526 w_k1_INTC_s107
                                 + 0.911231 w_k1_KO_s107
                                 + 0.990132 w_k1_MAT_s107
                                 + 0.723507 w_k1_MSFT_s107
                                 + 0.945442 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.31615 w_k1_ADM_s107
                                 + 1.58198 w_k1_AXP_s107
                                 + 1.36927 w_k1_DIS_s107
                                 + 1.19197 w_k1_FDX_s107
                                 + 1.59497 w_k1_IBM_s107
                                 + 1.16837 w_k1_INTC_s107
                                 + 1.51081 w_k1_KO_s107 + 2.50256 w_k1_MAT_s107
                                 + 1.32585 w_k1_MSFT_s107
                                 + 1.18271 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.939146 w_k1_ADM_s107
                                 + 1.06628 w_k1_AXP_s107
                                 + 0.866396 w_k1_DIS_s107
                                 + 0.925154 w_k1_FDX_s107
                                 + 1.42704 w_k1_IBM_s107
                                 + 1.02466 w_k1_INTC_s107
                                 + 1.18558 w_k1_KO_s107 + 1.44995 w_k1_MAT_s107
                                 + 0.934862 w_k1_MSFT_s107
                                 + 0.965946 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.816752 w_k1_ADM_s107
                                 + 0.71037 w_k1_AXP_s107
                                 + 0.536343 w_k1_DIS_s107
                                 + 0.753488 w_k1_FDX_s107
                                 + 0.691393 w_k1_IBM_s107
                                 + 0.826882 w_k1_INTC_s107
                                 + 0.737185 w_k1_KO_s107
                                 + 0.834391 w_k1_MAT_s107
                                 + 0.727925 w_k1_MSFT_s107
                                 + 0.780581 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.20161 w_k1_ADM_s107
                                 + 1.23467 w_k1_AXP_s107
                                 + 1.29536 w_k1_DIS_s107
                                 + 1.24294 w_k1_FDX_s107
                                 + 0.898929 w_k1_IBM_s107
                                 + 1.36931 w_k1_INTC_s107
                                 + 1.18378 w_k1_KO_s107
                                 + 0.784101 w_k1_MAT_s107
                                 + 1.19102 w_k1_MSFT_s107
                                 + 1.11195 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20097 w_k1_ADM_s110
                                 + 1.19499 w_k1_AXP_s110
                                 + 1.14904 w_k1_DIS_s110
                                 + 1.13517 w_k1_FDX_s110
                                 + 0.908299 w_k1_IBM_s110
                                 + 1.26197 w_k1_INTC_s110
                                 + 0.858949 w_k1_KO_s110
                                 + 1.03707 w_k1_MAT_s110
                                 + 1.03887 w_k1_MSFT_s110
                                 + 0.973238 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.181 w_k1_ADM_s110
                                 + 1.27969 w_k1_AXP_s110
                                 + 1.08903 w_k1_DIS_s110
                                 + 1.06113 w_k1_FDX_s110
                                 + 1.76209 w_k1_IBM_s110
                                 + 0.985284 w_k1_INTC_s110
                                 + 1.27944 w_k1_KO_s110 + 1.78993 w_k1_MAT_s110
                                 + 0.968453 w_k1_MSFT_s110
                                 + 1.04806 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.04311 w_k1_ADM_s110
                                 + 0.773061 w_k1_AXP_s110
                                 + 0.774385 w_k1_DIS_s110
                                 + 0.897437 w_k1_FDX_s110
                                 + 0.62816 w_k1_IBM_s110
                                 + 0.968447 w_k1_INTC_s110
                                 + 0.715593 w_k1_KO_s110
                                 + 0.472092 w_k1_MAT_s110
                                 + 0.890258 w_k1_MSFT_s110
                                 + 0.954598 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.803688 w_k1_ADM_s110
                                 + 0.757042 w_k1_AXP_s110
                                 + 0.658697 w_k1_DIS_s110
                                 + 0.764647 w_k1_FDX_s110
                                 + 0.805203 w_k1_IBM_s110
                                 + 0.881981 w_k1_INTC_s110
                                 + 0.731975 w_k1_KO_s110
                                 + 0.885002 w_k1_MAT_s110
                                 + 0.782432 w_k1_MSFT_s110
                                 + 0.844888 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.09926 w_k1_ADM_s110
                                 + 1.04907 w_k1_AXP_s110
                                 + 1.11046 w_k1_DIS_s110
                                 + 1.01887 w_k1_FDX_s110
                                 + 1.31956 w_k1_IBM_s110
                                 + 0.997193 w_k1_INTC_s110
                                 + 1.09316 w_k1_KO_s110 + 1.56552 w_k1_MAT_s110
                                 + 0.910261 w_k1_MSFT_s110
                                 + 1.01716 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.978029 w_k1_ADM_s110
                                 + 0.935847 w_k1_AXP_s110
                                 + 0.859557 w_k1_DIS_s110
                                 + 0.832966 w_k1_FDX_s110
                                 + 1.09516 w_k1_IBM_s110
                                 + 0.875443 w_k1_INTC_s110
                                 + 0.923716 w_k1_KO_s110
                                 + 1.26544 w_k1_MAT_s110
                                 + 0.93172 w_k1_MSFT_s110
                                 + 0.912672 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.854375 w_k1_ADM_s110
                                 + 1.00073 w_k1_AXP_s110
                                 + 0.861037 w_k1_DIS_s110
                                 + 0.95577 w_k1_FDX_s110
                                 + 0.812329 w_k1_IBM_s110
                                 + 1.03458 w_k1_INTC_s110
                                 + 0.74347 w_k1_KO_s110 + 1.01151 w_k1_MAT_s110
                                 + 0.795379 w_k1_MSFT_s110
                                 + 0.894227 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.23097 w_k1_ADM_s110
                                 + 1.68547 w_k1_AXP_s110 + 1.6227 w_k1_DIS_s110
                                 + 1.38698 w_k1_FDX_s110
                                 + 1.54282 w_k1_IBM_s110
                                 + 1.26501 w_k1_INTC_s110
                                 + 1.43582 w_k1_KO_s110 + 1.76925 w_k1_MAT_s110
                                 + 1.19408 w_k1_MSFT_s110
                                 + 1.21144 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.984142 w_k1_ADM_s111
                                 + 0.738406 w_k1_AXP_s111
                                 + 0.811822 w_k1_DIS_s111
                                 + 0.877848 w_k1_FDX_s111
                                 + 1.0363 w_k1_IBM_s111
                                 + 0.965464 w_k1_INTC_s111
                                 + 0.973313 w_k1_KO_s111
                                 + 1.05179 w_k1_MAT_s111
                                 + 0.779284 w_k1_MSFT_s111
                                 + 0.929783 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.927982 w_k1_ADM_s111
                                 + 0.938844 w_k1_AXP_s111
                                 + 1.06077 w_k1_DIS_s111
                                 + 0.999916 w_k1_FDX_s111
                                 + 1.35314 w_k1_IBM_s111
                                 + 0.882478 w_k1_INTC_s111
                                 + 1.26205 w_k1_KO_s111 + 1.8237 w_k1_MAT_s111
                                 + 0.912619 w_k1_MSFT_s111
                                 + 0.980483 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.944498 w_k1_ADM_s111
                                 + 1.38916 w_k1_AXP_s111
                                 + 1.28475 w_k1_DIS_s111 + 1.1603 w_k1_FDX_s111
                                 + 1.82093 w_k1_IBM_s111
                                 + 0.998804 w_k1_INTC_s111
                                 + 1.26008 w_k1_KO_s111 + 2.15669 w_k1_MAT_s111
                                 + 0.933448 w_k1_MSFT_s111
                                 + 1.01509 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.27618 w_k1_ADM_s111
                                 + 1.38758 w_k1_AXP_s111
                                 + 1.72331 w_k1_DIS_s111
                                 + 1.33508 w_k1_FDX_s111
                                 + 1.26628 w_k1_IBM_s111
                                 + 0.935003 w_k1_INTC_s111
                                 + 1.25883 w_k1_KO_s111 + 1.64784 w_k1_MAT_s111
                                 + 1.06088 w_k1_MSFT_s111 + 1.1579 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03247 w_k1_ADM_s111
                                 + 1.01182 w_k1_AXP_s111
                                 + 1.21211 w_k1_DIS_s111
                                 + 1.05953 w_k1_FDX_s111
                                 + 1.17828 w_k1_IBM_s111
                                 + 0.857429 w_k1_INTC_s111
                                 + 1.09263 w_k1_KO_s111 + 1.31437 w_k1_MAT_s111
                                 + 0.935729 w_k1_MSFT_s111
                                 + 0.976024 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.895433 w_k1_ADM_s111
                                 + 1.0845 w_k1_AXP_s111
                                 + 0.991183 w_k1_DIS_s111
                                 + 1.07083 w_k1_FDX_s111
                                 + 0.802506 w_k1_IBM_s111
                                 + 0.940178 w_k1_INTC_s111
                                 + 0.86865 w_k1_KO_s111 + 1.07418 w_k1_MAT_s111
                                 + 0.979423 w_k1_MSFT_s111
                                 + 0.903137 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.01738 w_k1_ADM_s111
                                 + 0.70717 w_k1_AXP_s111
                                 + 0.797174 w_k1_DIS_s111
                                 + 0.920046 w_k1_FDX_s111
                                 + 0.644844 w_k1_IBM_s111
                                 + 0.831646 w_k1_INTC_s111
                                 + 0.753727 w_k1_KO_s111
                                 + 0.771246 w_k1_MAT_s111
                                 + 0.900466 w_k1_MSFT_s111
                                 + 0.858782 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05188 w_k1_ADM_s111
                                 + 1.37331 w_k1_AXP_s111
                                 + 1.20765 w_k1_DIS_s111
                                 + 1.30139 w_k1_FDX_s111
                                 + 1.03836 w_k1_IBM_s111
                                 + 1.12317 w_k1_INTC_s111
                                 + 1.27564 w_k1_KO_s111 + 1.93141 w_k1_MAT_s111
                                 + 0.982165 w_k1_MSFT_s111
                                 + 1.06988 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.576333 w_k1_ADM_s112
                                 + 0.72697 w_k1_AXP_s112
                                 + 0.715435 w_k1_DIS_s112
                                 + 0.654968 w_k1_FDX_s112
                                 + 0.839943 w_k1_IBM_s112
                                 + 0.75384 w_k1_INTC_s112
                                 + 0.736802 w_k1_KO_s112
                                 + 0.833046 w_k1_MAT_s112
                                 + 0.762826 w_k1_MSFT_s112
                                 + 0.822732 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.791075 w_k1_ADM_s112
                                 + 0.998945 w_k1_AXP_s112
                                 + 1.21724 w_k1_DIS_s112
                                 + 0.884579 w_k1_FDX_s112
                                 + 1.50702 w_k1_IBM_s112
                                 + 0.797447 w_k1_INTC_s112
                                 + 1.04181 w_k1_KO_s112 + 1.58684 w_k1_MAT_s112
                                 + 0.923649 w_k1_MSFT_s112
                                 + 1.02715 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.753782 w_k1_ADM_s112
                                 + 0.825122 w_k1_AXP_s112
                                 + 0.716511 w_k1_DIS_s112
                                 + 0.743054 w_k1_FDX_s112
                                 + 1.09522 w_k1_IBM_s112
                                 + 0.787986 w_k1_INTC_s112
                                 + 0.81558 w_k1_KO_s112 + 1.0094 w_k1_MAT_s112
                                 + 0.696402 w_k1_MSFT_s112
                                 + 0.942517 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.882446 w_k1_ADM_s112
                                 + 1.02325 w_k1_AXP_s112
                                 + 0.75156 w_k1_DIS_s112
                                 + 0.89134 w_k1_FDX_s112
                                 + 0.835022 w_k1_IBM_s112
                                 + 0.967108 w_k1_INTC_s112
                                 + 0.775284 w_k1_KO_s112
                                 + 0.912548 w_k1_MAT_s112
                                 + 0.867602 w_k1_MSFT_s112
                                 + 0.849873 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.943258 w_k1_ADM_s112
                                 + 1.08151 w_k1_AXP_s112
                                 + 1.26311 w_k1_DIS_s112
                                 + 1.11621 w_k1_FDX_s112
                                 + 1.00399 w_k1_IBM_s112
                                 + 1.0458 w_k1_INTC_s112 + 1.08488 w_k1_KO_s112
                                 + 1.67307 w_k1_MAT_s112
                                 + 1.1764 w_k1_MSFT_s112 + 1.12115 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.911024 w_k1_ADM_s112
                                 + 0.84636 w_k1_AXP_s112
                                 + 0.839652 w_k1_DIS_s112
                                 + 0.906323 w_k1_FDX_s112
                                 + 0.747963 w_k1_IBM_s112
                                 + 0.846365 w_k1_INTC_s112
                                 + 0.750113 w_k1_KO_s112
                                 + 0.705529 w_k1_MAT_s112
                                 + 1.0193 w_k1_MSFT_s112 + 1.01738 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.899789 w_k1_ADM_s112
                                 + 0.977925 w_k1_AXP_s112
                                 + 1.00965 w_k1_DIS_s112
                                 + 0.876424 w_k1_FDX_s112
                                 + 1.14378 w_k1_IBM_s112
                                 + 0.823746 w_k1_INTC_s112
                                 + 1.07997 w_k1_KO_s112 + 1.29323 w_k1_MAT_s112
                                 + 0.955465 w_k1_MSFT_s112
                                 + 1.0299 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.22652 w_k1_ADM_s112
                                 + 1.54505 w_k1_AXP_s112
                                 + 1.38182 w_k1_DIS_s112
                                 + 1.27988 w_k1_FDX_s112
                                 + 1.19773 w_k1_IBM_s112
                                 + 1.11255 w_k1_INTC_s112
                                 + 1.27304 w_k1_KO_s112 + 1.44618 w_k1_MAT_s112
                                 + 1.16579 w_k1_MSFT_s112
                                 + 1.16766 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.768883 w_k1_ADM_s113
                                 + 0.724477 w_k1_AXP_s113
                                 + 0.834721 w_k1_DIS_s113
                                 + 0.790009 w_k1_FDX_s113
                                 + 1.12485 w_k1_IBM_s113
                                 + 0.953807 w_k1_INTC_s113
                                 + 1.03177 w_k1_KO_s113 + 1.01421 w_k1_MAT_s113
                                 + 0.90793 w_k1_MSFT_s113
                                 + 0.931271 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.12929 w_k1_ADM_s113
                                 + 1.1587 w_k1_AXP_s113
                                 + 0.925438 w_k1_DIS_s113
                                 + 1.0023 w_k1_FDX_s113 + 1.32255 w_k1_IBM_s113
                                 + 1.12637 w_k1_INTC_s113
                                 + 1.09227 w_k1_KO_s113
                                 + 0.912082 w_k1_MAT_s113
                                 + 0.999369 w_k1_MSFT_s113
                                 + 0.949589 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.17445 w_k1_ADM_s113
                                 + 1.28065 w_k1_AXP_s113
                                 + 0.981356 w_k1_DIS_s113
                                 + 1.19332 w_k1_FDX_s113
                                 + 0.728192 w_k1_IBM_s113
                                 + 1.1477 w_k1_INTC_s113 + 0.81566 w_k1_KO_s113
                                 + 0.822077 w_k1_MAT_s113
                                 + 1.04935 w_k1_MSFT_s113
                                 + 0.980576 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.983757 w_k1_ADM_s113
                                 + 1.07495 w_k1_AXP_s113
                                 + 0.796531 w_k1_DIS_s113
                                 + 0.956803 w_k1_FDX_s113
                                 + 0.930043 w_k1_IBM_s113
                                 + 1.05306 w_k1_INTC_s113
                                 + 0.801984 w_k1_KO_s113
                                 + 0.987962 w_k1_MAT_s113
                                 + 0.971031 w_k1_MSFT_s113
                                 + 0.929642 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.970769 w_k1_ADM_s113
                                 + 0.783849 w_k1_AXP_s113
                                 + 1.17069 w_k1_DIS_s113
                                 + 1.01697 w_k1_FDX_s113
                                 + 0.94438 w_k1_IBM_s113
                                 + 1.17971 w_k1_INTC_s113
                                 + 0.989926 w_k1_KO_s113
                                 + 0.60223 w_k1_MAT_s113
                                 + 1.08838 w_k1_MSFT_s113
                                 + 0.900536 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.24549 w_k1_ADM_s113
                                 + 1.13413 w_k1_AXP_s113 + 1.3151 w_k1_DIS_s113
                                 + 1.1398 w_k1_FDX_s113 + 1.42745 w_k1_IBM_s113
                                 + 1.10173 w_k1_INTC_s113
                                 + 1.26774 w_k1_KO_s113 + 1.04771 w_k1_MAT_s113
                                 + 1.08015 w_k1_MSFT_s113
                                 + 1.06463 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.08423 w_k1_ADM_s113
                                 + 1.34184 w_k1_AXP_s113
                                 + 1.47025 w_k1_DIS_s113
                                 + 1.30057 w_k1_FDX_s113
                                 + 1.31511 w_k1_IBM_s113
                                 + 1.35544 w_k1_INTC_s113
                                 + 1.34192 w_k1_KO_s113 + 1.4571 w_k1_MAT_s113
                                 + 1.22722 w_k1_MSFT_s113
                                 + 1.18118 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.764058 w_k1_ADM_s113
                                 + 0.773096 w_k1_AXP_s113
                                 + 0.639392 w_k1_DIS_s113
                                 + 0.777565 w_k1_FDX_s113
                                 + 0.75497 w_k1_IBM_s113
                                 + 0.9374 w_k1_INTC_s113
                                 + 0.705928 w_k1_KO_s113
                                 + 0.654083 w_k1_MAT_s113
                                 + 0.688738 w_k1_MSFT_s113
                                 + 0.82124 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12481 w_k1_ADM_s114
                                 + 1.097 w_k1_AXP_s114 + 1.13587 w_k1_DIS_s114
                                 + 1.1115 w_k1_FDX_s114
                                 + 0.996603 w_k1_IBM_s114
                                 + 0.855463 w_k1_INTC_s114
                                 + 0.999177 w_k1_KO_s114
                                 + 1.09906 w_k1_MAT_s114
                                 + 0.985605 w_k1_MSFT_s114
                                 + 1.00774 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.05501 w_k1_ADM_s114
                                 + 1.08404 w_k1_AXP_s114
                                 + 1.03337 w_k1_DIS_s114
                                 + 1.10824 w_k1_FDX_s114
                                 + 0.708961 w_k1_IBM_s114
                                 + 0.938896 w_k1_INTC_s114
                                 + 0.936877 w_k1_KO_s114
                                 + 0.873564 w_k1_MAT_s114
                                 + 1.0528 w_k1_MSFT_s114
                                 + 0.959342 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.88 w_k1_ADM_s114
                                 + 1.29832 w_k1_AXP_s114
                                 + 1.50816 w_k1_DIS_s114
                                 + 1.29843 w_k1_FDX_s114 + 1.6073 w_k1_IBM_s114
                                 + 1.02085 w_k1_INTC_s114
                                 + 1.37225 w_k1_KO_s114 + 1.86326 w_k1_MAT_s114
                                 + 1.26291 w_k1_MSFT_s114
                                 + 1.40596 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.19827 w_k1_ADM_s114
                                 + 0.830609 w_k1_AXP_s114
                                 + 0.909023 w_k1_DIS_s114
                                 + 0.931981 w_k1_FDX_s114
                                 + 1.08164 w_k1_IBM_s114
                                 + 0.807348 w_k1_INTC_s114
                                 + 0.919976 w_k1_KO_s114
                                 + 0.890234 w_k1_MAT_s114
                                 + 0.881422 w_k1_MSFT_s114
                                 + 1.00062 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.16076 w_k1_ADM_s114
                                 + 1.24482 w_k1_AXP_s114
                                 + 1.23055 w_k1_DIS_s114
                                 + 1.20961 w_k1_FDX_s114
                                 + 1.27383 w_k1_IBM_s114
                                 + 1.0918 w_k1_INTC_s114 + 1.16322 w_k1_KO_s114
                                 + 1.36988 w_k1_MAT_s114
                                 + 1.09367 w_k1_MSFT_s114
                                 + 1.14037 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.906687 w_k1_ADM_s114
                                 + 0.69725 w_k1_AXP_s114
                                 + 0.753307 w_k1_DIS_s114
                                 + 0.789998 w_k1_FDX_s114
                                 + 0.738829 w_k1_IBM_s114
                                 + 0.77382 w_k1_INTC_s114
                                 + 0.750769 w_k1_KO_s114
                                 + 0.560962 w_k1_MAT_s114
                                 + 0.841604 w_k1_MSFT_s114
                                 + 0.800712 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.33542 w_k1_ADM_s114
                                 + 1.34745 w_k1_AXP_s114
                                 + 1.42635 w_k1_DIS_s114 + 1.3722 w_k1_FDX_s114
                                 + 0.963435 w_k1_IBM_s114
                                 + 1.28123 w_k1_INTC_s114
                                 + 1.12688 w_k1_KO_s114 + 1.00211 w_k1_MAT_s114
                                 + 1.31958 w_k1_MSFT_s114
                                 + 1.02619 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.17474 w_k1_ADM_s114
                                 + 0.951638 w_k1_AXP_s114
                                 + 1.23114 w_k1_DIS_s114
                                 + 0.981304 w_k1_FDX_s114
                                 + 1.3863 w_k1_IBM_s114
                                 + 0.973255 w_k1_INTC_s114
                                 + 1.26725 w_k1_KO_s114 + 1.42208 w_k1_MAT_s114
                                 + 1.0869 w_k1_MSFT_s114 + 1.05142 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.46527 w_k1_ADM_s115
                                 + 1.61842 w_k1_AXP_s115
                                 + 1.76654 w_k1_DIS_s115
                                 + 1.38077 w_k1_FDX_s115
                                 + 1.27374 w_k1_IBM_s115
                                 + 0.894114 w_k1_INTC_s115
                                 + 1.30823 w_k1_KO_s115 + 1.8239 w_k1_MAT_s115
                                 + 1.17033 w_k1_MSFT_s115
                                 + 1.28827 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.850031 w_k1_ADM_s115
                                 + 0.578918 w_k1_AXP_s115
                                 + 0.655329 w_k1_DIS_s115
                                 + 0.752678 w_k1_FDX_s115
                                 + 0.711985 w_k1_IBM_s115
                                 + 0.813004 w_k1_INTC_s115
                                 + 0.764439 w_k1_KO_s115
                                 + 0.467301 w_k1_MAT_s115
                                 + 0.811456 w_k1_MSFT_s115
                                 + 0.793529 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.25604 w_k1_ADM_s115
                                 + 1.15659 w_k1_AXP_s115
                                 + 1.02067 w_k1_DIS_s115
                                 + 1.21199 w_k1_FDX_s115
                                 + 0.785681 w_k1_IBM_s115
                                 + 1.11648 w_k1_INTC_s115
                                 + 1.03273 w_k1_KO_s115
                                 + 0.882294 w_k1_MAT_s115
                                 + 0.994723 w_k1_MSFT_s115
                                 + 1.04205 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.04148 w_k1_ADM_s115
                                 + 0.784955 w_k1_AXP_s115
                                 + 0.935227 w_k1_DIS_s115
                                 + 0.84695 w_k1_FDX_s115
                                 + 1.07774 w_k1_IBM_s115
                                 + 0.764831 w_k1_INTC_s115
                                 + 1.06236 w_k1_KO_s115 + 1.19595 w_k1_MAT_s115
                                 + 0.863281 w_k1_MSFT_s115
                                 + 0.937373 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.2522 w_k1_ADM_s115
                                 + 1.16929 w_k1_AXP_s115
                                 + 1.22366 w_k1_DIS_s115
                                 + 1.18798 w_k1_FDX_s115
                                 + 1.05336 w_k1_IBM_s115
                                 + 0.982575 w_k1_INTC_s115
                                 + 1.07001 w_k1_KO_s115
                                 + 0.998317 w_k1_MAT_s115
                                 + 1.11039 w_k1_MSFT_s115
                                 + 1.03969 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10277 w_k1_ADM_s115
                                 + 0.950809 w_k1_AXP_s115
                                 + 0.812582 w_k1_DIS_s115
                                 + 0.901156 w_k1_FDX_s115
                                 + 0.884329 w_k1_IBM_s115
                                 + 0.887237 w_k1_INTC_s115
                                 + 0.92613 w_k1_KO_s115
                                 + 0.790632 w_k1_MAT_s115
                                 + 0.807366 w_k1_MSFT_s115
                                 + 0.852162 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.49552 w_k1_ADM_s115
                                 + 1.13907 w_k1_AXP_s115
                                 + 1.14882 w_k1_DIS_s115 + 1.0123 w_k1_FDX_s115
                                 + 1.52944 w_k1_IBM_s115
                                 + 1.19652 w_k1_INTC_s115
                                 + 1.35771 w_k1_KO_s115 + 1.42691 w_k1_MAT_s115
                                 + 1.13801 w_k1_MSFT_s115
                                 + 1.12797 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.34845 w_k1_ADM_s115
                                 + 0.845474 w_k1_AXP_s115
                                 + 1.53221 w_k1_DIS_s115
                                 + 1.03594 w_k1_FDX_s115
                                 + 1.24899 w_k1_IBM_s115
                                 + 0.659345 w_k1_INTC_s115
                                 + 1.30878 w_k1_KO_s115 + 1.20392 w_k1_MAT_s115
                                 + 1.05668 w_k1_MSFT_s115
                                 + 1.06009 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.28443 w_k1_ADM_s116
                                 + 1.10642 w_k1_AXP_s116 + 1.6227 w_k1_DIS_s116
                                 + 1.16442 w_k1_FDX_s116
                                 + 1.34198 w_k1_IBM_s116
                                 + 1.12901 w_k1_INTC_s116
                                 + 1.29796 w_k1_KO_s116 + 1.56075 w_k1_MAT_s116
                                 + 1.33367 w_k1_MSFT_s116
                                 + 1.17476 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.1438 w_k1_ADM_s116
                                 + 1.09136 w_k1_AXP_s116
                                 + 0.898197 w_k1_DIS_s116
                                 + 1.01178 w_k1_FDX_s116
                                 + 0.82491 w_k1_IBM_s116
                                 + 0.897114 w_k1_INTC_s116
                                 + 0.765149 w_k1_KO_s116
                                 + 1.44807 w_k1_MAT_s116
                                 + 0.936819 w_k1_MSFT_s116
                                 + 0.98586 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.38218 w_k1_ADM_s116
                                 + 0.954312 w_k1_AXP_s116
                                 + 1.1478 w_k1_DIS_s116 + 1.1586 w_k1_FDX_s116
                                 + 0.832224 w_k1_IBM_s116
                                 + 1.01673 w_k1_INTC_s116
                                 + 0.958106 w_k1_KO_s116 + 1.1721 w_k1_MAT_s116
                                 + 1.23347 w_k1_MSFT_s116
                                 + 1.18427 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.09093 w_k1_ADM_s116
                                 + 0.782924 w_k1_AXP_s116
                                 + 0.971208 w_k1_DIS_s116
                                 + 0.866885 w_k1_FDX_s116
                                 + 1.07921 w_k1_IBM_s116
                                 + 0.79999 w_k1_INTC_s116
                                 + 1.06393 w_k1_KO_s116 + 1.4625 w_k1_MAT_s116
                                 + 0.924545 w_k1_MSFT_s116
                                 + 0.937095 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.90928 w_k1_ADM_s116
                                 + 0.655696 w_k1_AXP_s116
                                 + 0.756326 w_k1_DIS_s116
                                 + 0.796703 w_k1_FDX_s116
                                 + 0.862099 w_k1_IBM_s116
                                 + 0.747869 w_k1_INTC_s116
                                 + 0.81472 w_k1_KO_s116 + 1.07237 w_k1_MAT_s116
                                 + 0.837486 w_k1_MSFT_s116
                                 + 0.909347 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.1722 w_k1_ADM_s116
                                 + 1.15002 w_k1_AXP_s116
                                 + 1.28812 w_k1_DIS_s116
                                 + 1.13118 w_k1_FDX_s116
                                 + 1.55016 w_k1_IBM_s116
                                 + 0.917334 w_k1_INTC_s116
                                 + 1.15471 w_k1_KO_s116 + 1.80152 w_k1_MAT_s116
                                 + 1.13527 w_k1_MSFT_s116
                                 + 1.23073 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.770841 w_k1_ADM_s116
                                 + 0.578565 w_k1_AXP_s116
                                 + 0.699106 w_k1_DIS_s116
                                 + 0.624283 w_k1_FDX_s116
                                 + 0.846717 w_k1_IBM_s116
                                 + 0.658377 w_k1_INTC_s116
                                 + 0.751455 w_k1_KO_s116
                                 + 0.995409 w_k1_MAT_s116
                                 + 0.737282 w_k1_MSFT_s116
                                 + 0.883675 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.09531 w_k1_ADM_s116
                                 + 1.08039 w_k1_AXP_s116
                                 + 0.94776 w_k1_DIS_s116
                                 + 0.887108 w_k1_FDX_s116
                                 + 1.41908 w_k1_IBM_s116
                                 + 0.905021 w_k1_INTC_s116
                                 + 1.13403 w_k1_KO_s116 + 2.17035 w_k1_MAT_s116
                                 + 1.0735 w_k1_MSFT_s116 + 1.10289 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.997053 w_k1_ADM_s117
                                 + 1.03829 w_k1_AXP_s117
                                 + 1.26933 w_k1_DIS_s117
                                 + 1.26716 w_k1_FDX_s117
                                 + 0.954175 w_k1_IBM_s117
                                 + 1.59446 w_k1_INTC_s117
                                 + 0.977264 w_k1_KO_s117
                                 + 0.695122 w_k1_MAT_s117
                                 + 1.2019 w_k1_MSFT_s117 + 1.02569 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.606966 w_k1_ADM_s117
                                 + 0.802899 w_k1_AXP_s117
                                 + 0.524727 w_k1_DIS_s117
                                 + 0.784532 w_k1_FDX_s117
                                 + 0.632645 w_k1_IBM_s117
                                 + 0.870008 w_k1_INTC_s117
                                 + 0.557529 w_k1_KO_s117
                                 + 0.742177 w_k1_MAT_s117
                                 + 0.773761 w_k1_MSFT_s117
                                 + 0.849205 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.967257 w_k1_ADM_s117
                                 + 1.1298 w_k1_AXP_s117
                                 + 0.896907 w_k1_DIS_s117
                                 + 0.998373 w_k1_FDX_s117
                                 + 0.873969 w_k1_IBM_s117
                                 + 1.2525 w_k1_INTC_s117
                                 + 0.835391 w_k1_KO_s117
                                 + 0.527493 w_k1_MAT_s117
                                 + 0.99338 w_k1_MSFT_s117
                                 + 0.958167 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.531949 w_k1_ADM_s117
                                 + 0.724575 w_k1_AXP_s117
                                 + 0.527169 w_k1_DIS_s117
                                 + 0.65526 w_k1_FDX_s117
                                 + 0.811547 w_k1_IBM_s117
                                 + 1.12297 w_k1_INTC_s117
                                 + 0.644733 w_k1_KO_s117
                                 + 0.685286 w_k1_MAT_s117
                                 + 0.799358 w_k1_MSFT_s117
                                 + 0.788296 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.896215 w_k1_ADM_s117
                                 + 0.93753 w_k1_AXP_s117
                                 + 0.857774 w_k1_DIS_s117
                                 + 0.98337 w_k1_FDX_s117
                                 + 0.683252 w_k1_IBM_s117
                                 + 1.15277 w_k1_INTC_s117
                                 + 0.887884 w_k1_KO_s117
                                 + 0.61121 w_k1_MAT_s117
                                 + 0.935816 w_k1_MSFT_s117
                                 + 0.863799 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.734038 w_k1_ADM_s117
                                 + 0.869434 w_k1_AXP_s117
                                 + 0.682294 w_k1_DIS_s117
                                 + 0.825705 w_k1_FDX_s117
                                 + 1.24124 w_k1_IBM_s117
                                 + 1.29714 w_k1_INTC_s117
                                 + 0.861981 w_k1_KO_s117
                                 + 0.92444 w_k1_MAT_s117
                                 + 0.882432 w_k1_MSFT_s117
                                 + 0.882376 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.941259 w_k1_ADM_s117
                                 + 1.62158 w_k1_AXP_s117
                                 + 1.50158 w_k1_DIS_s117
                                 + 1.19194 w_k1_FDX_s117
                                 + 1.35867 w_k1_IBM_s117
                                 + 1.42067 w_k1_INTC_s117
                                 + 1.25808 w_k1_KO_s117 + 1.44078 w_k1_MAT_s117
                                 + 1.29001 w_k1_MSFT_s117
                                 + 1.24797 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.945848 w_k1_ADM_s117
                                 + 0.861697 w_k1_AXP_s117
                                 + 0.867266 w_k1_DIS_s117
                                 + 0.89844 w_k1_FDX_s117
                                 + 0.955101 w_k1_IBM_s117
                                 + 1.13619 w_k1_INTC_s117 + 1.009 w_k1_KO_s117
                                 + 0.839689 w_k1_MAT_s117
                                 + 1.06881 w_k1_MSFT_s117
                                 + 1.11407 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0118752359084863 w_k1_ADM_s000
                                 + 0.0177487826981469 w_k1_ADM_s001
                                 + 0.0071295344797692 w_k1_ADM_s002
                                 + 0.00488734454621005 w_k1_ADM_s003
                                 + 0.0167300289193247 w_k1_ADM_s004
                                 + 0.0181500438910641 w_k1_ADM_s005
                                 + 0.00542638453052665 w_k1_ADM_s006
                                 + 0.00584287992721201 w_k1_ADM_s007
                                 + 0.0329412874095993 w_k1_ADM_s010
                                 + 0.0143062003054048 w_k1_ADM_s011
                                 + 0.012972004628898 w_k1_ADM_s012
                                 + 0.0225053629370003 w_k1_ADM_s013
                                 + 0.0215109088706347 w_k1_ADM_s014
                                 + 0.0170112890981881 w_k1_ADM_s015
                                 + 0.0121629817193975 w_k1_ADM_s016
                                 + 0.0154801494088281 w_k1_ADM_s017
                                 + 0.148801479832624 w_k1_ADM_s100
                                 + 0.0649983065728011 w_k1_ADM_s101
                                 + 0.0444297492732483 w_k1_ADM_s102
                                 + 0.0497323950952083 w_k1_ADM_s103
                                 + 0.0376443827244228 w_k1_ADM_s104
                                 + 0.0658997179466253 w_k1_ADM_s105
                                 + 0.0538152866280755 w_k1_ADM_s106
                                 + 0.0948655618761704 w_k1_ADM_s107
                                 + 0.0221233449805137 w_k1_ADM_s110
                                 + 0.0259233734199997 w_k1_ADM_s111
                                 + 0.0250015627300131 w_k1_ADM_s112
                                 + 0.0430625432485703 w_k1_ADM_s113
                                 + 0.0358367521501443 w_k1_ADM_s114
                                 + 0.03564368395675 w_k1_ADM_s115
                                 + 0.0286761396046261 w_k1_ADM_s116
                                 + 0.0140282057638314 w_k1_ADM_s117
                                 + 0.0134607167752909 w_k1_AXP_s000
                                 + 0.0207293162163356 w_k1_AXP_s001
                                 + 0.00777924877982321 w_k1_AXP_s002
                                 + 0.00451475477549256 w_k1_AXP_s003
                                 + 0.0161017797221267 w_k1_AXP_s004
                                 + 0.016369005266947 w_k1_AXP_s005
                                 + 0.00473911062047641 w_k1_AXP_s006
                                 + 0.00648086860571965 w_k1_AXP_s007
                                 + 0.0362044977813955 w_k1_AXP_s010
                                 + 0.0137586864597333 w_k1_AXP_s011
                                 + 0.0107884660552946 w_k1_AXP_s012
                                 + 0.0196526279564368 w_k1_AXP_s013
                                 + 0.0238633163190433 w_k1_AXP_s014
                                 + 0.0150540479748787 w_k1_AXP_s015
                                 + 0.0129744686754656 w_k1_AXP_s016
                                 + 0.016746384153052 w_k1_AXP_s017
                                 + 0.163281828392507 w_k1_AXP_s100
                                 + 0.0784887263186351 w_k1_AXP_s101
                                 + 0.042577725960989 w_k1_AXP_s102
                                 + 0.052555808687241 w_k1_AXP_s103
                                 + 0.0368006485605676 w_k1_AXP_s104
                                 + 0.0587027188564025 w_k1_AXP_s105
                                 + 0.0533370370799974 w_k1_AXP_s106
                                 + 0.102685466129666 w_k1_AXP_s107
                                 + 0.0221039975262071 w_k1_AXP_s110
                                 + 0.0260215763711759 w_k1_AXP_s111
                                 + 0.0288823842304318 w_k1_AXP_s112
                                 + 0.0459459205843725 w_k1_AXP_s113
                                 + 0.0346133974201157 w_k1_AXP_s114
                                 + 0.0304348982389657 w_k1_AXP_s115
                                 + 0.0232833510034993 w_k1_AXP_s116
                                 + 0.018036181440465 w_k1_AXP_s117
                                 + 0.0128794638955208 w_k1_DIS_s000
                                 + 0.0188990173508415 w_k1_DIS_s001
                                 + 0.00823359847958065 w_k1_DIS_s002
                                 + 0.00487370180321147 w_k1_DIS_s003
                                 + 0.0157420156420058 w_k1_DIS_s004
                                 + 0.0179005515721954 w_k1_DIS_s005
                                 + 0.0051179508006095 w_k1_DIS_s006
                                 + 0.00627742176643519 w_k1_DIS_s007
                                 + 0.0312906776326551 w_k1_DIS_s010
                                 + 0.0158017745337023 w_k1_DIS_s011
                                 + 0.00984152696848535 w_k1_DIS_s012
                                 + 0.0193684330899818 w_k1_DIS_s013
                                 + 0.0217037078017212 w_k1_DIS_s014
                                 + 0.0156707899354997 w_k1_DIS_s015
                                 + 0.0137691501028684 w_k1_DIS_s016
                                 + 0.0154733937989324 w_k1_DIS_s017
                                 + 0.158152420164571 w_k1_DIS_s100
                                 + 0.0737492200144407 w_k1_DIS_s101
                                 + 0.0458342324746963 w_k1_DIS_s102
                                 + 0.0500664564110951 w_k1_DIS_s103
                                 + 0.0342290001950056 w_k1_DIS_s104
                                 + 0.059257025761525 w_k1_DIS_s105
                                 + 0.0568590490837 w_k1_DIS_s106
                                 + 0.086929556083427 w_k1_DIS_s107
                                 + 0.0212332379591014 w_k1_DIS_s110
                                 + 0.0270234700630261 w_k1_DIS_s111
                                 + 0.0289354201579448 w_k1_DIS_s112
                                 + 0.0431147188795146 w_k1_DIS_s113
                                 + 0.0369676106677536 w_k1_DIS_s114
                                 + 0.0325360882809108 w_k1_DIS_s115
                                 + 0.0258434005248705 w_k1_DIS_s116
                                 + 0.0162822504092179 w_k1_DIS_s117
                                 + 0.0129062565945093 w_k1_FDX_s000
                                 + 0.0181879610200616 w_k1_FDX_s001
                                 + 0.00765759657326054 w_k1_FDX_s002
                                 + 0.00467417660074549 w_k1_FDX_s003
                                 + 0.016254351072321 w_k1_FDX_s004
                                 + 0.0174506135318895 w_k1_FDX_s005
                                 + 0.00505036457014034 w_k1_FDX_s006
                                 + 0.00678129386286777 w_k1_FDX_s007
                                 + 0.0333250810520545 w_k1_FDX_s010
                                 + 0.0139989407888124 w_k1_FDX_s011
                                 + 0.0100556112280722 w_k1_FDX_s012
                                 + 0.0185050956805695 w_k1_FDX_s013
                                 + 0.0211328620588549 w_k1_FDX_s014
                                 + 0.0158446027263874 w_k1_FDX_s015
                                 + 0.0136504575182307 w_k1_FDX_s016
                                 + 0.0153375880203596 w_k1_FDX_s017
                                 + 0.156978291586384 w_k1_FDX_s100
                                 + 0.0735913456106387 w_k1_FDX_s101
                                 + 0.0420367967450593 w_k1_FDX_s102
                                 + 0.0512085042020111 w_k1_FDX_s103
                                 + 0.0358114255587442 w_k1_FDX_s104
                                 + 0.0559358294222218 w_k1_FDX_s105
                                 + 0.0557830763983568 w_k1_FDX_s106
                                 + 0.0922139452550483 w_k1_FDX_s107
                                 + 0.0209440054658121 w_k1_FDX_s110
                                 + 0.0272389726818228 w_k1_FDX_s111
                                 + 0.0267011646623191 w_k1_FDX_s112
                                 + 0.0441394901817373 w_k1_FDX_s113
                                 + 0.0358711847616072 w_k1_FDX_s114
                                 + 0.0311710559545979 w_k1_FDX_s115
                                 + 0.0246888324844213 w_k1_FDX_s116
                                 + 0.0163658104311716 w_k1_FDX_s117
                                 + 0.0132540806703455 w_k1_IBM_s000
                                 + 0.0191637436730516 w_k1_IBM_s001
                                 + 0.00726796340729017 w_k1_IBM_s002
                                 + 0.00456641766449032 w_k1_IBM_s003
                                 + 0.0162463210384996 w_k1_IBM_s004
                                 + 0.018870209648061 w_k1_IBM_s005
                                 + 0.00513639165824705 w_k1_IBM_s006
                                 + 0.00660435673653085 w_k1_IBM_s007
                                 + 0.033538008371175 w_k1_IBM_s010
                                 + 0.0147732915284313 w_k1_IBM_s011
                                 + 0.0106324465653733 w_k1_IBM_s012
                                 + 0.0181224051028261 w_k1_IBM_s013
                                 + 0.0218794668179593 w_k1_IBM_s014
                                 + 0.0150645209082548 w_k1_IBM_s015
                                 + 0.0141138403476905 w_k1_IBM_s016
                                 + 0.0153961722827192 w_k1_IBM_s017
                                 + 0.158047330947276 w_k1_IBM_s100
                                 + 0.0731740383924378 w_k1_IBM_s101
                                 + 0.0440532254241299 w_k1_IBM_s102
                                 + 0.05149725622996 w_k1_IBM_s103
                                 + 0.0353568646660082 w_k1_IBM_s104
                                 + 0.0561798657235285 w_k1_IBM_s105
                                 + 0.0573052308507352 w_k1_IBM_s106
                                 + 0.0944257521507291 w_k1_IBM_s107
                                 + 0.0217484217122645 w_k1_IBM_s110
                                 + 0.0268937492614977 w_k1_IBM_s111
                                 + 0.0287069775027147 w_k1_IBM_s112
                                 + 0.0440493244099693 w_k1_IBM_s113
                                 + 0.0345212153191223 w_k1_IBM_s114
                                 + 0.0305626700162013 w_k1_IBM_s115
                                 + 0.0261348034705383 w_k1_IBM_s116
                                 + 0.0174089029255378 w_k1_IBM_s117
                                 + 0.0116691180457119 w_k1_INTC_s000
                                 + 0.0205186236101189 w_k1_INTC_s001
                                 + 0.00765368662863786 w_k1_INTC_s002
                                 + 0.00451451367202725 w_k1_INTC_s003
                                 + 0.0165016898552558 w_k1_INTC_s004
                                 + 0.0203692697473921 w_k1_INTC_s005
                                 + 0.00536354952624085 w_k1_INTC_s006
                                 + 0.00730137285102907 w_k1_INTC_s007
                                 + 0.0289485243913944 w_k1_INTC_s010
                                 + 0.0121773156147472 w_k1_INTC_s011
                                 + 0.00896362630294716 w_k1_INTC_s012
                                 + 0.0190059797465099 w_k1_INTC_s013
                                 + 0.0189242523517712 w_k1_INTC_s014
                                 + 0.0173231380888586 w_k1_INTC_s015
                                 + 0.0139885142947756 w_k1_INTC_s016
                                 + 0.0153652331734885 w_k1_INTC_s017
                                 + 0.153067441037352 w_k1_INTC_s100
                                 + 0.0753658660920983 w_k1_INTC_s101
                                 + 0.04422431048748 w_k1_INTC_s102
                                 + 0.0479387109513118 w_k1_INTC_s103
                                 + 0.0392959712125703 w_k1_INTC_s104
                                 + 0.0528592085340756 w_k1_INTC_s105
                                 + 0.055526432397298 w_k1_INTC_s106
                                 + 0.0964371677938553 w_k1_INTC_s107
                                 + 0.0218633235427857 w_k1_INTC_s110
                                 + 0.0244869856587564 w_k1_INTC_s111
                                 + 0.0254008247990033 w_k1_INTC_s112
                                 + 0.0473022019620678 w_k1_INTC_s113
                                 + 0.0315599092389357 w_k1_INTC_s114
                                 + 0.0275736776761326 w_k1_INTC_s115
                                 + 0.0225573549573337 w_k1_INTC_s116
                                 + 0.0213147420692057 w_k1_INTC_s117
                                 + 0.0131253145348031 w_k1_KO_s000
                                 + 0.0197124551289954 w_k1_KO_s001
                                 + 0.00868601010776496 w_k1_KO_s002
                                 + 0.00453217037901697 w_k1_KO_s003
                                 + 0.0167541391411539 w_k1_KO_s004
                                 + 0.0187797369587279 w_k1_KO_s005
                                 + 0.00503499080715651 w_k1_KO_s006
                                 + 0.00596806287156629 w_k1_KO_s007
                                 + 0.0313948858427356 w_k1_KO_s010
                                 + 0.0152649307443946 w_k1_KO_s011
                                 + 0.00936266582325971 w_k1_KO_s012
                                 + 0.02043166936248 w_k1_KO_s013
                                 + 0.0222985419854878 w_k1_KO_s014
                                 + 0.0157910269197152 w_k1_KO_s015
                                 + 0.0120791013422692 w_k1_KO_s016
                                 + 0.0144524647954563 w_k1_KO_s017
                                 + 0.155055497002695 w_k1_KO_s100
                                 + 0.0702731309708454 w_k1_KO_s101
                                 + 0.0404112912370863 w_k1_KO_s102
                                 + 0.0471943378096274 w_k1_KO_s103
                                 + 0.0355261483989016 w_k1_KO_s104
                                 + 0.0527350388250767 w_k1_KO_s105
                                 + 0.0548531766683506 w_k1_KO_s106
                                 + 0.096417660090879 w_k1_KO_s107
                                 + 0.0194056865789292 w_k1_KO_s110
                                 + 0.0268076074187718 w_k1_KO_s111
                                 + 0.0269868207558057 w_k1_KO_s112
                                 + 0.04198339156494 w_k1_KO_s113
                                 + 0.0343297636050093 w_k1_KO_s114
                                 + 0.0319097420993507 w_k1_KO_s115
                                 + 0.0243494816029258 w_k1_KO_s116
                                 + 0.0157926484482166 w_k1_KO_s117
                                 + 0.0200114556716297 w_k1_MAT_s000
                                 + 0.0179589932193718 w_k1_MAT_s001
                                 + 0.0088921987765016 w_k1_MAT_s002
                                 + 0.00385772038714774 w_k1_MAT_s003
                                 + 0.0164339998597972 w_k1_MAT_s004
                                 + 0.0152380626437852 w_k1_MAT_s005
                                 + 0.00502661173832576 w_k1_MAT_s006
                                 + 0.00646820550948461 w_k1_MAT_s007
                                 + 0.0342072095274687 w_k1_MAT_s010
                                 + 0.0149706541614501 w_k1_MAT_s011
                                 + 0.00903800274177527 w_k1_MAT_s012
                                 + 0.0186494550043733 w_k1_MAT_s013
                                 + 0.0238957347143478 w_k1_MAT_s014
                                 + 0.013698241507048 w_k1_MAT_s015
                                 + 0.0183214725546646 w_k1_MAT_s016
                                 + 0.0173403826442931 w_k1_MAT_s017
                                 + 0.158905703876601 w_k1_MAT_s100
                                 + 0.0754006803890349 w_k1_MAT_s101
                                 + 0.0422373527824035 w_k1_MAT_s102
                                 + 0.0533908588853878 w_k1_MAT_s103
                                 + 0.0363138963286489 w_k1_MAT_s104
                                 + 0.0458294841783145 w_k1_MAT_s105
                                 + 0.0651155290366471 w_k1_MAT_s106
                                 + 0.103578980896403 w_k1_MAT_s107
                                 + 0.0243370672888401 w_k1_MAT_s110
                                 + 0.034345344138252 w_k1_MAT_s111
                                 + 0.0343392279175618 w_k1_MAT_s112
                                 + 0.0404898430971901 w_k1_MAT_s113
                                 + 0.0355012457587058 w_k1_MAT_s114
                                 + 0.0310568132402477 w_k1_MAT_s115
                                 + 0.0354794890119348 w_k1_MAT_s116
                                 + 0.0154546561239745 w_k1_MAT_s117
                                 + 0.0133854108964425 w_k1_MSFT_s000
                                 + 0.0190083340111513 w_k1_MSFT_s001
                                 + 0.00866092824466164 w_k1_MSFT_s002
                                 + 0.00489462312691263 w_k1_MSFT_s003
                                 + 0.0155730296602563 w_k1_MSFT_s004
                                 + 0.0185053641374026 w_k1_MSFT_s005
                                 + 0.00516408610657001 w_k1_MSFT_s006
                                 + 0.00617673484642054 w_k1_MSFT_s007
                                 + 0.0324152084768992 w_k1_MSFT_s010
                                 + 0.0137007529246038 w_k1_MSFT_s011
                                 + 0.00901644919352388 w_k1_MSFT_s012
                                 + 0.0186552339542208 w_k1_MSFT_s013
                                 + 0.0205397013686294 w_k1_MSFT_s014
                                 + 0.0142247626406338 w_k1_MSFT_s015
                                 + 0.0137938847837139 w_k1_MSFT_s016
                                 + 0.0154297409639356 w_k1_MSFT_s017
                                 + 0.14781116461259 w_k1_MSFT_s100
                                 + 0.0688446297293646 w_k1_MSFT_s101
                                 + 0.0491938736483631 w_k1_MSFT_s102
                                 + 0.0504961130629471 w_k1_MSFT_s103
                                 + 0.0346268824833112 w_k1_MSFT_s104
                                 + 0.0515240169116646 w_k1_MSFT_s105
                                 + 0.0531656664850745 w_k1_MSFT_s106
                                 + 0.0929871294700795 w_k1_MSFT_s107
                                 + 0.0195114011733907 w_k1_MSFT_s110
                                 + 0.0238013917458391 w_k1_MSFT_s111
                                 + 0.0273939597225134 w_k1_MSFT_s112
                                 + 0.0424177413619764 w_k1_MSFT_s113
                                 + 0.0349437310675127 w_k1_MSFT_s114
                                 + 0.0294810060611064 w_k1_MSFT_s115
                                 + 0.0262813210294014 w_k1_MSFT_s116
                                 + 0.0173103430639417 w_k1_MSFT_s117
                                 + 0.0121288575808741 w_k1_PG_s000
                                 + 0.0202220187191536 w_k1_PG_s001
                                 + 0.00761646878636678 w_k1_PG_s002
                                 + 0.00449979314090024 w_k1_PG_s003
                                 + 0.015633866236782 w_k1_PG_s004
                                 + 0.0179261182169883 w_k1_PG_s005
                                 + 0.00561117074529487 w_k1_PG_s006
                                 + 0.00621753086366031 w_k1_PG_s007
                                 + 0.0308645558654072 w_k1_PG_s010
                                 + 0.0138991050891555 w_k1_PG_s011
                                 + 0.00987685695158973 w_k1_PG_s012
                                 + 0.0193366642908027 w_k1_PG_s013
                                 + 0.0212979958544742 w_k1_PG_s014
                                 + 0.0159307189933493 w_k1_PG_s015
                                 + 0.0141929919366952 w_k1_PG_s016
                                 + 0.014884064209308 w_k1_PG_s017
                                 + 0.143264566821153 w_k1_PG_s100
                                 + 0.0728545117951585 w_k1_PG_s101
                                 + 0.0408326883028735 w_k1_PG_s102
                                 + 0.0482471502774283 w_k1_PG_s103
                                 + 0.0344523507647893 w_k1_PG_s104
                                 + 0.052546406442196 w_k1_PG_s105
                                 + 0.0556110975708853 w_k1_PG_s106
                                 + 0.0885599279193287 w_k1_PG_s107
                                 + 0.020107872048989 w_k1_PG_s110
                                 + 0.0249665205545675 w_k1_PG_s111
                                 + 0.0281732451358166 w_k1_PG_s112
                                 + 0.0408442429335727 w_k1_PG_s113
                                 + 0.03427943404747 w_k1_PG_s114
                                 + 0.0300853075149508 w_k1_PG_s115
                                 + 0.0268271526189219 w_k1_PG_s116
                                 + 0.0168350296494862 w_k1_PG_s117 >= 1.209
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
