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
 w_k1_ADM_s0_return_constraint:  1.14682 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.981397 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.796323 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.08286 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.760316 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.976832 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.38079 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.911454 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.38449 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.03333 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.18741 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.08871 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.947866 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.10218 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.849179 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02288 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.25058 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.994766 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.42809 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.99018 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.08624 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.33547 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.42843 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.01127 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.45703 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.09182 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.847113 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.08561 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.931188 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.03533 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.44403 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.950719 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.5562 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.23724 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.00119 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.762641 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.957214 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.10557 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.27956 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.914397 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.884659 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24592 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.685141 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.09638 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.11868 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.910061 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.60132 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.786615 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.3673 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.38389 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.737949 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.02178 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.605208 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.893894 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.3382 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.729947 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.3643 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.927769 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.02208 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.48652 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.6302 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.11555 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.5335 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.22862 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.83447 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.22977 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.19282 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.953892 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.59808 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.79449 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.67448 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.13249 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.06745 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.623958 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.982622 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.08342 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.36512 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.884216 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.815185 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.22547 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.671008 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.20031 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.16898 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.868934 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.77987 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.792278 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.04534 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.55585 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.677976 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.1558 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.577326 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.929889 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.42971 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.720385 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.22902 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.09223 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.05358 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.94385 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.69137 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.02755 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.62647 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.16607 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.755366 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.0665 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.18413 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.967936 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.65579 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.720444 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.81958 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.32474 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.924675 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.531479 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.91736 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.902709 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19486 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.934819 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.86271 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.1512 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.774812 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.07298 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.1941 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.8469 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.4324 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.928629 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.07804 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.14957 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.763164 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.961618 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.788938 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.880035 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.34721 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.851033 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.19025 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.97445 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.15175 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.4149 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.23978 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.09835 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.49612 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.1483 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.841725 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.01914 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.00615 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.89273 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.51719 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.86757 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.39218 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.21343 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.831377 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.678589 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.918141 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.0628 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.27163 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.960323 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.05339 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.978355 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.766636 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.07606 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.903988 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.1604 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.48627 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.633851 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.35601 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.88952 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.866273 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.5078 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.512178 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.15866 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.21759 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.758689 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.78568 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.6894 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.792626 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.28427 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.74769 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.746529 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.858748 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.89844 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.724437 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.43186 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.19304 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.13986 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.11591 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.756092 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.92383 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.875542 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.8472 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.826823 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.29835 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.812718 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13354 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.941423 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.926282 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.20688 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.791559 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.18438 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.13671 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.051 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.3545 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.942713 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.12683 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.05038 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.746955 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.926147 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.610617 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.880439 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.16852 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.835824 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.925943 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.858382 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.09589 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.08696 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.10262 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.03916 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.50648 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.19171 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.995709 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.22153 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01938 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.933071 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.20684 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.726793 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.938432 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.976583 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.869067 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.666644 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.808421 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.951942 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.25439 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.916107 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.921275 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.04378 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.777085 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.10981 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.06716 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.02839 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.60067 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.833573 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.13191 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.42792 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.846579 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.24369 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.628416 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.02954 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.32337 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.821478 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.26627 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.29954 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.974955 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.36258 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.40154 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.845112 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.1037 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.06354 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.761156 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.15019 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.08012 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.00206 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.3504 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.779231 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.59279 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.01781 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.29001 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.732883 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.08338 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.866785 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28246 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.900551 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.901553 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.993018 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.741816 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.05753 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.867244 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.24559 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.94 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.796472 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.84161 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 2.30206 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.975287 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.72561 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.607761 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.20241 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.53824 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.797978 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.94807 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.83455 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.985927 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.42661 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.88839 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.854808 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.8036 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.23117 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.769905 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.37516 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.23024 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.15598 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.4294 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.925418 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.43878 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.20585 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.77203 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.810277 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.55783 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.683514 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21715 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.957786 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.82007 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.07492 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.805831 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.07738 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.17137 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.948947 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.42378 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.930226 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.04607 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.1744 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.818476 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.06047 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.695612 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.05134 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.31485 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.879784 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.07711 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.934331 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.13603 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.42339 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.20576 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.974291 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.29912 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.06868 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.86694 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.01106 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.04842 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.933849 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.2844 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.912866 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.23708 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.15729 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.844876 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.748007 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.888801 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.939091 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15493 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.954372 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.909679 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.0963 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.835414 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.03784 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12523 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.988608 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.14787 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.951534 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.13039 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.37855 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.916805 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.02993 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.719959 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.914303 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.1809 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.879278 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.15989 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.966516 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04208 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.2097 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.23849 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.971255 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.21989 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08024 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.800025 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.15824 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04443 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.944828 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.33425 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.927493 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.33281 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.09951 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.08595 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.829063 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03899 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.962209 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.446997649045435 z_k1_s0000
                                 - 0.0919997538889677 z_k1_s0001
                                 - 0.358998817931513 z_k1_s0002
                                 - 0.381999561526395 z_k1_s0003
                                 - 0.257999939915041 z_k1_s0004
                                 - 0.233998697252184 z_k1_s0005
                                 - 0.211999311522759 z_k1_s0006
                                 - 0.162999547619632 z_k1_s0007
                                 - 0.0990002430025483 z_k1_s0010
                                 - 0.19899995265846 z_k1_s0011
                                 - 0.0990002430025483 z_k1_s0012
                                 - 0.19899995265846 z_k1_s0013
                                 - 0.174000878296405 z_k1_s0014
                                 - 0.439000682755864 z_k1_s0015
                                 - 0.155000309958376 z_k1_s0016
                                 - 0.186999838603415 z_k1_s0017
                                 - 0.150999620396834 z_k1_s0020
                                 - 0.115000043149974 z_k1_s0021
                                 - 0.131999891572112 z_k1_s0022
                                 - 0.0600000450782516 z_k1_s0023
                                 - 0.132999831773404 z_k1_s0024
                                 - 0.04499999055868 z_k1_s0025
                                 - 0.155000246202176 z_k1_s0026
                                 - 0.0739999863964964 z_k1_s0027
                                 - 0.42999936949393 z_k1_s0030
                                 - 0.317999678218628 z_k1_s0031
                                 - 0.328999755589312 z_k1_s0032
                                 - 0.82199774070802 z_k1_s0033
                                 - 0.88399744579804 z_k1_s0034
                                 - 0.241999874656827 z_k1_s0035
                                 - 0.26399966389855 z_k1_s0036
                                 - 0.366000381335807 z_k1_s0037
                                 - 0.0910001885985299 z_k1_s0040
                                 - 0.0480000513596437 z_k1_s0041
                                 - 0.0770002176603192 z_k1_s0042
                                 - 0.145999696927055 z_k1_s0043
                                 - 0.0959998007191593 z_k1_s0044
                                 - 0.141000084806425 z_k1_s0045
                                 - 0.103999784112423 z_k1_s0046
                                 - 0.0520000430562753 z_k1_s0047
                                 - 0.104999972154084 z_k1_s0050
                                 - 0.0839999777232672 z_k1_s0051
                                 - 0.163000152048572 z_k1_s0052
                                 - 0.0779999526859112 z_k1_s0053
                                 - 0.14499951773693 z_k1_s0054
                                 - 0.146999588216001 z_k1_s0055
                                 - 0.127999850664114 z_k1_s0056
                                 - 0.0819999072441958 z_k1_s0057
                                 - 0.20800033842758 z_k1_s0060
                                 - 0.178999803720408 z_k1_s0061
                                 - 0.285999459338994 z_k1_s0062
                                 - 0.205000449623212 z_k1_s0063
                                 - 0.146999782608439 z_k1_s0064
                                 - 0.465998823594619 z_k1_s0065
                                 - 0.310000581771791 z_k1_s0066
                                 - 0.211000227231947 z_k1_s0067
                                 - 0.141000010292103 z_k1_s0070
                                 - 0.174999742518708 z_k1_s0071
                                 - 0.221000230471907 z_k1_s0072
                                 - 0.21600046308554 z_k1_s0073
                                 - 0.139000103337556 z_k1_s0074
                                 - 0.172999835564162 z_k1_s0075
                                 - 0.145999777678469 z_k1_s0076
                                 - 0.102999938556678 z_k1_s0077
                                 - 0.199999747297261 z_k1_s0100
                                 - 0.259000359449979 z_k1_s0101
                                 - 0.265000809668913 z_k1_s0102
                                 - 0.0950000434662051 z_k1_s0103
                                 - 0.33600068725944 z_k1_s0104
                                 - 0.260999964522937 z_k1_s0105
                                 - 0.100000037148637 z_k1_s0106
                                 - 0.119000045841878 z_k1_s0107
                                 - 0.58000157501782 z_k1_s0110
                                 - 0.487000095855387 z_k1_s0111
                                 - 0.217000545639331 z_k1_s0112
                                 - 0.264000697511517 z_k1_s0113
                                 - 0.320000675043891 z_k1_s0114
                                 - 0.247000433174311 z_k1_s0115
                                 - 0.172000292535938 z_k1_s0116
                                 - 0.525001546869844 z_k1_s0117
                                 - 0.293000255840199 z_k1_s0120
                                 - 0.0980000089103362 z_k1_s0121
                                 - 0.106000009637711 z_k1_s0122
                                 - 0.20000001818436 z_k1_s0123
                                 - 0.286999796894444 z_k1_s0124
                                 - 0.180999328856508 z_k1_s0125
                                 - 0.131000088310793 z_k1_s0126
                                 - 0.231999409893557 z_k1_s0127
                                 - 0.455999652134684 z_k1_s0130
                                 - 0.820002303301224 z_k1_s0131
                                 - 0.628002075875277 z_k1_s0132
                                 - 0.363000984978393 z_k1_s0133
                                 - 0.174000428105249 z_k1_s0134
                                 - 0.195000253179526 z_k1_s0135
                                 - 0.591000811045735 z_k1_s0136
                                 - 0.325000540366135 z_k1_s0137
                                 - 0.157000238554713 z_k1_s0140
                                 - 0.0630000423499463 z_k1_s0141
                                 - 0.0840003358001073 z_k1_s0142
                                 - 0.148000112790358 z_k1_s0143
                                 - 0.0660000004713442 z_k1_s0144
                                 - 0.0840003358001073 z_k1_s0145
                                 - 0.173000182802276 z_k1_s0146
                                 - 0.0630000423499463 z_k1_s0147
                                 - 0.145999948366456 z_k1_s0150
                                 - 0.103000057194218 z_k1_s0151
                                 - 0.0960000498149964 z_k1_s0152
                                 - 0.100000036903109 z_k1_s0153
                                 - 0.143000047975442 z_k1_s0154
                                 - 0.260999727024821 z_k1_s0155
                                 - 0.120999579440392 z_k1_s0156
                                 - 0.228999590519727 z_k1_s0157
                                 - 0.576001800384719 z_k1_s0160
                                 - 0.417998126059989 z_k1_s0161
                                 - 0.346000038287201 z_k1_s0162
                                 - 0.711000250458984 z_k1_s0163
                                 - 0.647000050416237 z_k1_s0164
                                 - 0.648998911698661 z_k1_s0165
                                 - 0.250999983064408 z_k1_s0166
                                 - 0.47300141483154 z_k1_s0167
                                 - 0.227000694719852 z_k1_s0170
                                 - 0.286000598153461 z_k1_s0171
                                 - 0.224000079714339 z_k1_s0172
                                 - 0.21699962003546 z_k1_s0173
                                 - 0.0930000461581941 z_k1_s0174
                                 - 0.146000036281657 z_k1_s0175
                                 - 0.146999412249607 z_k1_s0176
                                 - 0.123000051609218 z_k1_s0177
                                 - 0.049999987951768 z_k1_s1000
                                 - 0.231999775328181 z_k1_s1001
                                 - 0.0850000146780104 z_k1_s1002
                                 - 0.086000053113051 z_k1_s1003
                                 - 0.223999995247927 z_k1_s1004
                                 - 0.0569999932970165 z_k1_s1005
                                 - 0.0519999769218373 z_k1_s1006
                                 - 0.0929999705582876 z_k1_s1007
                                 - 0.145000172711414 z_k1_s1010
                                 - 0.374000485018421 z_k1_s1011
                                 - 0.238000156620754 z_k1_s1012
                                 - 0.215000050140982 z_k1_s1013
                                 - 0.60699974598671 z_k1_s1014
                                 - 0.172999980343185 z_k1_s1015
                                 - 0.35500076060049 z_k1_s1016
                                 - 0.282000059099331 z_k1_s1017
                                 - 0.0690000453980942 z_k1_s1020
                                 - 0.239000655323907 z_k1_s1021
                                 - 0.103000064023242 z_k1_s1022
                                 - 0.2779996810879 z_k1_s1023
                                 - 0.109999991857819 z_k1_s1024
                                 - 0.124999984874794 z_k1_s1025
                                 - 0.13899984054395 z_k1_s1026
                                 - 0.229000573845912 z_k1_s1027
                                 - 0.091000113314147 z_k1_s1030
                                 - 0.201000045018953 z_k1_s1031
                                 - 0.125000138255679 z_k1_s1032
                                 - 0.102000074824596 z_k1_s1033
                                 - 0.341999580434732 z_k1_s1034
                                 - 0.3210010642253 z_k1_s1035
                                 - 0.13800006398618 z_k1_s1036
                                 - 0.262999885641539 z_k1_s1037
                                 - 1.0079999888947 z_k1_s1040
                                 - 0.62699959442798 z_k1_s1041
                                 - 0.383000588204083 z_k1_s1042
                                 - 0.465998906041043 z_k1_s1043
                                 - 0.288000243741692 z_k1_s1044
                                 - 0.43100048479416 z_k1_s1045
                                 - 0.816000402534385 z_k1_s1046
                                 - 0.302000303601426 z_k1_s1047
                                 - 0.106999988670617 z_k1_s1050
                                 - 0.0889999308905483 z_k1_s1051
                                 - 0.216999099971769 z_k1_s1052
                                 - 0.435999270762917 z_k1_s1053
                                 - 0.264999963651533 z_k1_s1054
                                 - 0.412999374332724 z_k1_s1055
                                 - 0.106999988670617 z_k1_s1056
                                 - 0.139999917200847 z_k1_s1057
                                 - 0.282999679567354 z_k1_s1060
                                 - 0.242998893309793 z_k1_s1061
                                 - 0.115999609756055 z_k1_s1062
                                 - 0.280998555408184 z_k1_s1063
                                 - 0.213998448447634 z_k1_s1064
                                 - 0.0759996579956414 z_k1_s1065
                                 - 0.297998931292505 z_k1_s1066
                                 - 0.157999517379632 z_k1_s1067
                                 - 0.283998605893383 z_k1_s1070
                                 - 0.229000361847625 z_k1_s1071
                                 - 0.194999569059526 z_k1_s1072
                                 - 0.416999315157581 z_k1_s1073
                                 - 0.341999311827339 z_k1_s1074
                                 - 0.262998956640199 z_k1_s1075
                                 - 0.0949997844187549 z_k1_s1076
                                 - 0.372999840722765 z_k1_s1077
                                 - 0.529999727085602 z_k1_s1100
                                 - 0.481001707971194 z_k1_s1101
                                 - 0.297999939676393 z_k1_s1102
                                 - 0.653001805655498 z_k1_s1103
                                 - 0.308999815487654 z_k1_s1104
                                 - 1.09599863982641 z_k1_s1105
                                 - 0.550998878134634 z_k1_s1106
                                 - 0.569000714461283 z_k1_s1107
                                 - 0.974999779848703 z_k1_s1110
                                 - 0.71300105625605 z_k1_s1111
                                 - 0.625000482467918 z_k1_s1112
                                 - 1.44100076216958 z_k1_s1113
                                 - 1.88400472296349 z_k1_s1114
                                 - 1.24100255338235 z_k1_s1115
                                 - 0.957001011139226 z_k1_s1116
                                 - 1.89200114923048 z_k1_s1117
                                 - 0.513998611408281 z_k1_s1120
                                 - 0.307999475326301 z_k1_s1121
                                 - 0.162000005235205 z_k1_s1122
                                 - 0.58500091296028 z_k1_s1123
                                 - 0.239999834363302 z_k1_s1124
                                 - 0.34199958840774 z_k1_s1125
                                 - 0.287999976795926 z_k1_s1126
                                 - 0.487000414902047 z_k1_s1127
                                 - 1.9109976946751 z_k1_s1130
                                 - 0.689000048110026 z_k1_s1131
                                 - 0.572000111668698 z_k1_s1132
                                 - 1.79900256437889 z_k1_s1133
                                 - 0.88199966708078 z_k1_s1134
                                 - 1.82399808310589 z_k1_s1135
                                 - 0.40700022567195 z_k1_s1136
                                 - 1.42000207168094 z_k1_s1137
                                 - 0.441999298118483 z_k1_s1140
                                 - 0.197999835003244 z_k1_s1141
                                 - 0.377999685006193 z_k1_s1142
                                 - 0.125999895002064 z_k1_s1143
                                 - 0.43499863684085 z_k1_s1144
                                 - 0.341999715005604 z_k1_s1145
                                 - 0.601999831898623 z_k1_s1146
                                 - 0.47899893373033 z_k1_s1147
                                 - 0.603998076058273 z_k1_s1150
                                 - 0.287999823630403 z_k1_s1151
                                 - 0.75899900540098 z_k1_s1152
                                 - 0.860001875062905 z_k1_s1153
                                 - 0.530998721193596 z_k1_s1154
                                 - 0.922000212400076 z_k1_s1155
                                 - 0.380999872636049 z_k1_s1156
                                 - 0.740999652174053 z_k1_s1157
                                 - 1.23100439345814 z_k1_s1160
                                 - 0.562000516711596 z_k1_s1161
                                 - 1.39700495641348 z_k1_s1162
                                 - 1.03900155461656 z_k1_s1163
                                 - 2.08700040511415 z_k1_s1164
                                 - 0.836000406763592 z_k1_s1165
                                 - 1.40199841023242 z_k1_s1166
                                 - 0.525000839714664 z_k1_s1167
                                 - 0.968999605440031 z_k1_s1170
                                 - 1.09900178616039 z_k1_s1171
                                 - 0.627999069864681 z_k1_s1172
                                 - 0.957999053267072 z_k1_s1173
                                 - 0.675999439875632 z_k1_s1174
                                 - 1.73300151190094 z_k1_s1175
                                 - 2.78499881063316 z_k1_s1176
                                 - 0.905999546537808 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.776207 w_k1_ADM_s000
                                 + 0.847115 w_k1_AXP_s000
                                 + 0.785213 w_k1_DIS_s000
                                 + 0.788222 w_k1_FDX_s000
                                 + 0.806961 w_k1_IBM_s000
                                 + 0.759183 w_k1_INTC_s000
                                 + 0.78054 w_k1_KO_s000
                                 + 0.869317 w_k1_MAT_s000
                                 + 0.884478 w_k1_MSFT_s000
                                 + 0.866223 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.938188 w_k1_ADM_s000
                                 + 1.13712 w_k1_AXP_s000
                                 + 0.831288 w_k1_DIS_s000
                                 + 1.05018 w_k1_FDX_s000
                                 + 0.789349 w_k1_IBM_s000
                                 + 0.886674 w_k1_INTC_s000
                                 + 0.78631 w_k1_KO_s000 + 1.50232 w_k1_MAT_s000
                                 + 0.872093 w_k1_MSFT_s000
                                 + 0.942404 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.1842 w_k1_ADM_s000
                                 + 1.3453 w_k1_AXP_s000 + 1.10198 w_k1_DIS_s000
                                 + 0.969362 w_k1_FDX_s000
                                 + 1.4249 w_k1_IBM_s000
                                 + 1.07188 w_k1_INTC_s000
                                 + 1.26873 w_k1_KO_s000 + 2.11241 w_k1_MAT_s000
                                 + 1.00389 w_k1_MSFT_s000 + 1.0247 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.05625 w_k1_ADM_s000
                                 + 1.42628 w_k1_AXP_s000 + 1.5606 w_k1_DIS_s000
                                 + 1.23568 w_k1_FDX_s000
                                 + 1.21602 w_k1_IBM_s000
                                 + 1.09896 w_k1_INTC_s000
                                 + 1.25005 w_k1_KO_s000 + 1.80398 w_k1_MAT_s000
                                 + 1.04041 w_k1_MSFT_s000
                                 + 1.08601 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.901042 w_k1_ADM_s000
                                 + 0.874871 w_k1_AXP_s000
                                 + 1.22826 w_k1_DIS_s000
                                 + 1.03007 w_k1_FDX_s000
                                 + 0.865997 w_k1_IBM_s000
                                 + 1.1619 w_k1_INTC_s000
                                 + 0.993506 w_k1_KO_s000
                                 + 0.970277 w_k1_MAT_s000
                                 + 1.14463 w_k1_MSFT_s000
                                 + 0.995529 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.88978 w_k1_ADM_s000
                                 + 0.906816 w_k1_AXP_s000
                                 + 0.760252 w_k1_DIS_s000
                                 + 0.826562 w_k1_FDX_s000
                                 + 1.15044 w_k1_IBM_s000
                                 + 0.662516 w_k1_INTC_s000
                                 + 0.930966 w_k1_KO_s000
                                 + 1.45245 w_k1_MAT_s000
                                 + 0.770387 w_k1_MSFT_s000
                                 + 0.884942 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.11546 w_k1_ADM_s000
                                 + 1.17995 w_k1_AXP_s000
                                 + 1.16355 w_k1_DIS_s000
                                 + 1.16749 w_k1_FDX_s000
                                 + 0.810753 w_k1_IBM_s000
                                 + 1.03757 w_k1_INTC_s000
                                 + 1.00415 w_k1_KO_s000 + 1.21038 w_k1_MAT_s000
                                 + 1.11485 w_k1_MSFT_s000
                                 + 1.00009 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.840581 w_k1_ADM_s000
                                 + 1.26066 w_k1_AXP_s000
                                 + 0.954049 w_k1_DIS_s000
                                 + 0.948013 w_k1_FDX_s000
                                 + 1.1461 w_k1_IBM_s000
                                 + 1.04122 w_k1_INTC_s000
                                 + 0.949818 w_k1_KO_s000
                                 + 1.50254 w_k1_MAT_s000
                                 + 0.948326 w_k1_MSFT_s000
                                 + 1.01161 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.30733 w_k1_ADM_s001
                                 + 0.95272 w_k1_AXP_s001
                                 + 1.12006 w_k1_DIS_s001
                                 + 1.17236 w_k1_FDX_s001
                                 + 0.733102 w_k1_IBM_s001
                                 + 0.85001 w_k1_INTC_s001
                                 + 0.839744 w_k1_KO_s001
                                 + 0.716306 w_k1_MAT_s001
                                 + 0.987421 w_k1_MSFT_s001
                                 + 0.912397 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.18377 w_k1_ADM_s001
                                 + 0.699694 w_k1_AXP_s001
                                 + 0.926899 w_k1_DIS_s001
                                 + 0.996605 w_k1_FDX_s001
                                 + 0.699872 w_k1_IBM_s001
                                 + 0.789454 w_k1_INTC_s001
                                 + 0.893458 w_k1_KO_s001
                                 + 0.925569 w_k1_MAT_s001
                                 + 1.01983 w_k1_MSFT_s001
                                 + 0.944691 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.2114 w_k1_ADM_s001
                                 + 1.07667 w_k1_AXP_s001
                                 + 1.45255 w_k1_DIS_s001
                                 + 1.22135 w_k1_FDX_s001
                                 + 1.44157 w_k1_IBM_s001
                                 + 1.03842 w_k1_INTC_s001
                                 + 1.38901 w_k1_KO_s001 + 1.51988 w_k1_MAT_s001
                                 + 1.34305 w_k1_MSFT_s001
                                 + 1.36522 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.974232 w_k1_ADM_s001
                                 + 0.592644 w_k1_AXP_s001
                                 + 0.583355 w_k1_DIS_s001
                                 + 0.742943 w_k1_FDX_s001
                                 + 0.817552 w_k1_IBM_s001
                                 + 0.54793 w_k1_INTC_s001
                                 + 0.695814 w_k1_KO_s001
                                 + 0.88148 w_k1_MAT_s001
                                 + 0.71075 w_k1_MSFT_s001
                                 + 0.819096 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.1082 w_k1_ADM_s001
                                 + 1.3504 w_k1_AXP_s001 + 1.11986 w_k1_DIS_s001
                                 + 1.13165 w_k1_FDX_s001
                                 + 1.37078 w_k1_IBM_s001
                                 + 0.897036 w_k1_INTC_s001
                                 + 1.21871 w_k1_KO_s001 + 1.94351 w_k1_MAT_s001
                                 + 1.03068 w_k1_MSFT_s001
                                 + 1.08532 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.06777 w_k1_ADM_s001
                                 + 0.91029 w_k1_AXP_s001
                                 + 1.05462 w_k1_DIS_s001
                                 + 0.971012 w_k1_FDX_s001
                                 + 1.00491 w_k1_IBM_s001
                                 + 0.949988 w_k1_INTC_s001
                                 + 0.989876 w_k1_KO_s001
                                 + 1.11851 w_k1_MAT_s001
                                 + 0.911222 w_k1_MSFT_s001
                                 + 0.990435 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.00789 w_k1_ADM_s001
                                 + 1.0382 w_k1_AXP_s001
                                 + 0.968342 w_k1_DIS_s001
                                 + 0.925373 w_k1_FDX_s001
                                 + 1.41172 w_k1_IBM_s001
                                 + 0.885687 w_k1_INTC_s001
                                 + 1.08788 w_k1_KO_s001 + 1.16106 w_k1_MAT_s001
                                 + 0.874697 w_k1_MSFT_s001
                                 + 0.896824 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.3593 w_k1_ADM_s001
                                 + 1.33142 w_k1_AXP_s001
                                 + 1.39765 w_k1_DIS_s001
                                 + 1.38114 w_k1_FDX_s001
                                 + 1.09409 w_k1_IBM_s001
                                 + 0.984558 w_k1_INTC_s001
                                 + 1.23896 w_k1_KO_s001 + 1.45949 w_k1_MAT_s001
                                 + 1.22789 w_k1_MSFT_s001
                                 + 1.21586 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.38646 w_k1_ADM_s002
                                 + 1.54568 w_k1_AXP_s002
                                 + 1.59781 w_k1_DIS_s002
                                 + 1.21672 w_k1_FDX_s002 + 1.573 w_k1_IBM_s002
                                 + 1.18685 w_k1_INTC_s002
                                 + 1.54164 w_k1_KO_s002 + 2.08511 w_k1_MAT_s002
                                 + 1.33573 w_k1_MSFT_s002
                                 + 1.43099 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.33589 w_k1_ADM_s002
                                 + 1.22205 w_k1_AXP_s002
                                 + 1.02792 w_k1_DIS_s002
                                 + 1.08052 w_k1_FDX_s002
                                 + 0.956987 w_k1_IBM_s002
                                 + 0.892253 w_k1_INTC_s002
                                 + 0.978856 w_k1_KO_s002
                                 + 1.01476 w_k1_MAT_s002
                                 + 0.939784 w_k1_MSFT_s002
                                 + 1.10393 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.3229 w_k1_ADM_s002
                                 + 0.912104 w_k1_AXP_s002
                                 + 0.882428 w_k1_DIS_s002
                                 + 1.03906 w_k1_FDX_s002
                                 + 0.744275 w_k1_IBM_s002
                                 + 1.02007 w_k1_INTC_s002
                                 + 0.796844 w_k1_KO_s002
                                 + 0.696574 w_k1_MAT_s002
                                 + 1.0412 w_k1_MSFT_s002 + 1.06495 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.785527 w_k1_ADM_s002
                                 + 0.723598 w_k1_AXP_s002
                                 + 0.784639 w_k1_DIS_s002
                                 + 0.790626 w_k1_FDX_s002
                                 + 0.789724 w_k1_IBM_s002
                                 + 0.851581 w_k1_INTC_s002
                                 + 0.762458 w_k1_KO_s002
                                 + 0.639646 w_k1_MAT_s002
                                 + 0.792913 w_k1_MSFT_s002
                                 + 0.852453 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.964288 w_k1_ADM_s002
                                 + 0.794914 w_k1_AXP_s002
                                 + 0.824981 w_k1_DIS_s002
                                 + 0.907096 w_k1_FDX_s002
                                 + 1.02519 w_k1_IBM_s002
                                 + 1.14249 w_k1_INTC_s002
                                 + 0.896786 w_k1_KO_s002
                                 + 0.837495 w_k1_MAT_s002
                                 + 0.956643 w_k1_MSFT_s002
                                 + 1.05936 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.35321 w_k1_ADM_s002
                                 + 1.29997 w_k1_AXP_s002
                                 + 1.58486 w_k1_DIS_s002
                                 + 1.46825 w_k1_FDX_s002
                                 + 0.769762 w_k1_IBM_s002
                                 + 1.23872 w_k1_INTC_s002
                                 + 1.02591 w_k1_KO_s002
                                 + 0.888737 w_k1_MAT_s002
                                 + 1.5092 w_k1_MSFT_s002 + 1.51174 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.36081 w_k1_ADM_s002
                                 + 1.10401 w_k1_AXP_s002
                                 + 1.22914 w_k1_DIS_s002
                                 + 1.05382 w_k1_FDX_s002
                                 + 1.16468 w_k1_IBM_s002
                                 + 1.15805 w_k1_INTC_s002
                                 + 0.999243 w_k1_KO_s002
                                 + 0.762412 w_k1_MAT_s002
                                 + 1.00504 w_k1_MSFT_s002
                                 + 1.24849 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.817555 w_k1_ADM_s002
                                 + 0.744256 w_k1_AXP_s002
                                 + 0.614332 w_k1_DIS_s002
                                 + 0.789395 w_k1_FDX_s002
                                 + 0.609326 w_k1_IBM_s002
                                 + 0.69586 w_k1_INTC_s002
                                 + 0.585204 w_k1_KO_s002
                                 + 0.46492 w_k1_MAT_s002
                                 + 0.70218 w_k1_MSFT_s002
                                 + 0.937843 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.04896 w_k1_ADM_s003
                                 + 0.948663 w_k1_AXP_s003
                                 + 1.12857 w_k1_DIS_s003
                                 + 1.23983 w_k1_FDX_s003
                                 + 0.919062 w_k1_IBM_s003
                                 + 1.10438 w_k1_INTC_s003
                                 + 0.922354 w_k1_KO_s003
                                 + 0.942609 w_k1_MAT_s003
                                 + 1.24877 w_k1_MSFT_s003
                                 + 0.980716 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.791407 w_k1_ADM_s003
                                 + 1.08844 w_k1_AXP_s003
                                 + 0.755415 w_k1_DIS_s003
                                 + 0.895559 w_k1_FDX_s003
                                 + 1.27265 w_k1_IBM_s003
                                 + 0.893345 w_k1_INTC_s003
                                 + 0.867877 w_k1_KO_s003
                                 + 1.36956 w_k1_MAT_s003
                                 + 0.876476 w_k1_MSFT_s003
                                 + 0.897815 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.15985 w_k1_ADM_s003
                                 + 1.20447 w_k1_AXP_s003
                                 + 1.31188 w_k1_DIS_s003
                                 + 1.14578 w_k1_FDX_s003
                                 + 1.14021 w_k1_IBM_s003
                                 + 1.07627 w_k1_INTC_s003
                                 + 1.07975 w_k1_KO_s003 + 1.54493 w_k1_MAT_s003
                                 + 1.2115 w_k1_MSFT_s003 + 0.98429 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.914119 w_k1_ADM_s003
                                 + 0.779402 w_k1_AXP_s003
                                 + 0.712133 w_k1_DIS_s003
                                 + 0.798104 w_k1_FDX_s003
                                 + 0.64289 w_k1_IBM_s003
                                 + 0.824749 w_k1_INTC_s003
                                 + 0.785262 w_k1_KO_s003
                                 + 0.790208 w_k1_MAT_s003
                                 + 0.861969 w_k1_MSFT_s003
                                 + 0.837578 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.888969 w_k1_ADM_s003
                                 + 0.9155 w_k1_AXP_s003 + 0.97675 w_k1_DIS_s003
                                 + 0.944392 w_k1_FDX_s003
                                 + 0.914351 w_k1_IBM_s003
                                 + 0.911606 w_k1_INTC_s003
                                 + 0.882489 w_k1_KO_s003
                                 + 0.805477 w_k1_MAT_s003
                                 + 1.01395 w_k1_MSFT_s003
                                 + 0.933968 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.862017 w_k1_ADM_s003
                                 + 1.24509 w_k1_AXP_s003
                                 + 1.21368 w_k1_DIS_s003
                                 + 0.987794 w_k1_FDX_s003
                                 + 1.57601 w_k1_IBM_s003
                                 + 1.20251 w_k1_INTC_s003
                                 + 1.14028 w_k1_KO_s003 + 1.7592 w_k1_MAT_s003
                                 + 1.18894 w_k1_MSFT_s003
                                 + 1.01471 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.31058 w_k1_ADM_s003
                                 + 1.83244 w_k1_AXP_s003
                                 + 1.84474 w_k1_DIS_s003
                                 + 1.35372 w_k1_FDX_s003
                                 + 1.56844 w_k1_IBM_s003
                                 + 1.38841 w_k1_INTC_s003
                                 + 1.35732 w_k1_KO_s003 + 1.65857 w_k1_MAT_s003
                                 + 1.48883 w_k1_MSFT_s003
                                 + 1.24427 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.42945 w_k1_ADM_s003
                                 + 1.5612 w_k1_AXP_s003 + 1.51143 w_k1_DIS_s003
                                 + 1.4084 w_k1_FDX_s003 + 1.13408 w_k1_IBM_s003
                                 + 1.17847 w_k1_INTC_s003
                                 + 1.16422 w_k1_KO_s003 + 1.24075 w_k1_MAT_s003
                                 + 1.33042 w_k1_MSFT_s003
                                 + 1.22139 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.10374 w_k1_ADM_s004
                                 + 1.13744 w_k1_AXP_s004
                                 + 1.18487 w_k1_DIS_s004
                                 + 1.03343 w_k1_FDX_s004
                                 + 1.01277 w_k1_IBM_s004
                                 + 0.971236 w_k1_INTC_s004
                                 + 1.1211 w_k1_KO_s004 + 0.977425 w_k1_MAT_s004
                                 + 1.12556 w_k1_MSFT_s004
                                 + 1.00272 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.41682 w_k1_ADM_s004
                                 + 1.36057 w_k1_AXP_s004
                                 + 0.925837 w_k1_DIS_s004
                                 + 0.980244 w_k1_FDX_s004
                                 + 1.03949 w_k1_IBM_s004
                                 + 0.947423 w_k1_INTC_s004
                                 + 1.09996 w_k1_KO_s004 + 1.07096 w_k1_MAT_s004
                                 + 0.915497 w_k1_MSFT_s004
                                 + 1.02339 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.620812 w_k1_ADM_s004
                                 + 1.18707 w_k1_AXP_s004
                                 + 1.07855 w_k1_DIS_s004
                                 + 0.866651 w_k1_FDX_s004
                                 + 1.4801 w_k1_IBM_s004 + 1.2858 w_k1_INTC_s004
                                 + 1.23817 w_k1_KO_s004 + 1.58064 w_k1_MAT_s004
                                 + 0.994658 w_k1_MSFT_s004
                                 + 1.04471 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.959292 w_k1_ADM_s004
                                 + 0.895728 w_k1_AXP_s004
                                 + 0.992322 w_k1_DIS_s004
                                 + 0.97394 w_k1_FDX_s004
                                 + 0.678649 w_k1_IBM_s004
                                 + 0.86296 w_k1_INTC_s004
                                 + 0.803633 w_k1_KO_s004
                                 + 0.635343 w_k1_MAT_s004
                                 + 1.08551 w_k1_MSFT_s004
                                 + 0.875608 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.39109 w_k1_ADM_s004
                                 + 1.38255 w_k1_AXP_s004
                                 + 1.46767 w_k1_DIS_s004
                                 + 1.35811 w_k1_FDX_s004
                                 + 1.08121 w_k1_IBM_s004
                                 + 1.16276 w_k1_INTC_s004
                                 + 1.09081 w_k1_KO_s004
                                 + 0.945919 w_k1_MAT_s004
                                 + 1.14328 w_k1_MSFT_s004
                                 + 1.05008 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.05757 w_k1_ADM_s004
                                 + 0.926853 w_k1_AXP_s004
                                 + 0.916516 w_k1_DIS_s004
                                 + 0.972304 w_k1_FDX_s004
                                 + 0.890007 w_k1_IBM_s004
                                 + 0.949768 w_k1_INTC_s004
                                 + 1.05148 w_k1_KO_s004 + 1.01663 w_k1_MAT_s004
                                 + 0.947206 w_k1_MSFT_s004
                                 + 0.938823 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.838881 w_k1_ADM_s004
                                 + 0.708605 w_k1_AXP_s004
                                 + 0.601285 w_k1_DIS_s004
                                 + 0.714521 w_k1_FDX_s004
                                 + 0.830937 w_k1_IBM_s004
                                 + 0.736222 w_k1_INTC_s004
                                 + 0.719886 w_k1_KO_s004
                                 + 0.638077 w_k1_MAT_s004
                                 + 0.779421 w_k1_MSFT_s004
                                 + 0.82013 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.35418 w_k1_ADM_s004
                                 + 1.39388 w_k1_AXP_s004
                                 + 1.54273 w_k1_DIS_s004
                                 + 1.22121 w_k1_FDX_s004
                                 + 1.37679 w_k1_IBM_s004
                                 + 0.992254 w_k1_INTC_s004
                                 + 1.33303 w_k1_KO_s004 + 1.35396 w_k1_MAT_s004
                                 + 1.21234 w_k1_MSFT_s004
                                 + 1.22522 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.04338 w_k1_ADM_s005
                                 + 0.722797 w_k1_AXP_s005
                                 + 0.632748 w_k1_DIS_s005
                                 + 0.74402 w_k1_FDX_s005
                                 + 0.681737 w_k1_IBM_s005
                                 + 0.755749 w_k1_INTC_s005
                                 + 0.856887 w_k1_KO_s005
                                 + 0.748053 w_k1_MAT_s005
                                 + 0.7806 w_k1_MSFT_s005
                                 + 0.771991 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.14747 w_k1_ADM_s005
                                 + 1.10357 w_k1_AXP_s005
                                 + 1.05971 w_k1_DIS_s005
                                 + 0.987137 w_k1_FDX_s005
                                 + 1.78599 w_k1_IBM_s005
                                 + 1.28863 w_k1_INTC_s005
                                 + 1.24231 w_k1_KO_s005 + 1.59809 w_k1_MAT_s005
                                 + 1.02915 w_k1_MSFT_s005
                                 + 1.02746 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.74195 w_k1_ADM_s005
                                 + 1.2754 w_k1_AXP_s005 + 1.29522 w_k1_DIS_s005
                                 + 1.2444 w_k1_FDX_s005 + 1.09014 w_k1_IBM_s005
                                 + 1.23924 w_k1_INTC_s005
                                 + 1.12775 w_k1_KO_s005 + 1.02797 w_k1_MAT_s005
                                 + 1.25804 w_k1_MSFT_s005
                                 + 0.983312 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.79916 w_k1_ADM_s005
                                 + 1.48233 w_k1_AXP_s005
                                 + 1.09307 w_k1_DIS_s005
                                 + 1.17158 w_k1_FDX_s005
                                 + 1.21159 w_k1_IBM_s005
                                 + 1.09558 w_k1_INTC_s005
                                 + 1.18514 w_k1_KO_s005 + 1.29021 w_k1_MAT_s005
                                 + 1.0916 w_k1_MSFT_s005
                                 + 0.936751 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.15997 w_k1_ADM_s005
                                 + 1.02048 w_k1_AXP_s005
                                 + 0.938263 w_k1_DIS_s005
                                 + 1.09631 w_k1_FDX_s005
                                 + 0.760907 w_k1_IBM_s005
                                 + 1.18498 w_k1_INTC_s005
                                 + 0.965723 w_k1_KO_s005
                                 + 0.606316 w_k1_MAT_s005
                                 + 1.07547 w_k1_MSFT_s005
                                 + 0.926229 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.2093 w_k1_ADM_s005
                                 + 0.85291 w_k1_AXP_s005
                                 + 0.816129 w_k1_DIS_s005
                                 + 0.945067 w_k1_FDX_s005
                                 + 0.950556 w_k1_IBM_s005
                                 + 0.892989 w_k1_INTC_s005
                                 + 0.993176 w_k1_KO_s005
                                 + 0.704162 w_k1_MAT_s005
                                 + 0.915266 w_k1_MSFT_s005
                                 + 0.851255 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.17944 w_k1_ADM_s005
                                 + 0.972423 w_k1_AXP_s005
                                 + 0.937056 w_k1_DIS_s005
                                 + 0.849408 w_k1_FDX_s005
                                 + 1.31422 w_k1_IBM_s005
                                 + 1.09223 w_k1_INTC_s005
                                 + 1.17768 w_k1_KO_s005 + 1.05455 w_k1_MAT_s005
                                 + 1.12976 w_k1_MSFT_s005
                                 + 0.813977 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.24648 w_k1_ADM_s005
                                 + 0.79614 w_k1_AXP_s005
                                 + 1.16032 w_k1_DIS_s005
                                 + 1.14629 w_k1_FDX_s005
                                 + 0.633715 w_k1_IBM_s005
                                 + 1.25594 w_k1_INTC_s005 + 1.1941 w_k1_KO_s005
                                 + 0.817147 w_k1_MAT_s005
                                 + 1.34193 w_k1_MSFT_s005
                                 + 1.00221 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.858544 w_k1_ADM_s006
                                 + 0.638429 w_k1_AXP_s006
                                 + 0.639213 w_k1_DIS_s006
                                 + 0.803304 w_k1_FDX_s006
                                 + 0.604249 w_k1_IBM_s006
                                 + 0.877077 w_k1_INTC_s006
                                 + 0.597497 w_k1_KO_s006
                                 + 0.449302 w_k1_MAT_s006
                                 + 0.919524 w_k1_MSFT_s006
                                 + 0.707932 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.33308 w_k1_ADM_s006
                                 + 1.36396 w_k1_AXP_s006
                                 + 1.21069 w_k1_DIS_s006
                                 + 1.31457 w_k1_FDX_s006
                                 + 0.841464 w_k1_IBM_s006
                                 + 1.2933 w_k1_INTC_s006
                                 + 0.993778 w_k1_KO_s006
                                 + 0.94918 w_k1_MAT_s006
                                 + 1.23476 w_k1_MSFT_s006
                                 + 0.96684 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.839954 w_k1_ADM_s006
                                 + 1.08256 w_k1_AXP_s006
                                 + 1.31822 w_k1_DIS_s006
                                 + 1.05517 w_k1_FDX_s006
                                 + 1.48837 w_k1_IBM_s006
                                 + 1.14293 w_k1_INTC_s006
                                 + 1.21732 w_k1_KO_s006 + 1.18651 w_k1_MAT_s006
                                 + 1.05504 w_k1_MSFT_s006
                                 + 0.989422 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.814147 w_k1_ADM_s006
                                 + 0.89684 w_k1_AXP_s006
                                 + 0.88163 w_k1_DIS_s006
                                 + 0.963813 w_k1_FDX_s006
                                 + 0.829022 w_k1_IBM_s006
                                 + 1.18238 w_k1_INTC_s006
                                 + 0.846354 w_k1_KO_s006
                                 + 0.683311 w_k1_MAT_s006
                                 + 1.00142 w_k1_MSFT_s006
                                 + 0.843016 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.653521 w_k1_ADM_s006
                                 + 0.771729 w_k1_AXP_s006
                                 + 0.653079 w_k1_DIS_s006
                                 + 0.738689 w_k1_FDX_s006
                                 + 1.10202 w_k1_IBM_s006
                                 + 0.984188 w_k1_INTC_s006
                                 + 0.809146 w_k1_KO_s006
                                 + 0.962743 w_k1_MAT_s006
                                 + 0.850765 w_k1_MSFT_s006
                                 + 0.845277 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.15795 w_k1_ADM_s006
                                 + 1.20131 w_k1_AXP_s006
                                 + 1.22312 w_k1_DIS_s006
                                 + 1.18849 w_k1_FDX_s006
                                 + 1.11456 w_k1_IBM_s006
                                 + 1.4093 w_k1_INTC_s006 + 1.20563 w_k1_KO_s006
                                 + 1.08136 w_k1_MAT_s006
                                 + 1.21819 w_k1_MSFT_s006
                                 + 1.05963 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.941467 w_k1_ADM_s006
                                 + 0.846803 w_k1_AXP_s006
                                 + 1.45624 w_k1_DIS_s006
                                 + 1.10054 w_k1_FDX_s006
                                 + 0.929216 w_k1_IBM_s006
                                 + 1.25016 w_k1_INTC_s006
                                 + 1.04184 w_k1_KO_s006
                                 + 0.741815 w_k1_MAT_s006
                                 + 1.28863 w_k1_MSFT_s006
                                 + 0.986985 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.865573 w_k1_ADM_s006
                                 + 0.624282 w_k1_AXP_s006
                                 + 0.62542 w_k1_DIS_s006
                                 + 0.736059 w_k1_FDX_s006
                                 + 0.808816 w_k1_IBM_s006
                                 + 0.861176 w_k1_INTC_s006
                                 + 0.768086 w_k1_KO_s006
                                 + 0.738661 w_k1_MAT_s006
                                 + 0.772388 w_k1_MSFT_s006
                                 + 0.817437 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.902434 w_k1_ADM_s007
                                 + 1.11786 w_k1_AXP_s007
                                 + 1.48288 w_k1_DIS_s007
                                 + 1.15485 w_k1_FDX_s007
                                 + 1.08086 w_k1_IBM_s007
                                 + 1.21018 w_k1_INTC_s007
                                 + 1.31559 w_k1_KO_s007 + 1.03896 w_k1_MAT_s007
                                 + 1.16174 w_k1_MSFT_s007
                                 + 1.10009 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.837722 w_k1_ADM_s007
                                 + 0.873746 w_k1_AXP_s007
                                 + 0.943672 w_k1_DIS_s007
                                 + 0.806427 w_k1_FDX_s007
                                 + 1.30363 w_k1_IBM_s007
                                 + 1.03019 w_k1_INTC_s007
                                 + 1.04385 w_k1_KO_s007
                                 + 0.930294 w_k1_MAT_s007
                                 + 0.961692 w_k1_MSFT_s007
                                 + 0.961381 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.75843 w_k1_ADM_s007
                                 + 0.663009 w_k1_AXP_s007
                                 + 0.746447 w_k1_DIS_s007
                                 + 0.811325 w_k1_FDX_s007
                                 + 0.734035 w_k1_IBM_s007
                                 + 0.961077 w_k1_INTC_s007
                                 + 0.836979 w_k1_KO_s007
                                 + 0.621471 w_k1_MAT_s007
                                 + 0.92881 w_k1_MSFT_s007
                                 + 0.838019 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.927496 w_k1_ADM_s007
                                 + 1.13405 w_k1_AXP_s007
                                 + 0.969454 w_k1_DIS_s007
                                 + 1.00681 w_k1_FDX_s007
                                 + 1.01239 w_k1_IBM_s007
                                 + 1.07205 w_k1_INTC_s007
                                 + 0.981275 w_k1_KO_s007
                                 + 1.05467 w_k1_MAT_s007
                                 + 0.944101 w_k1_MSFT_s007
                                 + 1.12632 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.945303 w_k1_ADM_s007
                                 + 0.881173 w_k1_AXP_s007
                                 + 0.759532 w_k1_DIS_s007
                                 + 0.827233 w_k1_FDX_s007
                                 + 1.05322 w_k1_IBM_s007
                                 + 0.965715 w_k1_INTC_s007
                                 + 0.935469 w_k1_KO_s007
                                 + 1.00844 w_k1_MAT_s007
                                 + 0.937941 w_k1_MSFT_s007
                                 + 0.949992 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.06679 w_k1_ADM_s007
                                 + 1.04646 w_k1_AXP_s007
                                 + 1.08892 w_k1_DIS_s007
                                 + 1.16257 w_k1_FDX_s007
                                 + 0.72781 w_k1_IBM_s007
                                 + 1.16329 w_k1_INTC_s007
                                 + 0.83077 w_k1_KO_s007
                                 + 0.735077 w_k1_MAT_s007
                                 + 1.01488 w_k1_MSFT_s007
                                 + 0.99366 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.56135 w_k1_ADM_s007
                                 + 1.28984 w_k1_AXP_s007
                                 + 1.37252 w_k1_DIS_s007
                                 + 1.31476 w_k1_FDX_s007
                                 + 0.899172 w_k1_IBM_s007
                                 + 1.29254 w_k1_INTC_s007
                                 + 1.21819 w_k1_KO_s007
                                 + 0.844718 w_k1_MAT_s007
                                 + 1.23761 w_k1_MSFT_s007
                                 + 1.32706 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.03743 w_k1_ADM_s007
                                 + 1.28194 w_k1_AXP_s007
                                 + 1.43678 w_k1_DIS_s007 + 1.0543 w_k1_FDX_s007
                                 + 1.61672 w_k1_IBM_s007
                                 + 1.34998 w_k1_INTC_s007
                                 + 1.46208 w_k1_KO_s007 + 1.73396 w_k1_MAT_s007
                                 + 1.18415 w_k1_MSFT_s007
                                 + 1.13196 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04274 w_k1_ADM_s010
                                 + 1.19081 w_k1_AXP_s010
                                 + 1.14479 w_k1_DIS_s010
                                 + 1.23153 w_k1_FDX_s010
                                 + 0.717754 w_k1_IBM_s010
                                 + 1.27991 w_k1_INTC_s010
                                 + 0.918025 w_k1_KO_s010
                                 + 0.539475 w_k1_MAT_s010
                                 + 1.19201 w_k1_MSFT_s010
                                 + 1.14015 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.904839 w_k1_ADM_s010
                                 + 1.24098 w_k1_AXP_s010
                                 + 0.949008 w_k1_DIS_s010
                                 + 0.948113 w_k1_FDX_s010
                                 + 1.46865 w_k1_IBM_s010
                                 + 1.38177 w_k1_INTC_s010
                                 + 1.14249 w_k1_KO_s010 + 1.20305 w_k1_MAT_s010
                                 + 1.04054 w_k1_MSFT_s010
                                 + 1.09144 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.924307 w_k1_ADM_s010
                                 + 1.06008 w_k1_AXP_s010
                                 + 0.863219 w_k1_DIS_s010
                                 + 0.998747 w_k1_FDX_s010
                                 + 1.01861 w_k1_IBM_s010
                                 + 1.24751 w_k1_INTC_s010
                                 + 1.02377 w_k1_KO_s010
                                 + 0.812062 w_k1_MAT_s010
                                 + 1.00781 w_k1_MSFT_s010
                                 + 0.985774 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.831928 w_k1_ADM_s010
                                 + 0.578057 w_k1_AXP_s010
                                 + 0.615789 w_k1_DIS_s010
                                 + 0.694456 w_k1_FDX_s010
                                 + 0.753089 w_k1_IBM_s010
                                 + 0.845991 w_k1_INTC_s010
                                 + 0.764446 w_k1_KO_s010
                                 + 0.540989 w_k1_MAT_s010
                                 + 0.884018 w_k1_MSFT_s010
                                 + 0.734899 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.3124 w_k1_ADM_s010
                                 + 1.35546 w_k1_AXP_s010
                                 + 1.44575 w_k1_DIS_s010
                                 + 1.30428 w_k1_FDX_s010
                                 + 1.04267 w_k1_IBM_s010
                                 + 1.34116 w_k1_INTC_s010
                                 + 1.04917 w_k1_KO_s010
                                 + 0.755382 w_k1_MAT_s010
                                 + 1.32856 w_k1_MSFT_s010
                                 + 1.22159 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.732109 w_k1_ADM_s010
                                 + 0.672816 w_k1_AXP_s010
                                 + 0.754497 w_k1_DIS_s010
                                 + 0.702792 w_k1_FDX_s010
                                 + 1.00505 w_k1_IBM_s010
                                 + 1.04125 w_k1_INTC_s010
                                 + 1.01313 w_k1_KO_s010
                                 + 0.807627 w_k1_MAT_s010
                                 + 0.985441 w_k1_MSFT_s010
                                 + 0.899873 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.00589 w_k1_ADM_s010
                                 + 0.83111 w_k1_AXP_s010
                                 + 0.704822 w_k1_DIS_s010
                                 + 0.878445 w_k1_FDX_s010
                                 + 0.731924 w_k1_IBM_s010
                                 + 0.983211 w_k1_INTC_s010
                                 + 0.86332 w_k1_KO_s010 + 0.68953 w_k1_MAT_s010
                                 + 0.891273 w_k1_MSFT_s010
                                 + 0.901791 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.08917 w_k1_ADM_s010
                                 + 0.73915 w_k1_AXP_s010
                                 + 0.981906 w_k1_DIS_s010
                                 + 0.997357 w_k1_FDX_s010
                                 + 0.706442 w_k1_IBM_s010
                                 + 1.03183 w_k1_INTC_s010
                                 + 0.883409 w_k1_KO_s010
                                 + 0.525554 w_k1_MAT_s010
                                 + 1.1232 w_k1_MSFT_s010
                                 + 0.978937 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.953952 w_k1_ADM_s011
                                 + 0.842197 w_k1_AXP_s011
                                 + 1.01784 w_k1_DIS_s011
                                 + 0.989789 w_k1_FDX_s011
                                 + 0.973272 w_k1_IBM_s011
                                 + 1.03166 w_k1_INTC_s011
                                 + 1.03762 w_k1_KO_s011
                                 + 0.930206 w_k1_MAT_s011
                                 + 0.924931 w_k1_MSFT_s011
                                 + 0.98273 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.28283 w_k1_ADM_s011
                                 + 1.46543 w_k1_AXP_s011
                                 + 1.53901 w_k1_DIS_s011
                                 + 1.20227 w_k1_FDX_s011
                                 + 1.35554 w_k1_IBM_s011
                                 + 1.17715 w_k1_INTC_s011
                                 + 1.42178 w_k1_KO_s011 + 1.52374 w_k1_MAT_s011
                                 + 0.994197 w_k1_MSFT_s011
                                 + 1.34968 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.765015 w_k1_ADM_s011
                                 + 0.68409 w_k1_AXP_s011
                                 + 0.584598 w_k1_DIS_s011
                                 + 0.758497 w_k1_FDX_s011
                                 + 0.681013 w_k1_IBM_s011
                                 + 0.781257 w_k1_INTC_s011
                                 + 0.591444 w_k1_KO_s011
                                 + 0.580565 w_k1_MAT_s011
                                 + 0.680661 w_k1_MSFT_s011
                                 + 0.806144 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.875713 w_k1_ADM_s011
                                 + 0.849594 w_k1_AXP_s011
                                 + 0.808424 w_k1_DIS_s011
                                 + 0.892022 w_k1_FDX_s011
                                 + 1.27888 w_k1_IBM_s011
                                 + 0.807668 w_k1_INTC_s011
                                 + 0.939147 w_k1_KO_s011
                                 + 1.32942 w_k1_MAT_s011
                                 + 0.824303 w_k1_MSFT_s011
                                 + 0.921661 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.16827 w_k1_ADM_s011
                                 + 1.19765 w_k1_AXP_s011
                                 + 0.994743 w_k1_DIS_s011
                                 + 1.06776 w_k1_FDX_s011 + 1.3622 w_k1_IBM_s011
                                 + 0.982384 w_k1_INTC_s011
                                 + 1.13931 w_k1_KO_s011 + 1.45648 w_k1_MAT_s011
                                 + 0.799623 w_k1_MSFT_s011
                                 + 1.17664 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.781818 w_k1_ADM_s011
                                 + 0.793503 w_k1_AXP_s011
                                 + 0.577231 w_k1_DIS_s011
                                 + 0.71878 w_k1_FDX_s011
                                 + 0.963503 w_k1_IBM_s011
                                 + 0.734847 w_k1_INTC_s011
                                 + 0.847889 w_k1_KO_s011
                                 + 0.911791 w_k1_MAT_s011
                                 + 0.699122 w_k1_MSFT_s011
                                 + 0.861138 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.869479 w_k1_ADM_s011
                                 + 0.768874 w_k1_AXP_s011
                                 + 0.763022 w_k1_DIS_s011
                                 + 0.940653 w_k1_FDX_s011
                                 + 0.774154 w_k1_IBM_s011
                                 + 0.799884 w_k1_INTC_s011
                                 + 0.828598 w_k1_KO_s011
                                 + 0.586216 w_k1_MAT_s011
                                 + 0.804197 w_k1_MSFT_s011
                                 + 0.983713 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.870894 w_k1_ADM_s011
                                 + 1.17183 w_k1_AXP_s011
                                 + 1.01916 w_k1_DIS_s011
                                 + 1.19736 w_k1_FDX_s011
                                 + 0.844649 w_k1_IBM_s011
                                 + 1.18366 w_k1_INTC_s011
                                 + 0.997764 w_k1_KO_s011
                                 + 0.807962 w_k1_MAT_s011
                                 + 1.08717 w_k1_MSFT_s011
                                 + 1.11488 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.21881 w_k1_ADM_s012
                                 + 0.746797 w_k1_AXP_s012
                                 + 0.906893 w_k1_DIS_s012
                                 + 0.85416 w_k1_FDX_s012
                                 + 1.12147 w_k1_IBM_s012
                                 + 0.76279 w_k1_INTC_s012
                                 + 1.06103 w_k1_KO_s012
                                 + 0.884933 w_k1_MAT_s012
                                 + 0.961167 w_k1_MSFT_s012
                                 + 1.05095 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.00017 w_k1_ADM_s012
                                 + 1.36484 w_k1_AXP_s012
                                 + 1.38713 w_k1_DIS_s012
                                 + 1.12983 w_k1_FDX_s012
                                 + 2.08241 w_k1_IBM_s012
                                 + 1.15916 w_k1_INTC_s012 + 1.3441 w_k1_KO_s012
                                 + 1.96291 w_k1_MAT_s012
                                 + 1.00296 w_k1_MSFT_s012
                                 + 1.20831 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.1301 w_k1_ADM_s012
                                 + 1.21275 w_k1_AXP_s012
                                 + 1.28157 w_k1_DIS_s012
                                 + 1.09046 w_k1_FDX_s012 + 1.4399 w_k1_IBM_s012
                                 + 0.9647 w_k1_INTC_s012 + 1.16466 w_k1_KO_s012
                                 + 1.07411 w_k1_MAT_s012
                                 + 1.09355 w_k1_MSFT_s012
                                 + 1.04307 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.56946 w_k1_ADM_s012
                                 + 1.48995 w_k1_AXP_s012
                                 + 1.45739 w_k1_DIS_s012
                                 + 1.25537 w_k1_FDX_s012
                                 + 1.23714 w_k1_IBM_s012
                                 + 1.16271 w_k1_INTC_s012
                                 + 1.24755 w_k1_KO_s012 + 1.0519 w_k1_MAT_s012
                                 + 1.03105 w_k1_MSFT_s012
                                 + 1.11768 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.18097 w_k1_ADM_s012
                                 + 1.10885 w_k1_AXP_s012
                                 + 1.05337 w_k1_DIS_s012
                                 + 1.12525 w_k1_FDX_s012
                                 + 0.761494 w_k1_IBM_s012
                                 + 1.05757 w_k1_INTC_s012
                                 + 0.876754 w_k1_KO_s012
                                 + 0.628856 w_k1_MAT_s012
                                 + 0.940322 w_k1_MSFT_s012
                                 + 0.935252 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.19649 w_k1_ADM_s012
                                 + 0.934394 w_k1_AXP_s012
                                 + 1.05572 w_k1_DIS_s012
                                 + 0.970895 w_k1_FDX_s012
                                 + 1.28096 w_k1_IBM_s012
                                 + 0.842339 w_k1_INTC_s012
                                 + 1.01465 w_k1_KO_s012 + 1.41583 w_k1_MAT_s012
                                 + 0.91177 w_k1_MSFT_s012
                                 + 1.03843 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.952865 w_k1_ADM_s012
                                 + 0.911212 w_k1_AXP_s012
                                 + 0.692328 w_k1_DIS_s012
                                 + 0.945019 w_k1_FDX_s012
                                 + 0.77403 w_k1_IBM_s012
                                 + 0.692274 w_k1_INTC_s012
                                 + 0.713161 w_k1_KO_s012
                                 + 0.696842 w_k1_MAT_s012
                                 + 0.747575 w_k1_MSFT_s012
                                 + 0.835099 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.794117 w_k1_ADM_s012
                                 + 0.583483 w_k1_AXP_s012
                                 + 0.667262 w_k1_DIS_s012
                                 + 0.751583 w_k1_FDX_s012
                                 + 0.749737 w_k1_IBM_s012
                                 + 0.956071 w_k1_INTC_s012
                                 + 0.716168 w_k1_KO_s012
                                 + 0.515262 w_k1_MAT_s012
                                 + 0.848497 w_k1_MSFT_s012
                                 + 0.842823 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.849854 w_k1_ADM_s013
                                 + 0.671015 w_k1_AXP_s013
                                 + 0.899335 w_k1_DIS_s013
                                 + 0.856727 w_k1_FDX_s013
                                 + 0.883136 w_k1_IBM_s013
                                 + 0.79145 w_k1_INTC_s013
                                 + 0.966446 w_k1_KO_s013
                                 + 0.80915 w_k1_MAT_s013
                                 + 0.831869 w_k1_MSFT_s013
                                 + 0.915791 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.00302 w_k1_ADM_s013
                                 + 1.00489 w_k1_AXP_s013
                                 + 1.32185 w_k1_DIS_s013
                                 + 1.16673 w_k1_FDX_s013
                                 + 0.836871 w_k1_IBM_s013
                                 + 1.07776 w_k1_INTC_s013
                                 + 0.984835 w_k1_KO_s013
                                 + 0.881134 w_k1_MAT_s013
                                 + 1.10372 w_k1_MSFT_s013
                                 + 0.947283 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.985023 w_k1_ADM_s013
                                 + 0.947453 w_k1_AXP_s013
                                 + 0.976876 w_k1_DIS_s013
                                 + 0.911162 w_k1_FDX_s013
                                 + 1.18626 w_k1_IBM_s013
                                 + 0.853673 w_k1_INTC_s013
                                 + 0.986606 w_k1_KO_s013
                                 + 1.18454 w_k1_MAT_s013
                                 + 0.899091 w_k1_MSFT_s013
                                 + 1.00098 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.757527 w_k1_ADM_s013
                                 + 0.679564 w_k1_AXP_s013
                                 + 0.604166 w_k1_DIS_s013
                                 + 0.702665 w_k1_FDX_s013
                                 + 0.849965 w_k1_IBM_s013
                                 + 0.729748 w_k1_INTC_s013
                                 + 0.728372 w_k1_KO_s013
                                 + 0.749658 w_k1_MAT_s013
                                 + 0.80001 w_k1_MSFT_s013
                                 + 0.670185 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.872904 w_k1_ADM_s013
                                 + 1.1863 w_k1_AXP_s013 + 1.03067 w_k1_DIS_s013
                                 + 0.854084 w_k1_FDX_s013
                                 + 1.65077 w_k1_IBM_s013
                                 + 0.914017 w_k1_INTC_s013
                                 + 1.28189 w_k1_KO_s013 + 1.83978 w_k1_MAT_s013
                                 + 0.929833 w_k1_MSFT_s013
                                 + 0.939003 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.822954 w_k1_ADM_s013
                                 + 0.951513 w_k1_AXP_s013
                                 + 0.722132 w_k1_DIS_s013
                                 + 0.926945 w_k1_FDX_s013
                                 + 0.732734 w_k1_IBM_s013
                                 + 0.869877 w_k1_INTC_s013
                                 + 0.725797 w_k1_KO_s013
                                 + 0.927399 w_k1_MAT_s013
                                 + 0.719721 w_k1_MSFT_s013
                                 + 0.869382 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.969295 w_k1_ADM_s013
                                 + 1.06513 w_k1_AXP_s013
                                 + 1.35278 w_k1_DIS_s013
                                 + 1.03048 w_k1_FDX_s013
                                 + 1.24763 w_k1_IBM_s013
                                 + 0.965924 w_k1_INTC_s013
                                 + 1.18179 w_k1_KO_s013 + 1.15758 w_k1_MAT_s013
                                 + 1.07215 w_k1_MSFT_s013
                                 + 1.06686 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.1032 w_k1_ADM_s013
                                 + 1.5764 w_k1_AXP_s013 + 1.59564 w_k1_DIS_s013
                                 + 1.40714 w_k1_FDX_s013
                                 + 1.46576 w_k1_IBM_s013
                                 + 1.43037 w_k1_INTC_s013
                                 + 1.27665 w_k1_KO_s013 + 1.36358 w_k1_MAT_s013
                                 + 1.20344 w_k1_MSFT_s013
                                 + 1.29491 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.759473 w_k1_ADM_s014
                                 + 0.850148 w_k1_AXP_s014
                                 + 0.772652 w_k1_DIS_s014
                                 + 0.760178 w_k1_FDX_s014
                                 + 1.00435 w_k1_IBM_s014
                                 + 0.864788 w_k1_INTC_s014
                                 + 0.991868 w_k1_KO_s014
                                 + 1.20191 w_k1_MAT_s014
                                 + 0.845313 w_k1_MSFT_s014
                                 + 0.961112 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.869417 w_k1_ADM_s014
                                 + 1.01206 w_k1_AXP_s014
                                 + 0.998897 w_k1_DIS_s014
                                 + 1.17249 w_k1_FDX_s014
                                 + 0.642612 w_k1_IBM_s014
                                 + 1.08481 w_k1_INTC_s014
                                 + 0.839918 w_k1_KO_s014
                                 + 1.02395 w_k1_MAT_s014
                                 + 1.10386 w_k1_MSFT_s014
                                 + 1.02135 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.941203 w_k1_ADM_s014
                                 + 0.799379 w_k1_AXP_s014
                                 + 0.713294 w_k1_DIS_s014
                                 + 0.859743 w_k1_FDX_s014
                                 + 0.562161 w_k1_IBM_s014
                                 + 1.14313 w_k1_INTC_s014
                                 + 0.747734 w_k1_KO_s014
                                 + 0.491791 w_k1_MAT_s014
                                 + 0.933304 w_k1_MSFT_s014
                                 + 0.828384 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.825188 w_k1_ADM_s014
                                 + 1.01043 w_k1_AXP_s014
                                 + 1.23825 w_k1_DIS_s014
                                 + 1.02692 w_k1_FDX_s014
                                 + 0.998846 w_k1_IBM_s014
                                 + 1.24362 w_k1_INTC_s014
                                 + 1.16404 w_k1_KO_s014 + 1.20566 w_k1_MAT_s014
                                 + 1.2848 w_k1_MSFT_s014 + 1.1507 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.05867 w_k1_ADM_s014
                                 + 1.48696 w_k1_AXP_s014
                                 + 1.19311 w_k1_DIS_s014
                                 + 1.28947 w_k1_FDX_s014
                                 + 0.883648 w_k1_IBM_s014
                                 + 1.5418 w_k1_INTC_s014 + 1.07549 w_k1_KO_s014
                                 + 1.10733 w_k1_MAT_s014
                                 + 1.28109 w_k1_MSFT_s014
                                 + 1.10906 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.945116 w_k1_ADM_s014
                                 + 0.989379 w_k1_AXP_s014
                                 + 0.942071 w_k1_DIS_s014
                                 + 0.870301 w_k1_FDX_s014
                                 + 0.785329 w_k1_IBM_s014
                                 + 0.985965 w_k1_INTC_s014
                                 + 0.942872 w_k1_KO_s014
                                 + 0.933234 w_k1_MAT_s014
                                 + 0.989884 w_k1_MSFT_s014
                                 + 0.951797 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.02266 w_k1_ADM_s014
                                 + 1.39534 w_k1_AXP_s014
                                 + 1.40438 w_k1_DIS_s014
                                 + 1.16984 w_k1_FDX_s014
                                 + 1.45432 w_k1_IBM_s014
                                 + 1.4474 w_k1_INTC_s014 + 1.62506 w_k1_KO_s014
                                 + 1.86951 w_k1_MAT_s014
                                 + 1.32211 w_k1_MSFT_s014
                                 + 1.27015 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.879363 w_k1_ADM_s014
                                 + 1.0822 w_k1_AXP_s014 + 0.75895 w_k1_DIS_s014
                                 + 0.895082 w_k1_FDX_s014
                                 + 0.758932 w_k1_IBM_s014
                                 + 1.14322 w_k1_INTC_s014
                                 + 0.928529 w_k1_KO_s014
                                 + 1.21088 w_k1_MAT_s014
                                 + 1.03671 w_k1_MSFT_s014
                                 + 0.867676 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.649218 w_k1_ADM_s015
                                 + 0.816233 w_k1_AXP_s015
                                 + 0.780471 w_k1_DIS_s015
                                 + 0.755048 w_k1_FDX_s015
                                 + 0.979127 w_k1_IBM_s015
                                 + 0.947264 w_k1_INTC_s015
                                 + 0.907093 w_k1_KO_s015
                                 + 0.943579 w_k1_MAT_s015
                                 + 0.869155 w_k1_MSFT_s015
                                 + 0.967532 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.84099 w_k1_ADM_s015
                                 + 1.19261 w_k1_AXP_s015
                                 + 0.976477 w_k1_DIS_s015
                                 + 0.892319 w_k1_FDX_s015
                                 + 1.3557 w_k1_IBM_s015
                                 + 1.26281 w_k1_INTC_s015
                                 + 1.11692 w_k1_KO_s015 + 1.03555 w_k1_MAT_s015
                                 + 1.12601 w_k1_MSFT_s015
                                 + 1.09655 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.37103 w_k1_ADM_s015
                                 + 1.58778 w_k1_AXP_s015
                                 + 1.27813 w_k1_DIS_s015
                                 + 1.18891 w_k1_FDX_s015 + 1.2046 w_k1_IBM_s015
                                 + 1.33628 w_k1_INTC_s015
                                 + 1.28948 w_k1_KO_s015 + 1.2325 w_k1_MAT_s015
                                 + 1.31574 w_k1_MSFT_s015
                                 + 1.15949 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.13081 w_k1_ADM_s015
                                 + 1.81471 w_k1_AXP_s015
                                 + 1.67519 w_k1_DIS_s015 + 1.3215 w_k1_FDX_s015
                                 + 1.73339 w_k1_IBM_s015
                                 + 1.54913 w_k1_INTC_s015
                                 + 1.39697 w_k1_KO_s015 + 1.31998 w_k1_MAT_s015
                                 + 1.27668 w_k1_MSFT_s015 + 1.3596 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.911308 w_k1_ADM_s015
                                 + 0.955186 w_k1_AXP_s015
                                 + 0.799094 w_k1_DIS_s015
                                 + 0.744965 w_k1_FDX_s015
                                 + 0.783535 w_k1_IBM_s015
                                 + 1.13388 w_k1_INTC_s015
                                 + 0.803472 w_k1_KO_s015
                                 + 0.652738 w_k1_MAT_s015
                                 + 0.990204 w_k1_MSFT_s015
                                 + 0.915125 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.996467 w_k1_ADM_s015
                                 + 1.11908 w_k1_AXP_s015
                                 + 1.09351 w_k1_DIS_s015
                                 + 1.01328 w_k1_FDX_s015
                                 + 0.728376 w_k1_IBM_s015
                                 + 1.03455 w_k1_INTC_s015
                                 + 0.883452 w_k1_KO_s015 + 0.7543 w_k1_MAT_s015
                                 + 1.12075 w_k1_MSFT_s015
                                 + 1.03171 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06878 w_k1_ADM_s015
                                 + 1.35992 w_k1_AXP_s015
                                 + 1.46849 w_k1_DIS_s015
                                 + 1.24635 w_k1_FDX_s015
                                 + 0.971791 w_k1_IBM_s015
                                 + 1.59101 w_k1_INTC_s015
                                 + 1.20652 w_k1_KO_s015
                                 + 0.752922 w_k1_MAT_s015
                                 + 1.38709 w_k1_MSFT_s015
                                 + 1.20191 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.13515 w_k1_ADM_s015
                                 + 1.23526 w_k1_AXP_s015
                                 + 1.02987 w_k1_DIS_s015
                                 + 0.993597 w_k1_FDX_s015
                                 + 1.0193 w_k1_IBM_s015
                                 + 1.45249 w_k1_INTC_s015
                                 + 1.11006 w_k1_KO_s015 + 1.01678 w_k1_MAT_s015
                                 + 1.26599 w_k1_MSFT_s015
                                 + 1.22773 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.918677 w_k1_ADM_s016
                                 + 0.786626 w_k1_AXP_s016
                                 + 0.617923 w_k1_DIS_s016
                                 + 0.731295 w_k1_FDX_s016
                                 + 0.81712 w_k1_IBM_s016
                                 + 0.70787 w_k1_INTC_s016
                                 + 0.766808 w_k1_KO_s016
                                 + 0.720541 w_k1_MAT_s016
                                 + 0.737173 w_k1_MSFT_s016
                                 + 0.771101 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.08016 w_k1_ADM_s016
                                 + 0.881206 w_k1_AXP_s016
                                 + 1.12329 w_k1_DIS_s016
                                 + 0.992719 w_k1_FDX_s016
                                 + 0.916947 w_k1_IBM_s016
                                 + 1.01275 w_k1_INTC_s016
                                 + 1.01041 w_k1_KO_s016
                                 + 0.839641 w_k1_MAT_s016
                                 + 1.04224 w_k1_MSFT_s016
                                 + 0.965199 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.49456 w_k1_ADM_s016
                                 + 1.14883 w_k1_AXP_s016
                                 + 1.07515 w_k1_DIS_s016
                                 + 1.20791 w_k1_FDX_s016
                                 + 0.807555 w_k1_IBM_s016
                                 + 1.09115 w_k1_INTC_s016
                                 + 0.84341 w_k1_KO_s016
                                 + 0.622574 w_k1_MAT_s016
                                 + 1.12419 w_k1_MSFT_s016
                                 + 0.874587 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.948558 w_k1_ADM_s016
                                 + 1.04919 w_k1_AXP_s016
                                 + 0.906465 w_k1_DIS_s016
                                 + 0.880106 w_k1_FDX_s016
                                 + 1.06057 w_k1_IBM_s016
                                 + 0.705397 w_k1_INTC_s016
                                 + 1.06002 w_k1_KO_s016 + 1.34814 w_k1_MAT_s016
                                 + 0.825128 w_k1_MSFT_s016
                                 + 0.923667 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.869279 w_k1_ADM_s016
                                 + 0.695891 w_k1_AXP_s016
                                 + 0.886644 w_k1_DIS_s016
                                 + 0.870577 w_k1_FDX_s016
                                 + 0.766664 w_k1_IBM_s016
                                 + 0.772196 w_k1_INTC_s016
                                 + 0.837857 w_k1_KO_s016
                                 + 0.67859 w_k1_MAT_s016
                                 + 0.830303 w_k1_MSFT_s016
                                 + 0.814207 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.19578 w_k1_ADM_s016
                                 + 1.4959 w_k1_AXP_s016 + 1.54334 w_k1_DIS_s016
                                 + 1.29698 w_k1_FDX_s016
                                 + 1.30694 w_k1_IBM_s016
                                 + 1.00884 w_k1_INTC_s016
                                 + 1.41922 w_k1_KO_s016 + 1.34957 w_k1_MAT_s016
                                 + 1.23353 w_k1_MSFT_s016
                                 + 1.13313 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.01433 w_k1_ADM_s016
                                 + 0.976255 w_k1_AXP_s016
                                 + 1.34211 w_k1_DIS_s016
                                 + 1.10228 w_k1_FDX_s016
                                 + 0.88453 w_k1_IBM_s016
                                 + 1.07701 w_k1_INTC_s016
                                 + 0.989836 w_k1_KO_s016
                                 + 1.08249 w_k1_MAT_s016
                                 + 1.0607 w_k1_MSFT_s016
                                 + 0.934314 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.786121 w_k1_ADM_s016
                                 + 1.16229 w_k1_AXP_s016
                                 + 1.04261 w_k1_DIS_s016
                                 + 0.93093 w_k1_FDX_s016
                                 + 1.46885 w_k1_IBM_s016
                                 + 0.952403 w_k1_INTC_s016 + 1.146 w_k1_KO_s016
                                 + 1.63497 w_k1_MAT_s016
                                 + 0.975594 w_k1_MSFT_s016
                                 + 1.00642 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.05969 w_k1_ADM_s017
                                 + 0.756898 w_k1_AXP_s017
                                 + 0.846239 w_k1_DIS_s017
                                 + 0.816267 w_k1_FDX_s017
                                 + 1.15313 w_k1_IBM_s017
                                 + 0.797667 w_k1_INTC_s017
                                 + 1.2428 w_k1_KO_s017 + 1.20147 w_k1_MAT_s017
                                 + 0.920253 w_k1_MSFT_s017
                                 + 1.0107 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.924222 w_k1_ADM_s017
                                 + 0.62627 w_k1_AXP_s017
                                 + 0.615758 w_k1_DIS_s017
                                 + 0.774954 w_k1_FDX_s017
                                 + 0.792381 w_k1_IBM_s017
                                 + 0.836366 w_k1_INTC_s017
                                 + 0.712345 w_k1_KO_s017
                                 + 0.708105 w_k1_MAT_s017
                                 + 0.76232 w_k1_MSFT_s017
                                 + 0.813473 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.28514 w_k1_ADM_s017
                                 + 1.30085 w_k1_AXP_s017
                                 + 1.26668 w_k1_DIS_s017
                                 + 1.34665 w_k1_FDX_s017
                                 + 0.948562 w_k1_IBM_s017
                                 + 1.20745 w_k1_INTC_s017
                                 + 1.03526 w_k1_KO_s017 + 1.07633 w_k1_MAT_s017
                                 + 0.98841 w_k1_MSFT_s017 + 1.0056 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.06575 w_k1_ADM_s017
                                 + 0.666215 w_k1_AXP_s017
                                 + 0.877859 w_k1_DIS_s017
                                 + 0.903251 w_k1_FDX_s017
                                 + 0.828848 w_k1_IBM_s017
                                 + 0.897655 w_k1_INTC_s017
                                 + 0.993734 w_k1_KO_s017
                                 + 0.82374 w_k1_MAT_s017
                                 + 0.874568 w_k1_MSFT_s017
                                 + 0.944697 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.19541 w_k1_ADM_s017
                                 + 1.16192 w_k1_AXP_s017
                                 + 1.00197 w_k1_DIS_s017
                                 + 0.990898 w_k1_FDX_s017
                                 + 2.0355 w_k1_IBM_s017
                                 + 1.20201 w_k1_INTC_s017
                                 + 1.46657 w_k1_KO_s017 + 1.55208 w_k1_MAT_s017
                                 + 0.99579 w_k1_MSFT_s017 + 0.9821 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.14579 w_k1_ADM_s017
                                 + 1.37063 w_k1_AXP_s017
                                 + 1.56581 w_k1_DIS_s017
                                 + 1.29465 w_k1_FDX_s017
                                 + 1.55523 w_k1_IBM_s017
                                 + 1.48633 w_k1_INTC_s017
                                 + 1.54405 w_k1_KO_s017 + 1.42411 w_k1_MAT_s017
                                 + 1.35636 w_k1_MSFT_s017
                                 + 1.11924 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.37776 w_k1_ADM_s017
                                 + 1.21396 w_k1_AXP_s017 + 1.1448 w_k1_DIS_s017
                                 + 1.1449 w_k1_FDX_s017 + 1.41409 w_k1_IBM_s017
                                 + 1.25998 w_k1_INTC_s017
                                 + 1.39352 w_k1_KO_s017 + 1.17731 w_k1_MAT_s017
                                 + 1.06638 w_k1_MSFT_s017
                                 + 1.09103 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.916373 w_k1_ADM_s017
                                 + 1.00009 w_k1_AXP_s017
                                 + 0.873877 w_k1_DIS_s017
                                 + 1.0773 w_k1_FDX_s017
                                 + 0.710559 w_k1_IBM_s017
                                 + 1.24873 w_k1_INTC_s017
                                 + 0.895483 w_k1_KO_s017
                                 + 0.863182 w_k1_MAT_s017
                                 + 0.918308 w_k1_MSFT_s017
                                 + 0.939511 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.976319 w_k1_ADM_s100
                                 + 1.11562 w_k1_AXP_s100
                                 + 1.80766 w_k1_DIS_s100
                                 + 1.45345 w_k1_FDX_s100
                                 + 0.979855 w_k1_IBM_s100
                                 + 2.00787 w_k1_INTC_s100
                                 + 1.55722 w_k1_KO_s100
                                 + 0.750751 w_k1_MAT_s100
                                 + 1.77273 w_k1_MSFT_s100
                                 + 1.42005 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.953355 w_k1_ADM_s100
                                 + 1.29238 w_k1_AXP_s100
                                 + 1.24828 w_k1_DIS_s100
                                 + 1.08969 w_k1_FDX_s100
                                 + 1.35305 w_k1_IBM_s100
                                 + 1.43382 w_k1_INTC_s100
                                 + 1.21961 w_k1_KO_s100 + 1.07307 w_k1_MAT_s100
                                 + 0.993714 w_k1_MSFT_s100
                                 + 1.0907 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.04224 w_k1_ADM_s100
                                 + 1.10946 w_k1_AXP_s100
                                 + 1.07101 w_k1_DIS_s100
                                 + 1.10906 w_k1_FDX_s100
                                 + 0.833448 w_k1_IBM_s100
                                 + 1.14183 w_k1_INTC_s100
                                 + 1.11528 w_k1_KO_s100
                                 + 0.921034 w_k1_MAT_s100
                                 + 1.19826 w_k1_MSFT_s100
                                 + 1.16404 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.990932 w_k1_ADM_s100
                                 + 0.677303 w_k1_AXP_s100
                                 + 0.758559 w_k1_DIS_s100
                                 + 0.909427 w_k1_FDX_s100
                                 + 0.627756 w_k1_IBM_s100
                                 + 0.899027 w_k1_INTC_s100
                                 + 0.776719 w_k1_KO_s100
                                 + 0.681321 w_k1_MAT_s100
                                 + 0.844777 w_k1_MSFT_s100
                                 + 0.926982 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.768231 w_k1_ADM_s100
                                 + 0.758568 w_k1_AXP_s100
                                 + 0.677397 w_k1_DIS_s100
                                 + 0.702725 w_k1_FDX_s100
                                 + 1.05845 w_k1_IBM_s100
                                 + 1.10857 w_k1_INTC_s100
                                 + 0.943857 w_k1_KO_s100
                                 + 0.846216 w_k1_MAT_s100
                                 + 0.797335 w_k1_MSFT_s100
                                 + 0.74526 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.705137 w_k1_ADM_s100
                                 + 0.681176 w_k1_AXP_s100
                                 + 0.805649 w_k1_DIS_s100
                                 + 0.909967 w_k1_FDX_s100
                                 + 0.809678 w_k1_IBM_s100
                                 + 1.11756 w_k1_INTC_s100
                                 + 0.922226 w_k1_KO_s100
                                 + 0.884199 w_k1_MAT_s100
                                 + 0.951649 w_k1_MSFT_s100
                                 + 0.869358 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.10349 w_k1_ADM_s100
                                 + 0.743308 w_k1_AXP_s100
                                 + 1.36308 w_k1_DIS_s100
                                 + 1.22806 w_k1_FDX_s100
                                 + 0.726263 w_k1_IBM_s100
                                 + 1.18191 w_k1_INTC_s100
                                 + 1.23446 w_k1_KO_s100
                                 + 0.555091 w_k1_MAT_s100
                                 + 1.2583 w_k1_MSFT_s100 + 1.15515 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.824365 w_k1_ADM_s100
                                 + 0.81868 w_k1_AXP_s100
                                 + 0.768804 w_k1_DIS_s100
                                 + 0.921226 w_k1_FDX_s100
                                 + 1.11312 w_k1_IBM_s100
                                 + 1.14066 w_k1_INTC_s100
                                 + 0.89851 w_k1_KO_s100 + 1.01512 w_k1_MAT_s100
                                 + 0.898984 w_k1_MSFT_s100
                                 + 0.952873 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.986712 w_k1_ADM_s101
                                 + 0.992272 w_k1_AXP_s101
                                 + 0.701421 w_k1_DIS_s101
                                 + 0.828813 w_k1_FDX_s101
                                 + 1.44786 w_k1_IBM_s101
                                 + 0.976616 w_k1_INTC_s101
                                 + 1.12525 w_k1_KO_s101 + 1.07367 w_k1_MAT_s101
                                 + 0.891947 w_k1_MSFT_s101
                                 + 1.05372 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.86838 w_k1_ADM_s101
                                 + 0.644856 w_k1_AXP_s101
                                 + 0.761256 w_k1_DIS_s101
                                 + 0.782595 w_k1_FDX_s101
                                 + 0.800691 w_k1_IBM_s101
                                 + 0.8297 w_k1_INTC_s101
                                 + 0.850302 w_k1_KO_s101
                                 + 0.675908 w_k1_MAT_s101
                                 + 0.871703 w_k1_MSFT_s101
                                 + 0.806943 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.27936 w_k1_ADM_s101
                                 + 1.15725 w_k1_AXP_s101
                                 + 1.32677 w_k1_DIS_s101
                                 + 1.05646 w_k1_FDX_s101
                                 + 1.45679 w_k1_IBM_s101
                                 + 1.10892 w_k1_INTC_s101 + 1.344 w_k1_KO_s101
                                 + 1.45996 w_k1_MAT_s101
                                 + 1.09486 w_k1_MSFT_s101
                                 + 1.16118 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.45651 w_k1_ADM_s101
                                 + 1.33413 w_k1_AXP_s101
                                 + 1.70722 w_k1_DIS_s101
                                 + 1.43442 w_k1_FDX_s101
                                 + 0.969977 w_k1_IBM_s101
                                 + 1.38647 w_k1_INTC_s101
                                 + 1.49927 w_k1_KO_s101
                                 + 0.962033 w_k1_MAT_s101
                                 + 1.32282 w_k1_MSFT_s101
                                 + 1.21528 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.06378 w_k1_ADM_s101
                                 + 0.929567 w_k1_AXP_s101
                                 + 0.910381 w_k1_DIS_s101
                                 + 0.863105 w_k1_FDX_s101
                                 + 1.07502 w_k1_IBM_s101
                                 + 0.838354 w_k1_INTC_s101
                                 + 1.10908 w_k1_KO_s101 + 1.21025 w_k1_MAT_s101
                                 + 0.935611 w_k1_MSFT_s101
                                 + 1.00789 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.19888 w_k1_ADM_s101
                                 + 0.894183 w_k1_AXP_s101
                                 + 0.897256 w_k1_DIS_s101
                                 + 1.06372 w_k1_FDX_s101
                                 + 0.651704 w_k1_IBM_s101
                                 + 0.911946 w_k1_INTC_s101
                                 + 0.832532 w_k1_KO_s101
                                 + 0.622592 w_k1_MAT_s101
                                 + 1.00636 w_k1_MSFT_s101
                                 + 0.935823 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.25605 w_k1_ADM_s101
                                 + 1.23465 w_k1_AXP_s101
                                 + 1.05952 w_k1_DIS_s101
                                 + 1.16199 w_k1_FDX_s101
                                 + 0.929235 w_k1_IBM_s101
                                 + 1.00742 w_k1_INTC_s101 + 1.1899 w_k1_KO_s101
                                 + 1.06519 w_k1_MAT_s101
                                 + 1.21422 w_k1_MSFT_s101
                                 + 1.04729 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.2106 w_k1_ADM_s101
                                 + 0.981602 w_k1_AXP_s101
                                 + 1.02989 w_k1_DIS_s101
                                 + 1.12799 w_k1_FDX_s101
                                 + 0.832987 w_k1_IBM_s101
                                 + 1.0031 w_k1_INTC_s101 + 1.09543 w_k1_KO_s101
                                 + 0.830398 w_k1_MAT_s101
                                 + 1.11913 w_k1_MSFT_s101
                                 + 1.06141 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.39709 w_k1_ADM_s102
                                 + 1.50538 w_k1_AXP_s102
                                 + 1.64814 w_k1_DIS_s102
                                 + 1.43765 w_k1_FDX_s102
                                 + 1.25483 w_k1_IBM_s102
                                 + 1.23073 w_k1_INTC_s102
                                 + 1.22202 w_k1_KO_s102 + 1.66177 w_k1_MAT_s102
                                 + 1.29933 w_k1_MSFT_s102
                                 + 1.21477 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.880951 w_k1_ADM_s102
                                 + 1.01797 w_k1_AXP_s102
                                 + 0.789947 w_k1_DIS_s102
                                 + 0.802881 w_k1_FDX_s102
                                 + 1.15266 w_k1_IBM_s102
                                 + 0.720594 w_k1_INTC_s102
                                 + 0.986849 w_k1_KO_s102
                                 + 1.51336 w_k1_MAT_s102
                                 + 0.740504 w_k1_MSFT_s102
                                 + 0.86223 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.893171 w_k1_ADM_s102
                                 + 1.01007 w_k1_AXP_s102
                                 + 0.893743 w_k1_DIS_s102
                                 + 1.01137 w_k1_FDX_s102
                                 + 0.773123 w_k1_IBM_s102
                                 + 0.937503 w_k1_INTC_s102
                                 + 0.801077 w_k1_KO_s102
                                 + 0.816471 w_k1_MAT_s102
                                 + 1.00207 w_k1_MSFT_s102
                                 + 0.799667 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.13747 w_k1_ADM_s102
                                 + 1.12292 w_k1_AXP_s102
                                 + 1.24765 w_k1_DIS_s102
                                 + 1.07058 w_k1_FDX_s102
                                 + 1.19777 w_k1_IBM_s102
                                 + 1.08029 w_k1_INTC_s102
                                 + 1.31317 w_k1_KO_s102 + 1.38776 w_k1_MAT_s102
                                 + 1.24168 w_k1_MSFT_s102
                                 + 0.968643 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.19663 w_k1_ADM_s102
                                 + 1.35566 w_k1_AXP_s102
                                 + 1.38571 w_k1_DIS_s102 + 1.1092 w_k1_FDX_s102
                                 + 1.8687 w_k1_IBM_s102
                                 + 1.18394 w_k1_INTC_s102
                                 + 1.43401 w_k1_KO_s102 + 1.7236 w_k1_MAT_s102
                                 + 1.26135 w_k1_MSFT_s102
                                 + 1.08213 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.800246 w_k1_ADM_s102
                                 + 0.779069 w_k1_AXP_s102
                                 + 0.893472 w_k1_DIS_s102
                                 + 0.955922 w_k1_FDX_s102
                                 + 0.978539 w_k1_IBM_s102
                                 + 0.813518 w_k1_INTC_s102
                                 + 1.05196 w_k1_KO_s102 + 1.10263 w_k1_MAT_s102
                                 + 0.885843 w_k1_MSFT_s102
                                 + 0.949376 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.16543 w_k1_ADM_s102
                                 + 1.3539 w_k1_AXP_s102 + 1.21189 w_k1_DIS_s102
                                 + 1.22785 w_k1_FDX_s102
                                 + 0.803354 w_k1_IBM_s102
                                 + 1.02097 w_k1_INTC_s102
                                 + 0.959703 w_k1_KO_s102
                                 + 1.26842 w_k1_MAT_s102
                                 + 1.30487 w_k1_MSFT_s102
                                 + 0.966713 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.7608 w_k1_ADM_s102
                                 + 0.709326 w_k1_AXP_s102
                                 + 0.729386 w_k1_DIS_s102
                                 + 0.781415 w_k1_FDX_s102
                                 + 0.687529 w_k1_IBM_s102
                                 + 0.696792 w_k1_INTC_s102
                                 + 0.805588 w_k1_KO_s102
                                 + 1.06835 w_k1_MAT_s102
                                 + 0.849415 w_k1_MSFT_s102
                                 + 0.829923 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.862814 w_k1_ADM_s103
                                 + 0.661888 w_k1_AXP_s103
                                 + 0.666459 w_k1_DIS_s103
                                 + 0.704475 w_k1_FDX_s103
                                 + 1.06022 w_k1_IBM_s103
                                 + 0.785952 w_k1_INTC_s103
                                 + 0.935529 w_k1_KO_s103
                                 + 0.669772 w_k1_MAT_s103
                                 + 0.782854 w_k1_MSFT_s103
                                 + 0.886564 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.13921 w_k1_ADM_s103
                                 + 1.32853 w_k1_AXP_s103
                                 + 1.47384 w_k1_DIS_s103
                                 + 1.37924 w_k1_FDX_s103 + 1.1625 w_k1_IBM_s103
                                 + 1.31709 w_k1_INTC_s103
                                 + 1.09058 w_k1_KO_s103
                                 + 0.904556 w_k1_MAT_s103
                                 + 1.2375 w_k1_MSFT_s103 + 1.0314 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.818848 w_k1_ADM_s103
                                 + 0.708382 w_k1_AXP_s103
                                 + 0.630929 w_k1_DIS_s103
                                 + 0.730681 w_k1_FDX_s103
                                 + 0.81714 w_k1_IBM_s103
                                 + 0.580448 w_k1_INTC_s103
                                 + 0.604485 w_k1_KO_s103
                                 + 0.653911 w_k1_MAT_s103
                                 + 0.741098 w_k1_MSFT_s103
                                 + 0.74254 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.829503 w_k1_ADM_s103
                                 + 0.722877 w_k1_AXP_s103
                                 + 0.766058 w_k1_DIS_s103
                                 + 0.815034 w_k1_FDX_s103
                                 + 0.727721 w_k1_IBM_s103
                                 + 0.901601 w_k1_INTC_s103
                                 + 0.746691 w_k1_KO_s103
                                 + 0.391701 w_k1_MAT_s103
                                 + 0.970336 w_k1_MSFT_s103
                                 + 0.855538 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.1177 w_k1_ADM_s103
                                 + 1.01216 w_k1_AXP_s103
                                 + 1.26897 w_k1_DIS_s103
                                 + 1.05147 w_k1_FDX_s103
                                 + 1.19522 w_k1_IBM_s103
                                 + 1.07523 w_k1_INTC_s103
                                 + 1.10169 w_k1_KO_s103
                                 + 0.807233 w_k1_MAT_s103
                                 + 1.11275 w_k1_MSFT_s103
                                 + 0.970051 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.03394 w_k1_ADM_s103
                                 + 0.906799 w_k1_AXP_s103
                                 + 0.886959 w_k1_DIS_s103
                                 + 0.878822 w_k1_FDX_s103
                                 + 0.988683 w_k1_IBM_s103
                                 + 0.828924 w_k1_INTC_s103
                                 + 0.94546 w_k1_KO_s103
                                 + 0.802719 w_k1_MAT_s103
                                 + 0.899636 w_k1_MSFT_s103
                                 + 0.94177 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.2123 w_k1_ADM_s103
                                 + 1.08253 w_k1_AXP_s103
                                 + 0.855193 w_k1_DIS_s103
                                 + 0.862178 w_k1_FDX_s103
                                 + 1.59593 w_k1_IBM_s103
                                 + 0.921275 w_k1_INTC_s103
                                 + 1.11764 w_k1_KO_s103 + 1.15604 w_k1_MAT_s103
                                 + 0.916479 w_k1_MSFT_s103
                                 + 0.939685 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.973575 w_k1_ADM_s103
                                 + 0.929624 w_k1_AXP_s103
                                 + 0.996358 w_k1_DIS_s103
                                 + 1.07135 w_k1_FDX_s103
                                 + 0.788636 w_k1_IBM_s103
                                 + 0.967553 w_k1_INTC_s103
                                 + 0.79798 w_k1_KO_s103
                                 + 0.593655 w_k1_MAT_s103
                                 + 1.03585 w_k1_MSFT_s103
                                 + 0.938259 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.933416 w_k1_ADM_s104
                                 + 0.841214 w_k1_AXP_s104
                                 + 0.835944 w_k1_DIS_s104
                                 + 0.797512 w_k1_FDX_s104
                                 + 1.25218 w_k1_IBM_s104
                                 + 0.804068 w_k1_INTC_s104
                                 + 1.05993 w_k1_KO_s104 + 1.08726 w_k1_MAT_s104
                                 + 0.895466 w_k1_MSFT_s104
                                 + 1.01587 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.29041 w_k1_ADM_s104
                                 + 0.946796 w_k1_AXP_s104
                                 + 1.06359 w_k1_DIS_s104
                                 + 1.15445 w_k1_FDX_s104
                                 + 0.563657 w_k1_IBM_s104
                                 + 0.97205 w_k1_INTC_s104
                                 + 0.785627 w_k1_KO_s104
                                 + 0.391271 w_k1_MAT_s104
                                 + 1.06991 w_k1_MSFT_s104
                                 + 0.994319 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.16894 w_k1_ADM_s104
                                 + 1.44644 w_k1_AXP_s104
                                 + 1.53872 w_k1_DIS_s104
                                 + 1.31683 w_k1_FDX_s104
                                 + 1.49653 w_k1_IBM_s104
                                 + 1.02202 w_k1_INTC_s104
                                 + 1.39289 w_k1_KO_s104 + 1.4712 w_k1_MAT_s104
                                 + 1.25157 w_k1_MSFT_s104
                                 + 1.36971 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.00283 w_k1_ADM_s104
                                 + 1.1053 w_k1_AXP_s104
                                 + 0.863581 w_k1_DIS_s104
                                 + 1.03699 w_k1_FDX_s104
                                 + 0.861423 w_k1_IBM_s104
                                 + 0.955762 w_k1_INTC_s104
                                 + 0.901486 w_k1_KO_s104
                                 + 0.685543 w_k1_MAT_s104
                                 + 1.00288 w_k1_MSFT_s104
                                 + 1.01522 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.676446 w_k1_ADM_s104
                                 + 0.590531 w_k1_AXP_s104
                                 + 0.46456 w_k1_DIS_s104
                                 + 0.685559 w_k1_FDX_s104
                                 + 0.675775 w_k1_IBM_s104
                                 + 0.651065 w_k1_INTC_s104
                                 + 0.661943 w_k1_KO_s104
                                 + 0.693358 w_k1_MAT_s104
                                 + 0.697504 w_k1_MSFT_s104
                                 + 0.833694 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.917592 w_k1_ADM_s104
                                 + 0.601764 w_k1_AXP_s104
                                 + 0.783352 w_k1_DIS_s104
                                 + 0.822619 w_k1_FDX_s104
                                 + 0.830413 w_k1_IBM_s104
                                 + 0.740198 w_k1_INTC_s104
                                 + 0.839677 w_k1_KO_s104
                                 + 0.70581 w_k1_MAT_s104
                                 + 0.865915 w_k1_MSFT_s104
                                 + 0.906331 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.21303 w_k1_ADM_s104
                                 + 1.11521 w_k1_AXP_s104
                                 + 1.33091 w_k1_DIS_s104
                                 + 1.08777 w_k1_FDX_s104
                                 + 1.12176 w_k1_IBM_s104
                                 + 0.957855 w_k1_INTC_s104
                                 + 1.30376 w_k1_KO_s104
                                 + 0.936554 w_k1_MAT_s104
                                 + 1.11821 w_k1_MSFT_s104
                                 + 1.17042 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.701948 w_k1_ADM_s104
                                 + 0.677561 w_k1_AXP_s104
                                 + 0.6522 w_k1_DIS_s104
                                 + 0.679695 w_k1_FDX_s104
                                 + 0.945905 w_k1_IBM_s104
                                 + 0.799338 w_k1_INTC_s104
                                 + 0.981944 w_k1_KO_s104 + 1.021 w_k1_MAT_s104
                                 + 0.724352 w_k1_MSFT_s104
                                 + 0.915113 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.37986 w_k1_ADM_s105
                                 + 1.63622 w_k1_AXP_s105
                                 + 1.83066 w_k1_DIS_s105
                                 + 1.31317 w_k1_FDX_s105
                                 + 1.50293 w_k1_IBM_s105
                                 + 1.0258 w_k1_INTC_s105 + 1.63205 w_k1_KO_s105
                                 + 1.5091 w_k1_MAT_s105
                                 + 1.32386 w_k1_MSFT_s105
                                 + 1.17687 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.48256 w_k1_ADM_s105
                                 + 1.19905 w_k1_AXP_s105
                                 + 1.24541 w_k1_DIS_s105 + 1.3423 w_k1_FDX_s105
                                 + 0.745717 w_k1_IBM_s105
                                 + 0.916542 w_k1_INTC_s105
                                 + 0.885314 w_k1_KO_s105
                                 + 0.610213 w_k1_MAT_s105
                                 + 0.986255 w_k1_MSFT_s105
                                 + 0.953783 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.873084 w_k1_ADM_s105
                                 + 1.0103 w_k1_AXP_s105
                                 + 0.992994 w_k1_DIS_s105
                                 + 0.802102 w_k1_FDX_s105
                                 + 1.60943 w_k1_IBM_s105
                                 + 0.609151 w_k1_INTC_s105
                                 + 1.10298 w_k1_KO_s105 + 1.66528 w_k1_MAT_s105
                                 + 0.705312 w_k1_MSFT_s105
                                 + 0.882475 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.04888 w_k1_ADM_s105
                                 + 0.964255 w_k1_AXP_s105
                                 + 0.908836 w_k1_DIS_s105
                                 + 0.876663 w_k1_FDX_s105
                                 + 1.08948 w_k1_IBM_s105
                                 + 0.717468 w_k1_INTC_s105
                                 + 0.999793 w_k1_KO_s105
                                 + 1.37934 w_k1_MAT_s105
                                 + 0.895313 w_k1_MSFT_s105
                                 + 0.908618 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.11191 w_k1_ADM_s105
                                 + 1.23835 w_k1_AXP_s105
                                 + 1.33348 w_k1_DIS_s105
                                 + 1.13714 w_k1_FDX_s105
                                 + 1.13184 w_k1_IBM_s105
                                 + 0.930013 w_k1_INTC_s105
                                 + 1.18916 w_k1_KO_s105 + 1.12555 w_k1_MAT_s105
                                 + 1.07074 w_k1_MSFT_s105
                                 + 0.900172 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.808657 w_k1_ADM_s105
                                 + 0.698863 w_k1_AXP_s105
                                 + 0.899745 w_k1_DIS_s105
                                 + 0.904804 w_k1_FDX_s105
                                 + 0.746767 w_k1_IBM_s105
                                 + 0.806568 w_k1_INTC_s105
                                 + 0.839779 w_k1_KO_s105
                                 + 0.819887 w_k1_MAT_s105
                                 + 0.918118 w_k1_MSFT_s105
                                 + 0.806605 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.07946 w_k1_ADM_s105
                                 + 0.981972 w_k1_AXP_s105
                                 + 1.18888 w_k1_DIS_s105
                                 + 1.10586 w_k1_FDX_s105
                                 + 0.882829 w_k1_IBM_s105
                                 + 1.00312 w_k1_INTC_s105
                                 + 1.03489 w_k1_KO_s105 + 0.90541 w_k1_MAT_s105
                                 + 1.08452 w_k1_MSFT_s105
                                 + 0.960726 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.29458 w_k1_ADM_s105
                                 + 1.52806 w_k1_AXP_s105 + 1.7375 w_k1_DIS_s105
                                 + 1.51354 w_k1_FDX_s105
                                 + 1.01947 w_k1_IBM_s105
                                 + 1.1983 w_k1_INTC_s105 + 1.53107 w_k1_KO_s105
                                 + 1.45549 w_k1_MAT_s105
                                 + 1.33381 w_k1_MSFT_s105
                                 + 1.14226 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.4167 w_k1_ADM_s106
                                 + 1.2327 w_k1_AXP_s106 + 1.59443 w_k1_DIS_s106
                                 + 1.31789 w_k1_FDX_s106
                                 + 0.922916 w_k1_IBM_s106
                                 + 1.19905 w_k1_INTC_s106
                                 + 1.32012 w_k1_KO_s106 + 1.03292 w_k1_MAT_s106
                                 + 1.20619 w_k1_MSFT_s106
                                 + 1.24155 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.930088 w_k1_ADM_s106
                                 + 0.727826 w_k1_AXP_s106
                                 + 0.755485 w_k1_DIS_s106
                                 + 0.787179 w_k1_FDX_s106
                                 + 0.87419 w_k1_IBM_s106
                                 + 0.803316 w_k1_INTC_s106
                                 + 0.808639 w_k1_KO_s106
                                 + 1.00938 w_k1_MAT_s106
                                 + 0.81036 w_k1_MSFT_s106
                                 + 0.780846 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.800484 w_k1_ADM_s106
                                 + 0.709511 w_k1_AXP_s106
                                 + 0.748351 w_k1_DIS_s106
                                 + 0.732452 w_k1_FDX_s106
                                 + 1.50554 w_k1_IBM_s106
                                 + 0.753717 w_k1_INTC_s106
                                 + 1.18662 w_k1_KO_s106 + 1.77094 w_k1_MAT_s106
                                 + 0.825861 w_k1_MSFT_s106
                                 + 1.02265 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.04612 w_k1_ADM_s106
                                 + 0.901705 w_k1_AXP_s106
                                 + 0.932795 w_k1_DIS_s106
                                 + 0.955002 w_k1_FDX_s106
                                 + 0.764623 w_k1_IBM_s106
                                 + 0.949277 w_k1_INTC_s106
                                 + 0.965935 w_k1_KO_s106
                                 + 0.831247 w_k1_MAT_s106
                                 + 0.98283 w_k1_MSFT_s106
                                 + 0.973297 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.013 w_k1_ADM_s106
                                 + 1.06507 w_k1_AXP_s106
                                 + 0.921176 w_k1_DIS_s106
                                 + 0.871499 w_k1_FDX_s106
                                 + 1.13274 w_k1_IBM_s106
                                 + 1.04491 w_k1_INTC_s106
                                 + 0.947447 w_k1_KO_s106
                                 + 1.05928 w_k1_MAT_s106
                                 + 0.880875 w_k1_MSFT_s106
                                 + 0.972236 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.24319 w_k1_ADM_s106
                                 + 1.52141 w_k1_AXP_s106
                                 + 1.15309 w_k1_DIS_s106
                                 + 1.21907 w_k1_FDX_s106
                                 + 1.24512 w_k1_IBM_s106
                                 + 1.26493 w_k1_INTC_s106
                                 + 1.36512 w_k1_KO_s106 + 1.62264 w_k1_MAT_s106
                                 + 1.10982 w_k1_MSFT_s106
                                 + 1.10026 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.26125 w_k1_ADM_s106
                                 + 1.13644 w_k1_AXP_s106
                                 + 1.20399 w_k1_DIS_s106 + 1.0896 w_k1_FDX_s106
                                 + 1.34751 w_k1_IBM_s106
                                 + 1.09567 w_k1_INTC_s106
                                 + 1.35096 w_k1_KO_s106 + 1.80056 w_k1_MAT_s106
                                 + 1.09993 w_k1_MSFT_s106
                                 + 1.24159 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.0933 w_k1_ADM_s106
                                 + 0.79361 w_k1_AXP_s106
                                 + 0.685286 w_k1_DIS_s106
                                 + 0.795663 w_k1_FDX_s106
                                 + 1.08077 w_k1_IBM_s106
                                 + 0.765261 w_k1_INTC_s106
                                 + 0.949137 w_k1_KO_s106
                                 + 1.09358 w_k1_MAT_s106
                                 + 0.764651 w_k1_MSFT_s106
                                 + 0.946678 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.691827 w_k1_ADM_s107
                                 + 0.690916 w_k1_AXP_s107
                                 + 0.581758 w_k1_DIS_s107
                                 + 0.845102 w_k1_FDX_s107
                                 + 0.712604 w_k1_IBM_s107
                                 + 0.945494 w_k1_INTC_s107
                                 + 0.696274 w_k1_KO_s107
                                 + 0.609504 w_k1_MAT_s107
                                 + 0.833648 w_k1_MSFT_s107
                                 + 0.804919 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.08267 w_k1_ADM_s107
                                 + 1.54718 w_k1_AXP_s107
                                 + 1.52562 w_k1_DIS_s107
                                 + 1.50984 w_k1_FDX_s107
                                 + 1.07183 w_k1_IBM_s107
                                 + 1.57358 w_k1_INTC_s107
                                 + 1.22107 w_k1_KO_s107 + 1.27429 w_k1_MAT_s107
                                 + 1.47101 w_k1_MSFT_s107
                                 + 1.17132 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.748615 w_k1_ADM_s107
                                 + 0.936205 w_k1_AXP_s107
                                 + 0.817104 w_k1_DIS_s107
                                 + 0.814355 w_k1_FDX_s107
                                 + 1.20327 w_k1_IBM_s107
                                 + 1.08269 w_k1_INTC_s107
                                 + 0.965951 w_k1_KO_s107
                                 + 1.16111 w_k1_MAT_s107
                                 + 0.889926 w_k1_MSFT_s107
                                 + 1.06338 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.18439 w_k1_ADM_s107
                                 + 1.0008 w_k1_AXP_s107
                                 + 0.996158 w_k1_DIS_s107
                                 + 1.13828 w_k1_FDX_s107
                                 + 0.759558 w_k1_IBM_s107
                                 + 1.07397 w_k1_INTC_s107
                                 + 0.909944 w_k1_KO_s107
                                 + 0.70323 w_k1_MAT_s107
                                 + 1.0633 w_k1_MSFT_s107
                                 + 0.981696 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.861904 w_k1_ADM_s107
                                 + 0.865873 w_k1_AXP_s107
                                 + 0.691904 w_k1_DIS_s107
                                 + 0.775152 w_k1_FDX_s107
                                 + 0.970814 w_k1_IBM_s107
                                 + 0.919568 w_k1_INTC_s107
                                 + 0.972029 w_k1_KO_s107
                                 + 1.00792 w_k1_MAT_s107
                                 + 0.747697 w_k1_MSFT_s107
                                 + 0.933867 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.26909 w_k1_ADM_s107
                                 + 1.16989 w_k1_AXP_s107
                                 + 1.26539 w_k1_DIS_s107
                                 + 1.20012 w_k1_FDX_s107
                                 + 1.44887 w_k1_IBM_s107
                                 + 1.07732 w_k1_INTC_s107
                                 + 1.39341 w_k1_KO_s107 + 1.28608 w_k1_MAT_s107
                                 + 1.13483 w_k1_MSFT_s107
                                 + 1.15911 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.643906 w_k1_ADM_s107
                                 + 0.834972 w_k1_AXP_s107
                                 + 0.882904 w_k1_DIS_s107
                                 + 0.894728 w_k1_FDX_s107
                                 + 1.45664 w_k1_IBM_s107
                                 + 0.990736 w_k1_INTC_s107
                                 + 1.18251 w_k1_KO_s107 + 1.18131 w_k1_MAT_s107
                                 + 0.901896 w_k1_MSFT_s107
                                 + 1.15315 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.975219 w_k1_ADM_s107
                                 + 0.946168 w_k1_AXP_s107
                                 + 1.30819 w_k1_DIS_s107
                                 + 1.08755 w_k1_FDX_s107
                                 + 1.02434 w_k1_IBM_s107
                                 + 1.13368 w_k1_INTC_s107 + 1.1527 w_k1_KO_s107
                                 + 1.12924 w_k1_MAT_s107
                                 + 1.14613 w_k1_MSFT_s107
                                 + 1.11761 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.987163 w_k1_ADM_s110
                                 + 1.09138 w_k1_AXP_s110
                                 + 0.751374 w_k1_DIS_s110
                                 + 0.879406 w_k1_FDX_s110
                                 + 1.06211 w_k1_IBM_s110
                                 + 1.07231 w_k1_INTC_s110
                                 + 0.826042 w_k1_KO_s110
                                 + 1.01746 w_k1_MAT_s110
                                 + 0.944582 w_k1_MSFT_s110
                                 + 0.920205 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.20395 w_k1_ADM_s110
                                 + 1.4441 w_k1_AXP_s110 + 1.23784 w_k1_DIS_s110
                                 + 1.22391 w_k1_FDX_s110
                                 + 0.933406 w_k1_IBM_s110
                                 + 1.15438 w_k1_INTC_s110
                                 + 0.975202 w_k1_KO_s110
                                 + 1.00848 w_k1_MAT_s110
                                 + 1.17308 w_k1_MSFT_s110 + 1.0898 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.925805 w_k1_ADM_s110
                                 + 0.664448 w_k1_AXP_s110
                                 + 0.791888 w_k1_DIS_s110
                                 + 0.779421 w_k1_FDX_s110
                                 + 1.3875 w_k1_IBM_s110
                                 + 0.826891 w_k1_INTC_s110
                                 + 0.994968 w_k1_KO_s110
                                 + 1.11436 w_k1_MAT_s110
                                 + 0.936997 w_k1_MSFT_s110
                                 + 0.838418 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.78508 w_k1_ADM_s110
                                 + 0.846604 w_k1_AXP_s110
                                 + 0.835403 w_k1_DIS_s110
                                 + 0.799971 w_k1_FDX_s110
                                 + 0.8634 w_k1_IBM_s110
                                 + 0.933089 w_k1_INTC_s110
                                 + 0.833793 w_k1_KO_s110
                                 + 0.762853 w_k1_MAT_s110
                                 + 0.880129 w_k1_MSFT_s110
                                 + 0.847403 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.901233 w_k1_ADM_s110
                                 + 0.69871 w_k1_AXP_s110
                                 + 0.61347 w_k1_DIS_s110
                                 + 0.76468 w_k1_FDX_s110
                                 + 0.687439 w_k1_IBM_s110
                                 + 0.866525 w_k1_INTC_s110
                                 + 0.820581 w_k1_KO_s110
                                 + 0.714494 w_k1_MAT_s110
                                 + 0.913309 w_k1_MSFT_s110
                                 + 0.806653 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.876427 w_k1_ADM_s110
                                 + 1.02133 w_k1_AXP_s110
                                 + 1.17699 w_k1_DIS_s110
                                 + 1.03678 w_k1_FDX_s110
                                 + 1.38822 w_k1_IBM_s110
                                 + 1.15087 w_k1_INTC_s110
                                 + 1.23775 w_k1_KO_s110 + 1.24952 w_k1_MAT_s110
                                 + 1.13632 w_k1_MSFT_s110
                                 + 0.970067 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.04248 w_k1_ADM_s110
                                 + 0.836148 w_k1_AXP_s110
                                 + 0.98038 w_k1_DIS_s110
                                 + 1.03656 w_k1_FDX_s110
                                 + 0.675523 w_k1_IBM_s110
                                 + 0.919898 w_k1_INTC_s110
                                 + 0.894336 w_k1_KO_s110
                                 + 0.832374 w_k1_MAT_s110
                                 + 1.08846 w_k1_MSFT_s110
                                 + 0.944289 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.5891 w_k1_ADM_s110
                                 + 1.28516 w_k1_AXP_s110
                                 + 1.62284 w_k1_DIS_s110
                                 + 1.39232 w_k1_FDX_s110
                                 + 1.07711 w_k1_IBM_s110
                                 + 1.15742 w_k1_INTC_s110
                                 + 1.06423 w_k1_KO_s110 + 1.01143 w_k1_MAT_s110
                                 + 1.39242 w_k1_MSFT_s110
                                 + 1.12352 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.790954 w_k1_ADM_s111
                                 + 1.10698 w_k1_AXP_s111
                                 + 0.819998 w_k1_DIS_s111
                                 + 0.794174 w_k1_FDX_s111
                                 + 1.15916 w_k1_IBM_s111
                                 + 0.936732 w_k1_INTC_s111
                                 + 1.06922 w_k1_KO_s111 + 1.18519 w_k1_MAT_s111
                                 + 0.825055 w_k1_MSFT_s111
                                 + 0.899039 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.10633 w_k1_ADM_s111
                                 + 1.68597 w_k1_AXP_s111
                                 + 1.56447 w_k1_DIS_s111
                                 + 1.20242 w_k1_FDX_s111
                                 + 1.70514 w_k1_IBM_s111
                                 + 1.52441 w_k1_INTC_s111
                                 + 1.46377 w_k1_KO_s111 + 1.29736 w_k1_MAT_s111
                                 + 1.11699 w_k1_MSFT_s111 + 1.2022 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.14466 w_k1_ADM_s111
                                 + 1.07364 w_k1_AXP_s111
                                 + 1.58835 w_k1_DIS_s111
                                 + 1.24831 w_k1_FDX_s111
                                 + 1.31657 w_k1_IBM_s111
                                 + 1.20206 w_k1_INTC_s111
                                 + 1.35183 w_k1_KO_s111
                                 + 0.960874 w_k1_MAT_s111
                                 + 1.21396 w_k1_MSFT_s111
                                 + 1.08591 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.9439 w_k1_ADM_s111
                                 + 1.00876 w_k1_AXP_s111
                                 + 0.861237 w_k1_DIS_s111
                                 + 0.98293 w_k1_FDX_s111
                                 + 1.01471 w_k1_IBM_s111
                                 + 1.04182 w_k1_INTC_s111
                                 + 0.946423 w_k1_KO_s111
                                 + 0.996119 w_k1_MAT_s111
                                 + 0.904952 w_k1_MSFT_s111
                                 + 0.943418 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.810389 w_k1_ADM_s111
                                 + 0.784785 w_k1_AXP_s111
                                 + 0.974131 w_k1_DIS_s111
                                 + 0.839573 w_k1_FDX_s111
                                 + 0.849929 w_k1_IBM_s111
                                 + 0.793274 w_k1_INTC_s111
                                 + 0.886286 w_k1_KO_s111
                                 + 0.686177 w_k1_MAT_s111
                                 + 0.808241 w_k1_MSFT_s111
                                 + 0.84192 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.938948 w_k1_ADM_s111
                                 + 1.06479 w_k1_AXP_s111
                                 + 1.15861 w_k1_DIS_s111
                                 + 0.979963 w_k1_FDX_s111
                                 + 1.52254 w_k1_IBM_s111
                                 + 1.39504 w_k1_INTC_s111
                                 + 1.26622 w_k1_KO_s111 + 1.02903 w_k1_MAT_s111
                                 + 1.00202 w_k1_MSFT_s111
                                 + 1.03398 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.11094 w_k1_ADM_s111
                                 + 1.33122 w_k1_AXP_s111
                                 + 1.34098 w_k1_DIS_s111
                                 + 1.31831 w_k1_FDX_s111
                                 + 0.832689 w_k1_IBM_s111
                                 + 1.33403 w_k1_INTC_s111
                                 + 1.00852 w_k1_KO_s111
                                 + 0.735079 w_k1_MAT_s111
                                 + 1.05021 w_k1_MSFT_s111
                                 + 1.03478 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.893855 w_k1_ADM_s111
                                 + 0.709836 w_k1_AXP_s111
                                 + 1.02036 w_k1_DIS_s111
                                 + 0.998436 w_k1_FDX_s111
                                 + 0.637093 w_k1_IBM_s111
                                 + 1.04997 w_k1_INTC_s111
                                 + 0.891703 w_k1_KO_s111
                                 + 0.489215 w_k1_MAT_s111
                                 + 0.994886 w_k1_MSFT_s111
                                 + 0.918874 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.962491 w_k1_ADM_s112
                                 + 1.23227 w_k1_AXP_s112
                                 + 1.66872 w_k1_DIS_s112
                                 + 1.17059 w_k1_FDX_s112
                                 + 1.33234 w_k1_IBM_s112
                                 + 1.51748 w_k1_INTC_s112
                                 + 1.46652 w_k1_KO_s112 + 1.16359 w_k1_MAT_s112
                                 + 1.31455 w_k1_MSFT_s112
                                 + 1.36498 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.72206 w_k1_ADM_s112
                                 + 0.785286 w_k1_AXP_s112
                                 + 0.753502 w_k1_DIS_s112
                                 + 0.917776 w_k1_FDX_s112
                                 + 0.66118 w_k1_IBM_s112
                                 + 1.23411 w_k1_INTC_s112
                                 + 0.87866 w_k1_KO_s112
                                 + 0.745916 w_k1_MAT_s112
                                 + 1.00014 w_k1_MSFT_s112
                                 + 0.929195 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.809845 w_k1_ADM_s112
                                 + 1.02248 w_k1_AXP_s112
                                 + 0.658092 w_k1_DIS_s112
                                 + 0.871055 w_k1_FDX_s112
                                 + 0.784975 w_k1_IBM_s112
                                 + 1.16285 w_k1_INTC_s112
                                 + 0.743991 w_k1_KO_s112
                                 + 0.770873 w_k1_MAT_s112
                                 + 0.811526 w_k1_MSFT_s112
                                 + 0.989883 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.16291 w_k1_ADM_s112
                                 + 1.16014 w_k1_AXP_s112
                                 + 1.22624 w_k1_DIS_s112
                                 + 1.19278 w_k1_FDX_s112
                                 + 0.826288 w_k1_IBM_s112
                                 + 1.39499 w_k1_INTC_s112
                                 + 0.982798 w_k1_KO_s112
                                 + 0.921775 w_k1_MAT_s112
                                 + 1.24547 w_k1_MSFT_s112
                                 + 1.09228 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.661976 w_k1_ADM_s112
                                 + 0.770731 w_k1_AXP_s112
                                 + 0.635937 w_k1_DIS_s112
                                 + 0.727542 w_k1_FDX_s112
                                 + 0.970561 w_k1_IBM_s112
                                 + 0.797619 w_k1_INTC_s112
                                 + 0.804444 w_k1_KO_s112
                                 + 1.11753 w_k1_MAT_s112
                                 + 0.727957 w_k1_MSFT_s112
                                 + 0.915589 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.650251 w_k1_ADM_s112
                                 + 0.694496 w_k1_AXP_s112
                                 + 0.686054 w_k1_DIS_s112
                                 + 0.741808 w_k1_FDX_s112
                                 + 0.806538 w_k1_IBM_s112
                                 + 1.01011 w_k1_INTC_s112
                                 + 0.875973 w_k1_KO_s112
                                 + 0.625423 w_k1_MAT_s112
                                 + 0.913478 w_k1_MSFT_s112
                                 + 0.865412 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.688852 w_k1_ADM_s112
                                 + 0.914237 w_k1_AXP_s112
                                 + 0.826726 w_k1_DIS_s112
                                 + 0.809818 w_k1_FDX_s112
                                 + 1.47545 w_k1_IBM_s112
                                 + 1.04621 w_k1_INTC_s112
                                 + 1.17928 w_k1_KO_s112 + 1.2817 w_k1_MAT_s112
                                 + 0.888393 w_k1_MSFT_s112
                                 + 0.896127 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.689393 w_k1_ADM_s112
                                 + 0.987988 w_k1_AXP_s112
                                 + 0.914372 w_k1_DIS_s112
                                 + 0.924162 w_k1_FDX_s112
                                 + 1.09359 w_k1_IBM_s112
                                 + 1.3816 w_k1_INTC_s112 + 1.07539 w_k1_KO_s112
                                 + 1.17 w_k1_MAT_s112 + 1.0935 w_k1_MSFT_s112
                                 + 1.0384 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.01142 w_k1_ADM_s113
                                 + 1.07409 w_k1_AXP_s113
                                 + 1.00622 w_k1_DIS_s113
                                 + 0.963367 w_k1_FDX_s113
                                 + 1.41184 w_k1_IBM_s113
                                 + 0.907879 w_k1_INTC_s113
                                 + 1.34036 w_k1_KO_s113 + 1.1366 w_k1_MAT_s113
                                 + 0.916743 w_k1_MSFT_s113
                                 + 1.11261 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.38654 w_k1_ADM_s113
                                 + 1.62339 w_k1_AXP_s113 + 1.5692 w_k1_DIS_s113
                                 + 1.36761 w_k1_FDX_s113
                                 + 1.13778 w_k1_IBM_s113
                                 + 1.19224 w_k1_INTC_s113
                                 + 1.46693 w_k1_KO_s113 + 1.47479 w_k1_MAT_s113
                                 + 1.35645 w_k1_MSFT_s113
                                 + 1.28199 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.91609 w_k1_ADM_s113
                                 + 1.03692 w_k1_AXP_s113
                                 + 1.28289 w_k1_DIS_s113
                                 + 1.03756 w_k1_FDX_s113
                                 + 1.22829 w_k1_IBM_s113
                                 + 1.22398 w_k1_INTC_s113
                                 + 1.40795 w_k1_KO_s113 + 1.14268 w_k1_MAT_s113
                                 + 1.20704 w_k1_MSFT_s113
                                 + 1.20484 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.05435 w_k1_ADM_s113
                                 + 1.08951 w_k1_AXP_s113
                                 + 0.986235 w_k1_DIS_s113
                                 + 1.10623 w_k1_FDX_s113
                                 + 0.971744 w_k1_IBM_s113
                                 + 1.07416 w_k1_INTC_s113
                                 + 1.11234 w_k1_KO_s113
                                 + 0.782997 w_k1_MAT_s113
                                 + 1.07616 w_k1_MSFT_s113 + 1.0474 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.12927 w_k1_ADM_s113
                                 + 1.19102 w_k1_AXP_s113
                                 + 1.13869 w_k1_DIS_s113
                                 + 1.31284 w_k1_FDX_s113
                                 + 0.687489 w_k1_IBM_s113
                                 + 1.13263 w_k1_INTC_s113
                                 + 0.926113 w_k1_KO_s113
                                 + 0.584881 w_k1_MAT_s113
                                 + 1.15659 w_k1_MSFT_s113
                                 + 1.03193 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.919993 w_k1_ADM_s113
                                 + 0.791308 w_k1_AXP_s113
                                 + 0.836374 w_k1_DIS_s113
                                 + 0.934988 w_k1_FDX_s113
                                 + 0.729351 w_k1_IBM_s113
                                 + 0.853658 w_k1_INTC_s113
                                 + 1.0004 w_k1_KO_s113 + 0.665263 w_k1_MAT_s113
                                 + 0.935951 w_k1_MSFT_s113
                                 + 0.861882 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.32004 w_k1_ADM_s113
                                 + 1.28157 w_k1_AXP_s113
                                 + 1.43956 w_k1_DIS_s113
                                 + 1.21086 w_k1_FDX_s113
                                 + 1.47125 w_k1_IBM_s113
                                 + 1.1862 w_k1_INTC_s113 + 1.557 w_k1_KO_s113
                                 + 1.1042 w_k1_MAT_s113
                                 + 1.33193 w_k1_MSFT_s113 + 1.1633 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.753801 w_k1_ADM_s113
                                 + 0.792342 w_k1_AXP_s113
                                 + 0.626622 w_k1_DIS_s113
                                 + 0.788816 w_k1_FDX_s113
                                 + 0.867482 w_k1_IBM_s113
                                 + 0.816811 w_k1_INTC_s113
                                 + 0.811877 w_k1_KO_s113
                                 + 0.661828 w_k1_MAT_s113
                                 + 0.796783 w_k1_MSFT_s113
                                 + 0.89744 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.691465 w_k1_ADM_s114
                                 + 0.568361 w_k1_AXP_s114
                                 + 0.591443 w_k1_DIS_s114
                                 + 0.706578 w_k1_FDX_s114
                                 + 0.715172 w_k1_IBM_s114
                                 + 0.764532 w_k1_INTC_s114
                                 + 0.623704 w_k1_KO_s114
                                 + 0.694348 w_k1_MAT_s114
                                 + 0.78338 w_k1_MSFT_s114
                                 + 0.749216 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.21137 w_k1_ADM_s114
                                 + 1.65967 w_k1_AXP_s114
                                 + 1.21679 w_k1_DIS_s114
                                 + 1.15486 w_k1_FDX_s114 + 1.7564 w_k1_IBM_s114
                                 + 1.28723 w_k1_INTC_s114
                                 + 1.40918 w_k1_KO_s114 + 1.838 w_k1_MAT_s114
                                 + 1.17596 w_k1_MSFT_s114
                                 + 1.07863 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.767343 w_k1_ADM_s114
                                 + 0.783005 w_k1_AXP_s114
                                 + 0.836146 w_k1_DIS_s114
                                 + 1.00952 w_k1_FDX_s114
                                 + 0.751441 w_k1_IBM_s114
                                 + 0.841287 w_k1_INTC_s114
                                 + 0.823989 w_k1_KO_s114
                                 + 0.887896 w_k1_MAT_s114
                                 + 0.973477 w_k1_MSFT_s114
                                 + 0.859533 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.896195 w_k1_ADM_s114
                                 + 0.983543 w_k1_AXP_s114
                                 + 0.679522 w_k1_DIS_s114
                                 + 0.920888 w_k1_FDX_s114
                                 + 0.585874 w_k1_IBM_s114
                                 + 0.953571 w_k1_INTC_s114
                                 + 0.662753 w_k1_KO_s114
                                 + 0.853642 w_k1_MAT_s114
                                 + 0.834074 w_k1_MSFT_s114
                                 + 0.864415 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0641 w_k1_ADM_s114
                                 + 1.48394 w_k1_AXP_s114
                                 + 1.38358 w_k1_DIS_s114
                                 + 1.21158 w_k1_FDX_s114
                                 + 1.35001 w_k1_IBM_s114
                                 + 1.19034 w_k1_INTC_s114
                                 + 1.23991 w_k1_KO_s114 + 1.71339 w_k1_MAT_s114
                                 + 1.33351 w_k1_MSFT_s114
                                 + 1.14271 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.807311 w_k1_ADM_s114
                                 + 0.762654 w_k1_AXP_s114
                                 + 0.891035 w_k1_DIS_s114
                                 + 0.854961 w_k1_FDX_s114
                                 + 1.1847 w_k1_IBM_s114
                                 + 0.862476 w_k1_INTC_s114
                                 + 0.924868 w_k1_KO_s114
                                 + 1.04944 w_k1_MAT_s114
                                 + 0.865717 w_k1_MSFT_s114
                                 + 0.821172 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.14071 w_k1_ADM_s114
                                 + 1.15808 w_k1_AXP_s114
                                 + 1.20274 w_k1_DIS_s114
                                 + 1.24343 w_k1_FDX_s114
                                 + 0.98691 w_k1_IBM_s114
                                 + 1.05664 w_k1_INTC_s114
                                 + 1.07352 w_k1_KO_s114 + 1.19611 w_k1_MAT_s114
                                 + 1.16906 w_k1_MSFT_s114
                                 + 1.12955 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.963018 w_k1_ADM_s114
                                 + 0.994396 w_k1_AXP_s114
                                 + 0.787615 w_k1_DIS_s114
                                 + 0.921244 w_k1_FDX_s114
                                 + 0.999643 w_k1_IBM_s114
                                 + 1.0004 w_k1_INTC_s114
                                 + 0.845036 w_k1_KO_s114
                                 + 1.40168 w_k1_MAT_s114
                                 + 1.01073 w_k1_MSFT_s114
                                 + 0.949753 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.2447 w_k1_ADM_s115
                                 + 1.12941 w_k1_AXP_s115 + 1.334 w_k1_DIS_s115
                                 + 1.12722 w_k1_FDX_s115
                                 + 1.33283 w_k1_IBM_s115
                                 + 1.23161 w_k1_INTC_s115
                                 + 1.42985 w_k1_KO_s115 + 1.21673 w_k1_MAT_s115
                                 + 1.08955 w_k1_MSFT_s115
                                 + 1.19562 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.976199 w_k1_ADM_s115
                                 + 0.978325 w_k1_AXP_s115
                                 + 0.861658 w_k1_DIS_s115
                                 + 0.820019 w_k1_FDX_s115
                                 + 1.42232 w_k1_IBM_s115
                                 + 0.949895 w_k1_INTC_s115
                                 + 1.11383 w_k1_KO_s115 + 1.41658 w_k1_MAT_s115
                                 + 0.819477 w_k1_MSFT_s115
                                 + 0.983204 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.26896 w_k1_ADM_s115
                                 + 1.27323 w_k1_AXP_s115
                                 + 1.12263 w_k1_DIS_s115
                                 + 1.11827 w_k1_FDX_s115
                                 + 0.782671 w_k1_IBM_s115
                                 + 1.1076 w_k1_INTC_s115 + 1.00736 w_k1_KO_s115
                                 + 1.12557 w_k1_MAT_s115
                                 + 1.07485 w_k1_MSFT_s115
                                 + 1.02523 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.1832 w_k1_ADM_s115
                                 + 1.15896 w_k1_AXP_s115
                                 + 1.45232 w_k1_DIS_s115
                                 + 1.13169 w_k1_FDX_s115
                                 + 1.08278 w_k1_IBM_s115
                                 + 0.989594 w_k1_INTC_s115
                                 + 1.28161 w_k1_KO_s115 + 1.05497 w_k1_MAT_s115
                                 + 1.05906 w_k1_MSFT_s115
                                 + 1.03165 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.979937 w_k1_ADM_s115
                                 + 1.16885 w_k1_AXP_s115
                                 + 1.21873 w_k1_DIS_s115
                                 + 0.946473 w_k1_FDX_s115
                                 + 1.63829 w_k1_IBM_s115
                                 + 1.28991 w_k1_INTC_s115
                                 + 1.33256 w_k1_KO_s115 + 1.31605 w_k1_MAT_s115
                                 + 1.02759 w_k1_MSFT_s115
                                 + 1.00315 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.04648 w_k1_ADM_s115
                                 + 1.01377 w_k1_AXP_s115
                                 + 0.837805 w_k1_DIS_s115
                                 + 0.861866 w_k1_FDX_s115
                                 + 0.968996 w_k1_IBM_s115
                                 + 0.851361 w_k1_INTC_s115
                                 + 0.970747 w_k1_KO_s115
                                 + 1.13731 w_k1_MAT_s115
                                 + 0.786375 w_k1_MSFT_s115
                                 + 0.955509 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.868032 w_k1_ADM_s115
                                 + 0.797284 w_k1_AXP_s115
                                 + 0.672094 w_k1_DIS_s115
                                 + 0.682279 w_k1_FDX_s115
                                 + 1.03482 w_k1_IBM_s115
                                 + 0.728386 w_k1_INTC_s115
                                 + 0.944988 w_k1_KO_s115
                                 + 0.753793 w_k1_MAT_s115
                                 + 0.682716 w_k1_MSFT_s115
                                 + 0.835359 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.9184 w_k1_ADM_s115
                                 + 0.767718 w_k1_AXP_s115
                                 + 0.915165 w_k1_DIS_s115
                                 + 0.954443 w_k1_FDX_s115
                                 + 0.614374 w_k1_IBM_s115
                                 + 0.962648 w_k1_INTC_s115
                                 + 0.797239 w_k1_KO_s115
                                 + 0.579473 w_k1_MAT_s115
                                 + 0.852316 w_k1_MSFT_s115
                                 + 0.852288 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.07379 w_k1_ADM_s116
                                 + 0.945802 w_k1_AXP_s116
                                 + 1.10611 w_k1_DIS_s116
                                 + 0.978677 w_k1_FDX_s116
                                 + 1.03608 w_k1_IBM_s116
                                 + 1.05425 w_k1_INTC_s116 + 1.1522 w_k1_KO_s116
                                 + 0.973449 w_k1_MAT_s116
                                 + 1.12406 w_k1_MSFT_s116
                                 + 1.00861 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.31817 w_k1_ADM_s116
                                 + 1.37983 w_k1_AXP_s116
                                 + 1.83455 w_k1_DIS_s116
                                 + 1.38685 w_k1_FDX_s116
                                 + 1.55341 w_k1_IBM_s116
                                 + 1.53681 w_k1_INTC_s116
                                 + 1.58125 w_k1_KO_s116 + 1.25429 w_k1_MAT_s116
                                 + 1.41673 w_k1_MSFT_s116
                                 + 1.23898 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.01624 w_k1_ADM_s116
                                 + 1.04387 w_k1_AXP_s116
                                 + 0.953762 w_k1_DIS_s116
                                 + 0.854917 w_k1_FDX_s116
                                 + 1.21922 w_k1_IBM_s116
                                 + 0.857943 w_k1_INTC_s116
                                 + 1.04821 w_k1_KO_s116 + 1.17485 w_k1_MAT_s116
                                 + 0.98926 w_k1_MSFT_s116
                                 + 0.947584 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.799683 w_k1_ADM_s116
                                 + 0.691752 w_k1_AXP_s116
                                 + 0.669927 w_k1_DIS_s116
                                 + 0.738021 w_k1_FDX_s116
                                 + 0.751946 w_k1_IBM_s116
                                 + 0.934732 w_k1_INTC_s116
                                 + 0.671659 w_k1_KO_s116
                                 + 0.627984 w_k1_MAT_s116
                                 + 0.873358 w_k1_MSFT_s116
                                 + 0.851941 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.21586 w_k1_ADM_s116
                                 + 1.01456 w_k1_AXP_s116
                                 + 0.869665 w_k1_DIS_s116
                                 + 0.946287 w_k1_FDX_s116
                                 + 0.830749 w_k1_IBM_s116
                                 + 1.09923 w_k1_INTC_s116
                                 + 0.959872 w_k1_KO_s116
                                 + 0.640944 w_k1_MAT_s116
                                 + 0.979355 w_k1_MSFT_s116
                                 + 0.916486 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.42534 w_k1_ADM_s116
                                 + 1.41896 w_k1_AXP_s116
                                 + 1.58122 w_k1_DIS_s116 + 1.4012 w_k1_FDX_s116
                                 + 1.09857 w_k1_IBM_s116
                                 + 1.44999 w_k1_INTC_s116
                                 + 1.45908 w_k1_KO_s116
                                 + 0.962798 w_k1_MAT_s116
                                 + 1.49033 w_k1_MSFT_s116
                                 + 1.27358 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.25332 w_k1_ADM_s116
                                 + 1.26297 w_k1_AXP_s116
                                 + 1.14944 w_k1_DIS_s116 + 1.1861 w_k1_FDX_s116
                                 + 1.04891 w_k1_IBM_s116
                                 + 1.11493 w_k1_INTC_s116
                                 + 1.14311 w_k1_KO_s116 + 1.12301 w_k1_MAT_s116
                                 + 1.0186 w_k1_MSFT_s116 + 1.11629 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.11125 w_k1_ADM_s116
                                 + 1.21097 w_k1_AXP_s116
                                 + 1.20411 w_k1_DIS_s116
                                 + 1.04839 w_k1_FDX_s116
                                 + 1.67922 w_k1_IBM_s116
                                 + 1.02324 w_k1_INTC_s116
                                 + 1.13649 w_k1_KO_s116 + 1.27498 w_k1_MAT_s116
                                 + 1.03648 w_k1_MSFT_s116 + 1.1982 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.953504 w_k1_ADM_s117
                                 + 0.711862 w_k1_AXP_s117
                                 + 0.849417 w_k1_DIS_s117
                                 + 0.902991 w_k1_FDX_s117
                                 + 0.835504 w_k1_IBM_s117
                                 + 0.904588 w_k1_INTC_s117
                                 + 0.978154 w_k1_KO_s117
                                 + 0.933226 w_k1_MAT_s117
                                 + 0.985583 w_k1_MSFT_s117
                                 + 0.963802 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.04104 w_k1_ADM_s117
                                 + 0.932994 w_k1_AXP_s117
                                 + 0.708468 w_k1_DIS_s117
                                 + 0.858042 w_k1_FDX_s117
                                 + 0.629323 w_k1_IBM_s117
                                 + 0.919655 w_k1_INTC_s117
                                 + 0.737471 w_k1_KO_s117
                                 + 0.868368 w_k1_MAT_s117
                                 + 0.846099 w_k1_MSFT_s117
                                 + 0.951149 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.874903 w_k1_ADM_s117
                                 + 0.776803 w_k1_AXP_s117
                                 + 0.75529 w_k1_DIS_s117
                                 + 0.733838 w_k1_FDX_s117
                                 + 1.30876 w_k1_IBM_s117
                                 + 0.840661 w_k1_INTC_s117
                                 + 1.10213 w_k1_KO_s117 + 1.63619 w_k1_MAT_s117
                                 + 0.809161 w_k1_MSFT_s117
                                 + 1.02784 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.41811 w_k1_ADM_s117
                                 + 1.4469 w_k1_AXP_s117 + 1.52151 w_k1_DIS_s117
                                 + 1.22457 w_k1_FDX_s117
                                 + 1.64438 w_k1_IBM_s117
                                 + 0.996302 w_k1_INTC_s117
                                 + 1.42598 w_k1_KO_s117 + 2.32716 w_k1_MAT_s117
                                 + 1.05312 w_k1_MSFT_s117 + 1.3112 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.651052 w_k1_ADM_s117
                                 + 0.575051 w_k1_AXP_s117
                                 + 0.81016 w_k1_DIS_s117
                                 + 0.827141 w_k1_FDX_s117
                                 + 0.587145 w_k1_IBM_s117
                                 + 0.94017 w_k1_INTC_s117
                                 + 0.800096 w_k1_KO_s117
                                 + 0.650673 w_k1_MAT_s117
                                 + 0.87705 w_k1_MSFT_s117
                                 + 0.899165 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.09477 w_k1_ADM_s117
                                 + 1.31867 w_k1_AXP_s117
                                 + 1.19174 w_k1_DIS_s117
                                 + 1.23425 w_k1_FDX_s117
                                 + 0.811601 w_k1_IBM_s117
                                 + 1.10324 w_k1_INTC_s117
                                 + 0.968365 w_k1_KO_s117
                                 + 1.13748 w_k1_MAT_s117
                                 + 1.18469 w_k1_MSFT_s117
                                 + 1.04632 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.03306 w_k1_ADM_s117
                                 + 1.01464 w_k1_AXP_s117
                                 + 1.17838 w_k1_DIS_s117
                                 + 1.00815 w_k1_FDX_s117
                                 + 1.13498 w_k1_IBM_s117
                                 + 1.08094 w_k1_INTC_s117
                                 + 1.19698 w_k1_KO_s117 + 1.35283 w_k1_MAT_s117
                                 + 1.09156 w_k1_MSFT_s117
                                 + 1.08076 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.02094 w_k1_ADM_s117
                                 + 0.978009 w_k1_AXP_s117
                                 + 0.719448 w_k1_DIS_s117
                                 + 0.810961 w_k1_FDX_s117
                                 + 1.0398 w_k1_IBM_s117
                                 + 0.749147 w_k1_INTC_s117
                                 + 0.828537 w_k1_KO_s117
                                 + 1.78442 w_k1_MAT_s117
                                 + 0.865967 w_k1_MSFT_s117
                                 + 0.981384 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0207605642380176 w_k1_ADM_s000
                                 + 0.0175078885326655 w_k1_ADM_s001
                                 + 0.0104530617622774 w_k1_ADM_s002
                                 + 0.0369933968518119 w_k1_ADM_s003
                                 + 0.00796629664994905 w_k1_ADM_s004
                                 + 0.0122935591598759 w_k1_ADM_s005
                                 + 0.0193448759194381 w_k1_ADM_s006
                                 + 0.0129256079629892 w_k1_ADM_s007
                                 + 0.0162912470456776 w_k1_ADM_s010
                                 + 0.0274895742351026 w_k1_ADM_s011
                                 + 0.0175337306182031 w_k1_ADM_s012
                                 + 0.0334734800987947 w_k1_ADM_s013
                                 + 0.0075678259017955 w_k1_ADM_s014
                                 + 0.0120577332100902 w_k1_ADM_s015
                                 + 0.0413712388812169 w_k1_ADM_s016
                                 + 0.0161771962177459 w_k1_ADM_s017
                                 + 0.00790128623860062 w_k1_ADM_s100
                                 + 0.0272589393811937 w_k1_ADM_s101
                                 + 0.0128303835606427 w_k1_ADM_s102
                                 + 0.016319592769636 w_k1_ADM_s103
                                 + 0.044571129419204 w_k1_ADM_s104
                                 + 0.018517363444395 w_k1_ADM_s105
                                 + 0.0187360450866824 w_k1_ADM_s106
                                 + 0.0213774653015145 w_k1_ADM_s107
                                 + 0.046084945950838 w_k1_ADM_s110
                                 + 0.0908192706560314 w_k1_ADM_s111
                                 + 0.0244399518633802 w_k1_ADM_s112
                                 + 0.0959066614315255 w_k1_ADM_s113
                                 + 0.0283542799778589 w_k1_ADM_s114
                                 + 0.0551008937968936 w_k1_ADM_s115
                                 + 0.103828615888552 w_k1_ADM_s116
                                 + 0.10115427027769 w_k1_ADM_s117
                                 + 0.0240461864340477 w_k1_AXP_s000
                                 + 0.0150257066843512 w_k1_AXP_s001
                                 + 0.00928165259393493 w_k1_AXP_s002
                                 + 0.0395675970536077 w_k1_AXP_s003
                                 + 0.00800582812428643 w_k1_AXP_s004
                                 + 0.00955205515658013 w_k1_AXP_s005
                                 + 0.0193789230679457 w_k1_AXP_s006
                                 + 0.0132588070395587 w_k1_AXP_s007
                                 + 0.0169752382203083 w_k1_AXP_s010
                                 + 0.0290158495759585 w_k1_AXP_s011
                                 + 0.0152120855883053 w_k1_AXP_s012
                                 + 0.0350546510275883 w_k1_AXP_s013
                                 + 0.00904745369902699 w_k1_AXP_s014
                                 + 0.0145200190073147 w_k1_AXP_s015
                                 + 0.0418079381267213 w_k1_AXP_s016
                                 + 0.0139652204346521 w_k1_AXP_s017
                                 + 0.00831700537027475 w_k1_AXP_s100
                                 + 0.0238137385167595 w_k1_AXP_s101
                                 + 0.0136050960933131 w_k1_AXP_s102
                                 + 0.0152066978316496 w_k1_AXP_s103
                                 + 0.0405470945660073 w_k1_AXP_s104
                                 + 0.0185723150921288 w_k1_AXP_s105
                                 + 0.0166899328867171 w_k1_AXP_s106
                                 + 0.0218646693960027 w_k1_AXP_s107
                                 + 0.0455113619326603 w_k1_AXP_s110
                                 + 0.098229882630968 w_k1_AXP_s111
                                 + 0.0288652092856951 w_k1_AXP_s112
                                 + 0.0986479902101 w_k1_AXP_s113
                                 + 0.030795494241584 w_k1_AXP_s114
                                 + 0.0535500485531827 w_k1_AXP_s115
                                 + 0.0982685502481051 w_k1_AXP_s116
                                 + 0.0997495462257397 w_k1_AXP_s117
                                 + 0.0231619502762571 w_k1_DIS_s000
                                 + 0.0162451864783981 w_k1_DIS_s001
                                 + 0.00940056905024296 w_k1_DIS_s002
                                 + 0.0393983564119675 w_k1_DIS_s003
                                 + 0.00793071310889165 w_k1_DIS_s004
                                 + 0.00922942073871456 w_k1_DIS_s005
                                 + 0.0215678850784068 w_k1_DIS_s006
                                 + 0.0139092979826869 w_k1_DIS_s007
                                 + 0.0163203060855053 w_k1_DIS_s010
                                 + 0.028073247277593 w_k1_DIS_s011
                                 + 0.0156788409913211 w_k1_DIS_s012
                                 + 0.0396504038419501 w_k1_DIS_s013
                                 + 0.00876066596585135 w_k1_DIS_s014
                                 + 0.0131794829057689 w_k1_DIS_s015
                                 + 0.0424726694418819 w_k1_DIS_s016
                                 + 0.0143999890393685 w_k1_DIS_s017
                                 + 0.00876293428065654 w_k1_DIS_s100
                                 + 0.0244362588561918 w_k1_DIS_s101
                                 + 0.0134101587833855 w_k1_DIS_s102
                                 + 0.016126546557065 w_k1_DIS_s103
                                 + 0.0425566668645647 w_k1_DIS_s104
                                 + 0.0201387891696684 w_k1_DIS_s105
                                 + 0.0173555494039653 w_k1_DIS_s106
                                 + 0.0223057541756595 w_k1_DIS_s107
                                 + 0.0471826036831325 w_k1_DIS_s110
                                 + 0.106356762033996 w_k1_DIS_s111
                                 + 0.0298441198292259 w_k1_DIS_s112
                                 + 0.095176900498661 w_k1_DIS_s113
                                 + 0.0291192938840952 w_k1_DIS_s114
                                 + 0.0550876889662757 w_k1_DIS_s115
                                 + 0.098016733375624 w_k1_DIS_s116
                                 + 0.100801802805399 w_k1_DIS_s117
                                 + 0.0213018390845335 w_k1_FDX_s000
                                 + 0.0160803692512885 w_k1_FDX_s001
                                 + 0.00901049937984951 w_k1_FDX_s002
                                 + 0.0379765731901958 w_k1_FDX_s003
                                 + 0.00755307585845901 w_k1_FDX_s004
                                 + 0.00955870791023456 w_k1_FDX_s005
                                 + 0.0206065414448885 w_k1_FDX_s006
                                 + 0.0131739021170966 w_k1_FDX_s007
                                 + 0.0165070713522746 w_k1_FDX_s010
                                 + 0.0286930481271757 w_k1_FDX_s011
                                 + 0.0152449749172029 w_k1_FDX_s012
                                 + 0.0357036948333209 w_k1_FDX_s013
                                 + 0.00834401358138443 w_k1_FDX_s014
                                 + 0.0119776831115959 w_k1_FDX_s015
                                 + 0.0400187985082195 w_k1_FDX_s016
                                 + 0.0148656540415557 w_k1_FDX_s017
                                 + 0.00856772796344367 w_k1_FDX_s100
                                 + 0.0241123559042938 w_k1_FDX_s101
                                 + 0.0128399673787782 w_k1_FDX_s102
                                 + 0.0155825400467817 w_k1_FDX_s103
                                 + 0.0416019384019245 w_k1_FDX_s104
                                 + 0.0182150164985927 w_k1_FDX_s105
                                 + 0.0164712674454825 w_k1_FDX_s106
                                 + 0.0229060925298486 w_k1_FDX_s107
                                 + 0.0454540757943932 w_k1_FDX_s110
                                 + 0.0977680558157474 w_k1_FDX_s111
                                 + 0.0283484729733195 w_k1_FDX_s112
                                 + 0.0984315192466721 w_k1_FDX_s113
                                 + 0.0304785071428164 w_k1_FDX_s114
                                 + 0.0500343210925456 w_k1_FDX_s115
                                 + 0.0930491542307626 w_k1_FDX_s116
                                 + 0.0969250491336873 w_k1_FDX_s117
                                 + 0.0226070760065708 w_k1_IBM_s000
                                 + 0.0162034515166504 w_k1_IBM_s001
                                 + 0.00889808826803384 w_k1_IBM_s002
                                 + 0.0372230082618618 w_k1_IBM_s003
                                 + 0.00742405819778005 w_k1_IBM_s004
                                 + 0.00964049458237292 w_k1_IBM_s005
                                 + 0.0201202699501249 w_k1_IBM_s006
                                 + 0.0133154447037879 w_k1_IBM_s007
                                 + 0.0163532236750177 w_k1_IBM_s010
                                 + 0.0296054013141703 w_k1_IBM_s011
                                 + 0.0165846269540496 w_k1_IBM_s012
                                 + 0.0378629886722719 w_k1_IBM_s013
                                 + 0.00816917848364733 w_k1_IBM_s014
                                 + 0.0122472745763961 w_k1_IBM_s015
                                 + 0.0414844374287803 w_k1_IBM_s016
                                 + 0.0159235483866965 w_k1_IBM_s017
                                 + 0.00912260397787606 w_k1_IBM_s100
                                 + 0.0239472328533676 w_k1_IBM_s101
                                 + 0.0137166617726719 w_k1_IBM_s102
                                 + 0.0166029555553706 w_k1_IBM_s103
                                 + 0.043437561614133 w_k1_IBM_s104
                                 + 0.018969798726335 w_k1_IBM_s105
                                 + 0.0177246312067653 w_k1_IBM_s106
                                 + 0.0223272845871923 w_k1_IBM_s107
                                 + 0.0470815909631578 w_k1_IBM_s110
                                 + 0.101240133153994 w_k1_IBM_s111
                                 + 0.0296528768426854 w_k1_IBM_s112
                                 + 0.0970003063926892 w_k1_IBM_s113
                                 + 0.0308710411331758 w_k1_IBM_s114
                                 + 0.0535275562696065 w_k1_IBM_s115
                                 + 0.0963730622953423 w_k1_IBM_s116
                                 + 0.0980483642307133 w_k1_IBM_s117
                                 + 0.0207001322279968 w_k1_INTC_s000
                                 + 0.0134761876067466 w_k1_INTC_s001
                                 + 0.00906252092508364 w_k1_INTC_s002
                                 + 0.0368572539541318 w_k1_INTC_s003
                                 + 0.00732564066773016 w_k1_INTC_s004
                                 + 0.0102093279604178 w_k1_INTC_s005
                                 + 0.0235386477579872 w_k1_INTC_s006
                                 + 0.0145812178535408 w_k1_INTC_s007
                                 + 0.0196832679369287 w_k1_INTC_s010
                                 + 0.0280926938179327 w_k1_INTC_s011
                                 + 0.0144037661774442 w_k1_INTC_s012
                                 + 0.0341007274085518 w_k1_INTC_s013
                                 + 0.00991197767189612 w_k1_INTC_s014
                                 + 0.0150885931916071 w_k1_INTC_s015
                                 + 0.0358530175331498 w_k1_INTC_s016
                                 + 0.0155313322290569 w_k1_INTC_s017
                                 + 0.0108697255656134 w_k1_INTC_s100
                                 + 0.0232108837493904 w_k1_INTC_s101
                                 + 0.0118742957170662 w_k1_INTC_s102
                                 + 0.0151619235239938 w_k1_INTC_s103
                                 + 0.0378633675821132 w_k1_INTC_s104
                                 + 0.0149099472654853 w_k1_INTC_s105
                                 + 0.0165587473218656 w_k1_INTC_s106
                                 + 0.0240264603019414 w_k1_INTC_s107
                                 + 0.0467384862161773 w_k1_INTC_s110
                                 + 0.107417635981657 w_k1_INTC_s111
                                 + 0.036755738360769 w_k1_INTC_s112
                                 + 0.0938764820305368 w_k1_INTC_s113
                                 + 0.0295900046113466 w_k1_INTC_s114
                                 + 0.0516991204653456 w_k1_INTC_s115
                                 + 0.0993783065222774 w_k1_INTC_s116
                                 + 0.0960625303961017 w_k1_INTC_s117
                                 + 0.0219609932610455 w_k1_KO_s000
                                 + 0.0158383214782018 w_k1_KO_s001
                                 + 0.00859912699502123 w_k1_KO_s002
                                 + 0.0351381994792508 w_k1_KO_s003
                                 + 0.00764649899648012 w_k1_KO_s004
                                 + 0.0100527619452743 w_k1_KO_s005
                                 + 0.0198962523196974 w_k1_KO_s006
                                 + 0.0136730307147855 w_k1_KO_s007
                                 + 0.0163183884950028 w_k1_KO_s010
                                 + 0.0291086302550995 w_k1_KO_s011
                                 + 0.0151042176365617 w_k1_KO_s012
                                 + 0.0361018717356525 w_k1_KO_s013
                                 + 0.0093354328907818 w_k1_KO_s014
                                 + 0.0125663451425812 w_k1_KO_s015
                                 + 0.0416320085022108 w_k1_KO_s016
                                 + 0.0161019929231901 w_k1_KO_s017
                                 + 0.00934151053240579 w_k1_KO_s100
                                 + 0.0267195616812053 w_k1_KO_s101
                                 + 0.0137486356735072 w_k1_KO_s102
                                 + 0.0150043734475912 w_k1_KO_s103
                                 + 0.0442752342805717 w_k1_KO_s104
                                 + 0.019157153523847 w_k1_KO_s105
                                 + 0.0183821438087998 w_k1_KO_s106
                                 + 0.0228636610744785 w_k1_KO_s107
                                 + 0.044563020577654 w_k1_KO_s110
                                 + 0.101882565108418 w_k1_KO_s111
                                 + 0.0307587715677142 w_k1_KO_s112
                                 + 0.108067151936336 w_k1_KO_s113
                                 + 0.0285636248331113 w_k1_KO_s114
                                 + 0.0560459245922894 w_k1_KO_s115
                                 + 0.0989157722313352 w_k1_KO_s116
                                 + 0.10119821370709 w_k1_KO_s117
                                 + 0.03065983870013 w_k1_MAT_s000
                                 + 0.0186307377782109 w_k1_MAT_s001
                                 + 0.00865832715681755 w_k1_MAT_s002
                                 + 0.0402841438391239 w_k1_MAT_s003
                                 + 0.00725740304089175 w_k1_MAT_s004
                                 + 0.0087439592618207 w_k1_MAT_s005
                                 + 0.0177403450248882 w_k1_MAT_s006
                                 + 0.0124371690179128 w_k1_MAT_s007
                                 + 0.0128216977732424 w_k1_MAT_s010
                                 + 0.0297483117866824 w_k1_MAT_s011
                                 + 0.0142345914990172 w_k1_MAT_s012
                                 + 0.0373578470249673 w_k1_MAT_s013
                                 + 0.0102414428814921 w_k1_MAT_s014
                                 + 0.0110890141302578 w_k1_MAT_s015
                                 + 0.0429990510365217 w_k1_MAT_s016
                                 + 0.0152660118294148 w_k1_MAT_s017
                                 + 0.00786593651926771 w_k1_MAT_s100
                                 + 0.0241742549876245 w_k1_MAT_s101
                                 + 0.0169463729998151 w_k1_MAT_s102
                                 + 0.0121386941300063 w_k1_MAT_s103
                                 + 0.0380068005579208 w_k1_MAT_s104
                                 + 0.0211606895940689 w_k1_MAT_s105
                                 + 0.0203595623650223 w_k1_MAT_s106
                                 + 0.0220094971541656 w_k1_MAT_s107
                                 + 0.0447894999142176 w_k1_MAT_s110
                                 + 0.0831539024600968 w_k1_MAT_s111
                                 + 0.0291296750663718 w_k1_MAT_s112
                                 + 0.0836890712354905 w_k1_MAT_s113
                                 + 0.0360969997572916 w_k1_MAT_s114
                                 + 0.053684653149148 w_k1_MAT_s115
                                 + 0.0858335296181151 w_k1_MAT_s116
                                 + 0.129109992788749 w_k1_MAT_s117
                                 + 0.020999315821937 w_k1_MSFT_s000
                                 + 0.0151966237670918 w_k1_MSFT_s001
                                 + 0.00897673128061625 w_k1_MSFT_s002
                                 + 0.0398684656114636 w_k1_MSFT_s003
                                 + 0.00768855215234315 w_k1_MSFT_s004
                                 + 0.0100375061823564 w_k1_MSFT_s005
                                 + 0.0217450299084219 w_k1_MSFT_s006
                                 + 0.0134990081273961 w_k1_MSFT_s007
                                 + 0.0178533947091925 w_k1_MSFT_s010
                                 + 0.0252360723402752 w_k1_MSFT_s011
                                 + 0.0143172377053244 w_k1_MSFT_s012
                                 + 0.0346631763636051 w_k1_MSFT_s013
                                 + 0.00932546538314766 w_k1_MSFT_s014
                                 + 0.0138871799399734 w_k1_MSFT_s015
                                 + 0.0390136207454856 w_k1_MSFT_s016
                                 + 0.0139845317859075 w_k1_MSFT_s017
                                 + 0.00875564967164201 w_k1_MSFT_s100
                                 + 0.0248899268088344 w_k1_MSFT_s101
                                 + 0.0134040678412129 w_k1_MSFT_s102
                                 + 0.0157983579421331 w_k1_MSFT_s103
                                 + 0.0422546132403266 w_k1_MSFT_s104
                                 + 0.0173854265406436 w_k1_MSFT_s105
                                 + 0.0163168560777789 w_k1_MSFT_s106
                                 + 0.0225790444882429 w_k1_MSFT_s107
                                 + 0.0483847826209892 w_k1_MSFT_s110
                                 + 0.0931722232770892 w_k1_MSFT_s111
                                 + 0.0311929854971693 w_k1_MSFT_s112
                                 + 0.0971374363542836 w_k1_MSFT_s113
                                 + 0.0311622304717301 w_k1_MSFT_s114
                                 + 0.0478306646352424 w_k1_MSFT_s115
                                 + 0.0973140645244255 w_k1_MSFT_s116
                                 + 0.0987244804967261 w_k1_MSFT_s117
                                 + 0.0209745407431648 w_k1_PG_s000
                                 + 0.0156650006216676 w_k1_PG_s001
                                 + 0.0100659106152068 w_k1_PG_s002
                                 + 0.0356623162267714 w_k1_PG_s003
                                 + 0.00730838310746209 w_k1_PG_s004
                                 + 0.00846519144068418 w_k1_PG_s005
                                 + 0.0187259426754774 w_k1_PG_s006
                                 + 0.013661364406981 w_k1_PG_s007
                                 + 0.0169375399129392 w_k1_PG_s010
                                 + 0.0298927177357656 w_k1_PG_s011
                                 + 0.0152174920852137 w_k1_PG_s012
                                 + 0.0345054816143315 w_k1_PG_s013
                                 + 0.00882677205559844 w_k1_PG_s014
                                 + 0.0132819470175896 w_k1_PG_s015
                                 + 0.0377968453033358 w_k1_PG_s016
                                 + 0.0142302183311839 w_k1_PG_s017
                                 + 0.00867885127080215 w_k1_PG_s100
                                 + 0.0243702570205707 w_k1_PG_s101
                                 + 0.0120367299040275 w_k1_PG_s102
                                 + 0.014785762206532 w_k1_PG_s103
                                 + 0.0450728647406436 w_k1_PG_s104
                                 + 0.016328516480522 w_k1_PG_s105
                                 + 0.0174446855736097 w_k1_PG_s106
                                 + 0.0226419772449805 w_k1_PG_s107
                                 + 0.0428714186580077 w_k1_PG_s110
                                 + 0.0937004227467603 w_k1_PG_s111
                                 + 0.0306663278765486 w_k1_PG_s112
                                 + 0.0981299327127007 w_k1_PG_s113
                                 + 0.0289137837736259 w_k1_PG_s114
                                 + 0.0503415189527843 w_k1_PG_s115
                                 + 0.0931837117237274 w_k1_PG_s116
                                 + 0.102010070820878 w_k1_PG_s117 >= 1.07
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
