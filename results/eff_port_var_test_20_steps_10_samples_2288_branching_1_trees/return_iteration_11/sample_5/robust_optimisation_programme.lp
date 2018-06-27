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
 w_k1_ADM_s0_return_constraint:  1.10334 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.862889 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.12652 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.951125 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04007 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.25325 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.947712 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.13137 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.880699 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.00361 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.19877 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.879427 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.970795 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.28447 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.883381 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.996419 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.980965 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.887807 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.705326 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.908586 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.979053 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.25826 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.786938 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.944173 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.962563 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.4436 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.03717 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.714748 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.772063 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.27129 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.18782 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.34358 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.954828 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.670642 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.885584 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.929694 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.08244 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.13637 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23155 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.868849 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.12549 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.849329 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.05533 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.38505 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.71101 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.17581 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.884884 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.845309 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.11261 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.985147 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.48204 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.67848 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.639687 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.21548 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.9082 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.932748 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.704186 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.937088 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.40125 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.35456 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.696255 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.842125 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.797443 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.4579 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.00526 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.505594 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.60545 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.14172 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.383 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.3212 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.80623 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.952649 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.956887 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.842923 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.885352 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.1816 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33477 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.844496 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.25351 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.741895 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.957959 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.44035 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.550407 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.16881 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.615225 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.814077 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.13046 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.922797 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.29496 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.80431 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.567186 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.31864 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.979585 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.864978 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.712701 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.00513 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.4596 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.43956 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.720767 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.816818 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.966978 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.82654 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.03275 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.503938 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.656763 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.32014 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.40266 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.53796 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.0105 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.792101 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.820704 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.755426 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.925326 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.18526 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19222 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.893017 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.13548 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.843988 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.991149 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.21626 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.791114 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.10246 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.769102 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.8782 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.16448 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.885569 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.20441 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.60408 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.813743 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.08975 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.02307 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.820269 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.743237 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.90134 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.18222 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.33299 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.711873 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.02026 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.963897 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.51321 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.974094 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.60324 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.747045 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.23199 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.16696 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.36881 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.925381 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.769465 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.825876 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.896533 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.02337 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.08716 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.11746 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.975697 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.12494 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.876327 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.910329 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.59659 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.524939 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.1074 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.06439 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.854679 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.697065 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.65234 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.41866 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.16426 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.546265 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.31071 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.773057 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.54124 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.88218 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.22844 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.89251 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.941428 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.2623 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.525519 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.852531 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.04499 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.39721 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.862101 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.642544 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.05375 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.4826 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.843854 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.03214 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.28424 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.87131 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.711849 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.556325 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.07112 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18242 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.981642 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.056 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.853409 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.898278 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.25236 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.88513 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.21848 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.909362 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.924181 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04634 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.986881 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.10587 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.44263 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.727447 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.977104 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.867246 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.93127 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.781922 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.974819 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.33776 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.984435 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.86205 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.00583 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.969726 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.45512 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.982942 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.769374 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.756352 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13226 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.03975 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.17542 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.00555 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.755739 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.828061 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.897683 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.841772 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.846047 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21451 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.947608 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.12627 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.852903 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.887413 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.222 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.649983 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.16138 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.79199 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.886142 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.856738 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.21569 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.28092 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.32164 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.601184 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.09711 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.859948 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.21776 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.818111 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.990523 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.39854 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.02437 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.00223 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.758836 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.948592 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.37537 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.2551 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.771841 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.73131 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.20856 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.58965 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.33768 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.08621 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.04043 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.03332 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.803051 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.884451 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.26467 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.1595 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.96149 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.22179 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.812262 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.998316 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.56007 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.536493 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.869765 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.865515 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.89123 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.789236 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.45788 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.47291 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.4251 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.656228 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.3524 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.767656 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.60724 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.994608 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.06866 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.0689 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.31898 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.952308 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.435278 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.812738 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.07539 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.31377 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.824455 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.444942 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.0094 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.70975 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.93614 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.913196 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.29254 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.07251 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.652229 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.626303 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.06667 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14217 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.941745 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16364 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.941125 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.929841 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.15649 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.748121 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.18844 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.738698 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.905658 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.14773 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.916408 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.09647 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.55256 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.737071 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.08596 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.01985 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.788634 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.715521 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.768321 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.18457 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0764 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.771572 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.978101 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.920912 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.53597 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.02795 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.717511 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.763356 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.11786 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.14926 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.27535 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.02364 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.761468 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.885309 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.80133 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.12277 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.02096 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12581 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.950428 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.09682 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.84557 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.940466 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.2477 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.794739 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.0551 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.835735 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.93513 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.961317 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.00117 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.12137 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.19297 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.797946 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.13252 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.03182 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.06244 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.902217 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.00798 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.45227 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.14365 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.905681 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.956018 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.895489 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.2218 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.10875 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.795571 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.798647 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.08599 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.16012 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.08374 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.936011 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.890047 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.880447 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.792667 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.904427 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.07248 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.423000045294842 z_k1_s0000
                                 - 0.172999841461098 z_k1_s0001
                                 - 0.118000033137558 z_k1_s0002
                                 - 0.260000001747197 z_k1_s0003
                                 - 0.424999952317558 z_k1_s0004
                                 - 0.546000897188677 z_k1_s0005
                                 - 0.390998904932667 z_k1_s0006
                                 - 0.291999565310142 z_k1_s0007
                                 - 0.19700012653052 z_k1_s0010
                                 - 0.535000085123352 z_k1_s0011
                                 - 0.222999589929679 z_k1_s0012
                                 - 0.16300009479276 z_k1_s0013
                                 - 0.185000140823089 z_k1_s0014
                                 - 0.16800001660415 z_k1_s0015
                                 - 0.455000902740871 z_k1_s0016
                                 - 0.24100086869153 z_k1_s0017
                                 - 0.178000149434145 z_k1_s0020
                                 - 0.128000061096415 z_k1_s0021
                                 - 0.382000409387977 z_k1_s0022
                                 - 0.0910000280185289 z_k1_s0023
                                 - 0.207000907506771 z_k1_s0024
                                 - 0.435999866128055 z_k1_s0025
                                 - 0.243001143334117 z_k1_s0026
                                 - 0.129000217147332 z_k1_s0027
                                 - 0.34100060601104 z_k1_s0030
                                 - 0.472000812422278 z_k1_s0031
                                 - 1.50300194008858 z_k1_s0032
                                 - 0.698002600892747 z_k1_s0033
                                 - 0.915998155719273 z_k1_s0034
                                 - 0.759003135704769 z_k1_s0035
                                 - 0.701000697552634 z_k1_s0036
                                 - 1.53400073626944 z_k1_s0037
                                 - 0.202000508399282 z_k1_s0040
                                 - 0.58600146826522 z_k1_s0041
                                 - 0.453000856492492 z_k1_s0042
                                 - 0.317001002312426 z_k1_s0043
                                 - 0.255000892441514 z_k1_s0044
                                 - 0.82400221238367 z_k1_s0045
                                 - 0.506000574332501 z_k1_s0046
                                 - 0.188000601786889 z_k1_s0047
                                 - 0.48199934022732 z_k1_s0050
                                 - 0.358999966641154 z_k1_s0051
                                 - 0.512000110563019 z_k1_s0052
                                 - 0.181999521849913 z_k1_s0053
                                 - 0.668999516131821 z_k1_s0054
                                 - 0.332998457952392 z_k1_s0055
                                 - 0.432998923076833 z_k1_s0056
                                 - 0.183999468072565 z_k1_s0057
                                 - 0.687000228033686 z_k1_s0060
                                 - 0.307000903997257 z_k1_s0061
                                 - 0.16899988063718 z_k1_s0062
                                 - 0.26299980405668 z_k1_s0063
                                 - 0.284999780027204 z_k1_s0064
                                 - 0.472998765866559 z_k1_s0065
                                 - 0.488000249709482 z_k1_s0066
                                 - 0.197999783734715 z_k1_s0067
                                 - 0.203000787985069 z_k1_s0070
                                 - 0.201001034113533 z_k1_s0071
                                 - 0.281999655918799 z_k1_s0072
                                 - 0.169000161764447 z_k1_s0073
                                 - 0.199999439176157 z_k1_s0074
                                 - 0.255999419585609 z_k1_s0075
                                 - 0.0830000936784105 z_k1_s0076
                                 - 0.323999641225888 z_k1_s0077
                                 - 0.223999552326221 z_k1_s0100
                                 - 0.120999366680139 z_k1_s0101
                                 - 0.157999103486841 z_k1_s0102
                                 - 0.0819996309812533 z_k1_s0103
                                 - 0.373998660244332 z_k1_s0104
                                 - 0.105999455888327 z_k1_s0105
                                 - 0.346997623505602 z_k1_s0106
                                 - 0.20599940049746 z_k1_s0107
                                 - 0.299000514927927 z_k1_s0110
                                 - 0.263999096739652 z_k1_s0111
                                 - 0.141999908252211 z_k1_s0112
                                 - 0.0779999205639035 z_k1_s0113
                                 - 0.226999319454127 z_k1_s0114
                                 - 0.121000008938816 z_k1_s0115
                                 - 0.155999999727735 z_k1_s0116
                                 - 0.299000514927927 z_k1_s0117
                                 - 0.353999826803827 z_k1_s0120
                                 - 0.337000075301701 z_k1_s0121
                                 - 0.700999519177332 z_k1_s0122
                                 - 0.551000232099533 z_k1_s0123
                                 - 0.326999955129823 z_k1_s0124
                                 - 0.883001001965407 z_k1_s0125
                                 - 0.400000278974451 z_k1_s0126
                                 - 1.47800236654129 z_k1_s0127
                                 - 0.116999904501761 z_k1_s0130
                                 - 0.248000501106252 z_k1_s0131
                                 - 0.134999725348197 z_k1_s0132
                                 - 0.0760000156293442 z_k1_s0133
                                 - 0.194999484502958 z_k1_s0134
                                 - 0.0630000143624826 z_k1_s0135
                                 - 0.100999968727158 z_k1_s0136
                                 - 0.134000210412253 z_k1_s0137
                                 - 0.156999663555119 z_k1_s0140
                                 - 0.369000356688004 z_k1_s0141
                                 - 0.305999062426405 z_k1_s0142
                                 - 0.0919998833697346 z_k1_s0143
                                 - 0.315000150319303 z_k1_s0144
                                 - 0.122999697496755 z_k1_s0145
                                 - 0.386998318481341 z_k1_s0146
                                 - 0.357999729373205 z_k1_s0147
                                 - 0.130999886475833 z_k1_s0150
                                 - 0.215999907132604 z_k1_s0151
                                 - 0.0769999096926821 z_k1_s0152
                                 - 0.117999924396533 z_k1_s0153
                                 - 0.0939999944039784 z_k1_s0154
                                 - 0.378999700265175 z_k1_s0155
                                 - 0.0989999798190939 z_k1_s0156
                                 - 0.229000137811711 z_k1_s0157
                                 - 0.127999683534108 z_k1_s0160
                                 - 0.0839999820691093 z_k1_s0161
                                 - 0.0519999599856618 z_k1_s0162
                                 - 0.0969999214655645 z_k1_s0163
                                 - 0.132999971609423 z_k1_s0164
                                 - 0.12899953874933 z_k1_s0165
                                 - 0.21600009846331 z_k1_s0166
                                 - 0.173000252213534 z_k1_s0167
                                 - 0.336998255474995 z_k1_s0170
                                 - 0.271997909953108 z_k1_s0171
                                 - 0.295999557646891 z_k1_s0172
                                 - 0.387997997840619 z_k1_s0173
                                 - 0.128999512594689 z_k1_s0174
                                 - 0.167999290309688 z_k1_s0175
                                 - 0.230999212125004 z_k1_s0176
                                 - 0.326999358928139 z_k1_s0177
                                 - 0.419000170158799 z_k1_s1000
                                 - 0.156999969400766 z_k1_s1001
                                 - 0.253000816327582 z_k1_s1002
                                 - 0.296001169940628 z_k1_s1003
                                 - 0.639000772574059 z_k1_s1004
                                 - 0.215000038364764 z_k1_s1005
                                 - 0.25499985586746 z_k1_s1006
                                 - 0.23700009860629 z_k1_s1007
                                 - 0.222000211658598 z_k1_s1010
                                 - 0.135999990471844 z_k1_s1011
                                 - 0.253000462351607 z_k1_s1012
                                 - 0.345999470029515 z_k1_s1013
                                 - 0.425000904725114 z_k1_s1014
                                 - 0.35500057980557 z_k1_s1015
                                 - 0.271999482543369 z_k1_s1016
                                 - 0.48299974994325 z_k1_s1017
                                 - 1.39400148390394 z_k1_s1020
                                 - 0.374000787681999 z_k1_s1021
                                 - 1.36300124779106 z_k1_s1022
                                 - 0.657000999355364 z_k1_s1023
                                 - 1.31800371537242 z_k1_s1024
                                 - 0.599000791789606 z_k1_s1025
                                 - 0.789997820286376 z_k1_s1026
                                 - 0.765001611167725 z_k1_s1027
                                 - 0.527998781654719 z_k1_s1030
                                 - 0.853001717569773 z_k1_s1031
                                 - 0.295000023108373 z_k1_s1032
                                 - 0.458999391176967 z_k1_s1033
                                 - 0.664998081936467 z_k1_s1034
                                 - 0.712999572268191 z_k1_s1035
                                 - 0.211999977800606 z_k1_s1036
                                 - 0.678000136692437 z_k1_s1037
                                 - 1.02900121902586 z_k1_s1040
                                 - 0.516000334875287 z_k1_s1041
                                 - 1.57600449716783 z_k1_s1042
                                 - 1.92500462582695 z_k1_s1043
                                 - 2.37600577995222 z_k1_s1044
                                 - 1.19000129249097 z_k1_s1045
                                 - 2.41400374767174 z_k1_s1046
                                 - 1.28700569601914 z_k1_s1047
                                 - 0.61299857854325 z_k1_s1050
                                 - 0.464000035783894 z_k1_s1051
                                 - 0.193999994347593 z_k1_s1052
                                 - 0.481999779499853 z_k1_s1053
                                 - 0.337000515566793 z_k1_s1054
                                 - 0.270999861142875 z_k1_s1055
                                 - 0.303998992755783 z_k1_s1056
                                 - 0.32399837588489 z_k1_s1057
                                 - 0.544999410749408 z_k1_s1060
                                 - 0.848997437847801 z_k1_s1061
                                 - 0.980999519348413 z_k1_s1062
                                 - 1.27099925901319 z_k1_s1063
                                 - 0.262999603903026 z_k1_s1064
                                 - 0.302999847994467 z_k1_s1065
                                 - 0.727997946470071 z_k1_s1066
                                 - 0.860000027970683 z_k1_s1067
                                 - 0.716999679933539 z_k1_s1070
                                 - 2.13900018879716 z_k1_s1071
                                 - 1.00599902772126 z_k1_s1072
                                 - 2.12399470880022 z_k1_s1073
                                 - 0.434999569681269 z_k1_s1074
                                 - 1.27500102823576 z_k1_s1075
                                 - 1.65100117523879 z_k1_s1076
                                 - 1.71900167271756 z_k1_s1077
                                 - 0.120000505184863 z_k1_s1100
                                 - 0.122000436080976 z_k1_s1101
                                 - 0.107000328207693 z_k1_s1102
                                 - 0.49800229330795 z_k1_s1103
                                 - 0.161000609211091 z_k1_s1104
                                 - 0.127000531672302 z_k1_s1105
                                 - 0.169000690596935 z_k1_s1106
                                 - 0.485001579628688 z_k1_s1107
                                 - 0.109000186712954 z_k1_s1110
                                 - 0.392999926999896 z_k1_s1111
                                 - 0.116000144711883 z_k1_s1112
                                 - 0.172000563204265 z_k1_s1113
                                 - 0.192999833600294 z_k1_s1114
                                 - 0.179000823003574 z_k1_s1115
                                 - 0.192999833600294 z_k1_s1116
                                 - 0.153999679577201 z_k1_s1117
                                 - 0.171000197820763 z_k1_s1120
                                 - 0.0889999689686644 z_k1_s1121
                                 - 0.257000044367978 z_k1_s1122
                                 - 0.127000194773328 z_k1_s1123
                                 - 0.298000392594091 z_k1_s1124
                                 - 0.0990000653067439 z_k1_s1125
                                 - 0.0629999990497345 z_k1_s1126
                                 - 0.0650000416973568 z_k1_s1127
                                 - 0.0899999945111802 z_k1_s1130
                                 - 0.232999977290052 z_k1_s1131
                                 - 0.181999954900372 z_k1_s1132
                                 - 0.0890000115721745 z_k1_s1133
                                 - 0.197999936924574 z_k1_s1134
                                 - 0.188999707973355 z_k1_s1135
                                 - 0.188000107534517 z_k1_s1136
                                 - 0.106000104035438 z_k1_s1137
                                 - 0.194000880432261 z_k1_s1140
                                 - 0.208000809958238 z_k1_s1141
                                 - 0.236999735475901 z_k1_s1142
                                 - 0.278000457588124 z_k1_s1143
                                 - 0.264999594527855 z_k1_s1144
                                 - 0.300999413308939 z_k1_s1145
                                 - 0.222999805949924 z_k1_s1146
                                 - 0.15099998268767 z_k1_s1147
                                 - 0.131999837664086 z_k1_s1150
                                 - 0.183000549628095 z_k1_s1151
                                 - 0.199001364226849 z_k1_s1152
                                 - 0.129000520692432 z_k1_s1153
                                 - 0.128000486167571 z_k1_s1154
                                 - 0.17800037700379 z_k1_s1155
                                 - 0.120000472169659 z_k1_s1156
                                 - 0.24200101338904 z_k1_s1157
                                 - 0.116000129005091 z_k1_s1160
                                 - 0.0530000272936799 z_k1_s1161
                                 - 0.0640000809131614 z_k1_s1162
                                 - 0.111999929797863 z_k1_s1163
                                 - 0.0309999906547736 z_k1_s1164
                                 - 0.111000232996339 z_k1_s1165
                                 - 0.109999830194248 z_k1_s1166
                                 - 0.109000133392724 z_k1_s1167
                                 - 0.208999993954563 z_k1_s1170
                                 - 0.113000367018715 z_k1_s1171
                                 - 0.0770000436674978 z_k1_s1172
                                 - 0.155999802520995 z_k1_s1173
                                 - 0.162000256079768 z_k1_s1174
                                 - 0.155999802520995 z_k1_s1175
                                 - 0.100000038308428 z_k1_s1176
                                 - 0.116999579390585 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.703083 w_k1_ADM_s000
                                 + 0.796181 w_k1_AXP_s000
                                 + 0.881395 w_k1_DIS_s000
                                 + 0.782184 w_k1_FDX_s000
                                 + 1.24925 w_k1_IBM_s000
                                 + 0.826565 w_k1_INTC_s000
                                 + 1.08439 w_k1_KO_s000 + 0.76149 w_k1_MAT_s000
                                 + 0.890102 w_k1_MSFT_s000
                                 + 0.859839 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.963264 w_k1_ADM_s000
                                 + 1.1165 w_k1_AXP_s000
                                 + 0.762796 w_k1_DIS_s000
                                 + 0.885505 w_k1_FDX_s000
                                 + 1.08376 w_k1_IBM_s000
                                 + 0.88967 w_k1_INTC_s000
                                 + 0.844301 w_k1_KO_s000
                                 + 1.11517 w_k1_MAT_s000
                                 + 0.931691 w_k1_MSFT_s000
                                 + 0.913959 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.647195 w_k1_ADM_s000
                                 + 0.784486 w_k1_AXP_s000
                                 + 1.35986 w_k1_DIS_s000
                                 + 1.00699 w_k1_FDX_s000
                                 + 0.99166 w_k1_IBM_s000
                                 + 0.883934 w_k1_INTC_s000
                                 + 1.07434 w_k1_KO_s000 + 1.00043 w_k1_MAT_s000
                                 + 1.17224 w_k1_MSFT_s000
                                 + 1.05999 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.07315 w_k1_ADM_s000
                                 + 1.46995 w_k1_AXP_s000
                                 + 1.71732 w_k1_DIS_s000
                                 + 1.27835 w_k1_FDX_s000 + 1.3803 w_k1_IBM_s000
                                 + 1.13629 w_k1_INTC_s000
                                 + 1.23998 w_k1_KO_s000 + 1.2344 w_k1_MAT_s000
                                 + 1.28846 w_k1_MSFT_s000
                                 + 1.25504 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.692 w_k1_ADM_s000
                                 + 0.661116 w_k1_AXP_s000
                                 + 0.733185 w_k1_DIS_s000
                                 + 0.780956 w_k1_FDX_s000
                                 + 0.801435 w_k1_IBM_s000
                                 + 0.612755 w_k1_INTC_s000
                                 + 0.658592 w_k1_KO_s000
                                 + 0.808398 w_k1_MAT_s000
                                 + 0.881419 w_k1_MSFT_s000
                                 + 0.752225 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.11854 w_k1_ADM_s000
                                 + 1.15926 w_k1_AXP_s000
                                 + 1.20015 w_k1_DIS_s000
                                 + 1.06083 w_k1_FDX_s000
                                 + 1.05447 w_k1_IBM_s000
                                 + 0.891364 w_k1_INTC_s000
                                 + 0.957389 w_k1_KO_s000
                                 + 1.03819 w_k1_MAT_s000
                                 + 1.03657 w_k1_MSFT_s000
                                 + 1.08926 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.986508 w_k1_ADM_s000
                                 + 1.20595 w_k1_AXP_s000
                                 + 1.34316 w_k1_DIS_s000
                                 + 1.30266 w_k1_FDX_s000
                                 + 0.809216 w_k1_IBM_s000
                                 + 1.15904 w_k1_INTC_s000
                                 + 0.866173 w_k1_KO_s000
                                 + 0.696235 w_k1_MAT_s000
                                 + 1.26664 w_k1_MSFT_s000
                                 + 1.00051 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.931822 w_k1_ADM_s000
                                 + 0.997295 w_k1_AXP_s000
                                 + 1.23405 w_k1_DIS_s000
                                 + 0.984688 w_k1_FDX_s000
                                 + 1.49749 w_k1_IBM_s000
                                 + 0.866822 w_k1_INTC_s000
                                 + 1.13408 w_k1_KO_s000
                                 + 0.917774 w_k1_MAT_s000
                                 + 0.98508 w_k1_MSFT_s000
                                 + 1.02947 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.912336 w_k1_ADM_s001
                                 + 0.65214 w_k1_AXP_s001
                                 + 0.976825 w_k1_DIS_s001
                                 + 0.980506 w_k1_FDX_s001
                                 + 0.798593 w_k1_IBM_s001
                                 + 0.732103 w_k1_INTC_s001
                                 + 0.867476 w_k1_KO_s001
                                 + 1.03084 w_k1_MAT_s001
                                 + 0.903866 w_k1_MSFT_s001
                                 + 0.968884 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.890858 w_k1_ADM_s001
                                 + 1.06488 w_k1_AXP_s001
                                 + 0.875452 w_k1_DIS_s001
                                 + 1.02887 w_k1_FDX_s001
                                 + 0.980931 w_k1_IBM_s001
                                 + 0.91315 w_k1_INTC_s001
                                 + 0.882642 w_k1_KO_s001
                                 + 1.15989 w_k1_MAT_s001
                                 + 0.896187 w_k1_MSFT_s001
                                 + 1.01367 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.18202 w_k1_ADM_s001
                                 + 0.904919 w_k1_AXP_s001
                                 + 1.21868 w_k1_DIS_s001
                                 + 1.22921 w_k1_FDX_s001
                                 + 0.813667 w_k1_IBM_s001
                                 + 1.0423 w_k1_INTC_s001
                                 + 0.997591 w_k1_KO_s001
                                 + 0.949143 w_k1_MAT_s001
                                 + 1.12811 w_k1_MSFT_s001
                                 + 1.09826 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.0932 w_k1_ADM_s001
                                 + 1.22172 w_k1_AXP_s001
                                 + 1.79166 w_k1_DIS_s001
                                 + 1.37632 w_k1_FDX_s001
                                 + 1.39288 w_k1_IBM_s001
                                 + 1.26622 w_k1_INTC_s001
                                 + 1.40635 w_k1_KO_s001 + 1.28252 w_k1_MAT_s001
                                 + 1.26206 w_k1_MSFT_s001
                                 + 1.24682 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.836105 w_k1_ADM_s001
                                 + 0.844081 w_k1_AXP_s001
                                 + 1.01854 w_k1_DIS_s001
                                 + 0.886201 w_k1_FDX_s001
                                 + 1.48468 w_k1_IBM_s001
                                 + 0.798572 w_k1_INTC_s001
                                 + 1.29931 w_k1_KO_s001 + 1.74402 w_k1_MAT_s001
                                 + 0.823864 w_k1_MSFT_s001
                                 + 1.15548 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.859009 w_k1_ADM_s001
                                 + 0.771012 w_k1_AXP_s001
                                 + 1.22792 w_k1_DIS_s001
                                 + 0.954749 w_k1_FDX_s001
                                 + 1.07463 w_k1_IBM_s001
                                 + 0.909481 w_k1_INTC_s001
                                 + 1.1709 w_k1_KO_s001 + 1.08553 w_k1_MAT_s001
                                 + 1.00712 w_k1_MSFT_s001
                                 + 1.08934 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.933478 w_k1_ADM_s001
                                 + 0.698686 w_k1_AXP_s001
                                 + 0.722182 w_k1_DIS_s001
                                 + 0.755587 w_k1_FDX_s001
                                 + 0.924601 w_k1_IBM_s001
                                 + 0.928037 w_k1_INTC_s001
                                 + 0.842148 w_k1_KO_s001
                                 + 1.04479 w_k1_MAT_s001
                                 + 0.791393 w_k1_MSFT_s001
                                 + 0.924763 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.1187 w_k1_ADM_s001
                                 + 1.06397 w_k1_AXP_s001
                                 + 1.30547 w_k1_DIS_s001
                                 + 1.15378 w_k1_FDX_s001
                                 + 1.27556 w_k1_IBM_s001
                                 + 0.919126 w_k1_INTC_s001
                                 + 1.15215 w_k1_KO_s001 + 1.73587 w_k1_MAT_s001
                                 + 1.0723 w_k1_MSFT_s001 + 1.15379 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.22987 w_k1_ADM_s002
                                 + 1.78318 w_k1_AXP_s002
                                 + 1.53906 w_k1_DIS_s002
                                 + 1.37591 w_k1_FDX_s002 + 1.23 w_k1_IBM_s002
                                 + 1.26124 w_k1_INTC_s002
                                 + 1.49042 w_k1_KO_s002 + 2.35374 w_k1_MAT_s002
                                 + 1.2579 w_k1_MSFT_s002 + 1.42622 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.11042 w_k1_ADM_s002
                                 + 0.886769 w_k1_AXP_s002
                                 + 1.1111 w_k1_DIS_s002 + 1.05461 w_k1_FDX_s002
                                 + 0.816897 w_k1_IBM_s002
                                 + 1.02982 w_k1_INTC_s002
                                 + 0.961049 w_k1_KO_s002
                                 + 0.908148 w_k1_MAT_s002
                                 + 1.20422 w_k1_MSFT_s002
                                 + 1.08729 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07709 w_k1_ADM_s002
                                 + 0.960578 w_k1_AXP_s002
                                 + 0.917808 w_k1_DIS_s002
                                 + 0.856871 w_k1_FDX_s002
                                 + 1.06338 w_k1_IBM_s002
                                 + 0.836248 w_k1_INTC_s002
                                 + 0.983361 w_k1_KO_s002
                                 + 1.14201 w_k1_MAT_s002
                                 + 1.00698 w_k1_MSFT_s002
                                 + 0.971826 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.866615 w_k1_ADM_s002
                                 + 0.831779 w_k1_AXP_s002
                                 + 0.875545 w_k1_DIS_s002
                                 + 0.769443 w_k1_FDX_s002
                                 + 1.45006 w_k1_IBM_s002
                                 + 0.643262 w_k1_INTC_s002
                                 + 1.0649 w_k1_KO_s002 + 1.41965 w_k1_MAT_s002
                                 + 0.842978 w_k1_MSFT_s002
                                 + 0.869812 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.877747 w_k1_ADM_s002
                                 + 1.1419 w_k1_AXP_s002 + 1.32149 w_k1_DIS_s002
                                 + 1.12202 w_k1_FDX_s002
                                 + 1.09145 w_k1_IBM_s002
                                 + 1.05598 w_k1_INTC_s002
                                 + 1.27495 w_k1_KO_s002 + 1.47182 w_k1_MAT_s002
                                 + 1.22213 w_k1_MSFT_s002
                                 + 1.27252 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.852255 w_k1_ADM_s002
                                 + 0.736546 w_k1_AXP_s002
                                 + 0.673014 w_k1_DIS_s002
                                 + 0.843027 w_k1_FDX_s002
                                 + 0.673793 w_k1_IBM_s002
                                 + 0.839979 w_k1_INTC_s002
                                 + 0.880932 w_k1_KO_s002
                                 + 0.884793 w_k1_MAT_s002
                                 + 0.930488 w_k1_MSFT_s002
                                 + 0.927481 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.00946 w_k1_ADM_s002
                                 + 1.19699 w_k1_AXP_s002
                                 + 0.935884 w_k1_DIS_s002
                                 + 1.14106 w_k1_FDX_s002
                                 + 0.850229 w_k1_IBM_s002
                                 + 0.864832 w_k1_INTC_s002
                                 + 0.907292 w_k1_KO_s002
                                 + 1.31879 w_k1_MAT_s002
                                 + 0.907507 w_k1_MSFT_s002
                                 + 1.04358 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.07964 w_k1_ADM_s002
                                 + 1.40447 w_k1_AXP_s002
                                 + 1.49131 w_k1_DIS_s002 + 1.1138 w_k1_FDX_s002
                                 + 1.60251 w_k1_IBM_s002
                                 + 1.26193 w_k1_INTC_s002
                                 + 1.40038 w_k1_KO_s002 + 1.85022 w_k1_MAT_s002
                                 + 1.20722 w_k1_MSFT_s002
                                 + 1.05791 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.684634 w_k1_ADM_s003
                                 + 0.653512 w_k1_AXP_s003
                                 + 0.714335 w_k1_DIS_s003
                                 + 0.680502 w_k1_FDX_s003
                                 + 1.19841 w_k1_IBM_s003
                                 + 0.79738 w_k1_INTC_s003
                                 + 0.914805 w_k1_KO_s003
                                 + 0.819352 w_k1_MAT_s003
                                 + 0.762039 w_k1_MSFT_s003
                                 + 0.800928 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.24663 w_k1_ADM_s003
                                 + 1.58437 w_k1_AXP_s003
                                 + 1.51459 w_k1_DIS_s003
                                 + 1.33456 w_k1_FDX_s003 + 1.6955 w_k1_IBM_s003
                                 + 1.23312 w_k1_INTC_s003
                                 + 1.40797 w_k1_KO_s003 + 1.77432 w_k1_MAT_s003
                                 + 1.34845 w_k1_MSFT_s003
                                 + 1.26654 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.12954 w_k1_ADM_s003
                                 + 1.09285 w_k1_AXP_s003
                                 + 0.964187 w_k1_DIS_s003
                                 + 1.11053 w_k1_FDX_s003
                                 + 0.797585 w_k1_IBM_s003
                                 + 1.0037 w_k1_INTC_s003
                                 + 0.934771 w_k1_KO_s003
                                 + 0.651467 w_k1_MAT_s003
                                 + 0.982864 w_k1_MSFT_s003
                                 + 0.974008 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.00339 w_k1_ADM_s003
                                 + 1.09963 w_k1_AXP_s003
                                 + 1.42558 w_k1_DIS_s003
                                 + 1.13028 w_k1_FDX_s003
                                 + 1.13595 w_k1_IBM_s003
                                 + 1.0218 w_k1_INTC_s003 + 1.17415 w_k1_KO_s003
                                 + 0.793288 w_k1_MAT_s003
                                 + 1.12798 w_k1_MSFT_s003 + 1.0836 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.27777 w_k1_ADM_s003
                                 + 1.56802 w_k1_AXP_s003
                                 + 1.38295 w_k1_DIS_s003
                                 + 1.36683 w_k1_FDX_s003
                                 + 0.973553 w_k1_IBM_s003
                                 + 1.38444 w_k1_INTC_s003
                                 + 1.22764 w_k1_KO_s003 + 1.17808 w_k1_MAT_s003
                                 + 1.32114 w_k1_MSFT_s003
                                 + 1.06644 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.796888 w_k1_ADM_s003
                                 + 0.922311 w_k1_AXP_s003
                                 + 0.970929 w_k1_DIS_s003
                                 + 1.05249 w_k1_FDX_s003
                                 + 0.964696 w_k1_IBM_s003
                                 + 1.03181 w_k1_INTC_s003
                                 + 0.942741 w_k1_KO_s003
                                 + 0.66652 w_k1_MAT_s003
                                 + 0.935113 w_k1_MSFT_s003
                                 + 0.956476 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.902367 w_k1_ADM_s003
                                 + 1.07584 w_k1_AXP_s003
                                 + 0.999806 w_k1_DIS_s003
                                 + 0.945362 w_k1_FDX_s003
                                 + 1.51068 w_k1_IBM_s003
                                 + 0.969875 w_k1_INTC_s003
                                 + 1.0195 w_k1_KO_s003 + 1.18854 w_k1_MAT_s003
                                 + 0.851683 w_k1_MSFT_s003
                                 + 0.980885 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.906317 w_k1_ADM_s003
                                 + 0.799001 w_k1_AXP_s003
                                 + 0.716251 w_k1_DIS_s003
                                 + 0.809981 w_k1_FDX_s003
                                 + 0.818818 w_k1_IBM_s003
                                 + 0.900064 w_k1_INTC_s003
                                 + 0.912979 w_k1_KO_s003
                                 + 0.589883 w_k1_MAT_s003
                                 + 0.894287 w_k1_MSFT_s003
                                 + 0.81267 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.87285 w_k1_ADM_s004
                                 + 1.01716 w_k1_AXP_s004 + 1.2493 w_k1_DIS_s004
                                 + 1.09269 w_k1_FDX_s004
                                 + 1.58836 w_k1_IBM_s004
                                 + 1.31719 w_k1_INTC_s004
                                 + 1.45773 w_k1_KO_s004 + 1.21249 w_k1_MAT_s004
                                 + 0.976415 w_k1_MSFT_s004
                                 + 1.17632 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.25473 w_k1_ADM_s004
                                 + 1.42961 w_k1_AXP_s004
                                 + 1.34894 w_k1_DIS_s004
                                 + 1.35249 w_k1_FDX_s004
                                 + 1.31359 w_k1_IBM_s004
                                 + 1.12504 w_k1_INTC_s004
                                 + 1.56302 w_k1_KO_s004 + 1.45284 w_k1_MAT_s004
                                 + 1.28606 w_k1_MSFT_s004
                                 + 1.36422 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.761058 w_k1_ADM_s004
                                 + 0.786905 w_k1_AXP_s004
                                 + 0.662573 w_k1_DIS_s004
                                 + 0.79922 w_k1_FDX_s004 + 1.0289 w_k1_IBM_s004
                                 + 0.853131 w_k1_INTC_s004
                                 + 0.830476 w_k1_KO_s004
                                 + 0.803834 w_k1_MAT_s004
                                 + 0.79159 w_k1_MSFT_s004
                                 + 0.98398 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10547 w_k1_ADM_s004
                                 + 0.957554 w_k1_AXP_s004
                                 + 0.753075 w_k1_DIS_s004
                                 + 0.948356 w_k1_FDX_s004
                                 + 0.942753 w_k1_IBM_s004
                                 + 0.87392 w_k1_INTC_s004
                                 + 0.796591 w_k1_KO_s004
                                 + 0.849709 w_k1_MAT_s004
                                 + 0.777011 w_k1_MSFT_s004
                                 + 0.946423 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.00854 w_k1_ADM_s004
                                 + 0.96784 w_k1_AXP_s004
                                 + 0.827314 w_k1_DIS_s004
                                 + 1.04026 w_k1_FDX_s004
                                 + 0.702207 w_k1_IBM_s004
                                 + 1.05113 w_k1_INTC_s004
                                 + 0.790043 w_k1_KO_s004
                                 + 0.701393 w_k1_MAT_s004
                                 + 0.91905 w_k1_MSFT_s004
                                 + 0.926772 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.08173 w_k1_ADM_s004
                                 + 0.940678 w_k1_AXP_s004
                                 + 1.08915 w_k1_DIS_s004
                                 + 1.11017 w_k1_FDX_s004
                                 + 0.939606 w_k1_IBM_s004
                                 + 1.07879 w_k1_INTC_s004
                                 + 1.09645 w_k1_KO_s004
                                 + 0.787878 w_k1_MAT_s004
                                 + 1.20905 w_k1_MSFT_s004
                                 + 1.08579 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.867831 w_k1_ADM_s004
                                 + 0.641844 w_k1_AXP_s004
                                 + 0.821497 w_k1_DIS_s004
                                 + 0.858594 w_k1_FDX_s004
                                 + 0.858812 w_k1_IBM_s004
                                 + 0.826468 w_k1_INTC_s004
                                 + 0.86754 w_k1_KO_s004
                                 + 0.616369 w_k1_MAT_s004
                                 + 0.762579 w_k1_MSFT_s004
                                 + 0.865481 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.883996 w_k1_ADM_s004
                                 + 1.2915 w_k1_AXP_s004
                                 + 0.989656 w_k1_DIS_s004
                                 + 1.2642 w_k1_FDX_s004
                                 + 0.719135 w_k1_IBM_s004
                                 + 1.17389 w_k1_INTC_s004
                                 + 0.822627 w_k1_KO_s004
                                 + 0.73901 w_k1_MAT_s004
                                 + 1.01595 w_k1_MSFT_s004
                                 + 0.942386 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.37317 w_k1_ADM_s005
                                 + 0.964705 w_k1_AXP_s005
                                 + 1.19696 w_k1_DIS_s005
                                 + 1.12544 w_k1_FDX_s005
                                 + 0.838267 w_k1_IBM_s005
                                 + 0.9692 w_k1_INTC_s005
                                 + 0.962344 w_k1_KO_s005
                                 + 0.816924 w_k1_MAT_s005
                                 + 1.04342 w_k1_MSFT_s005
                                 + 1.03687 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.841731 w_k1_ADM_s005
                                 + 0.800006 w_k1_AXP_s005
                                 + 0.706751 w_k1_DIS_s005
                                 + 0.804805 w_k1_FDX_s005
                                 + 0.839338 w_k1_IBM_s005
                                 + 0.892181 w_k1_INTC_s005
                                 + 0.781515 w_k1_KO_s005
                                 + 0.843957 w_k1_MAT_s005
                                 + 0.86813 w_k1_MSFT_s005
                                 + 0.84655 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.1201 w_k1_ADM_s005
                                 + 1.00499 w_k1_AXP_s005
                                 + 0.794386 w_k1_DIS_s005
                                 + 0.916774 w_k1_FDX_s005
                                 + 1.08993 w_k1_IBM_s005
                                 + 0.978919 w_k1_INTC_s005
                                 + 0.934797 w_k1_KO_s005
                                 + 1.24149 w_k1_MAT_s005
                                 + 0.869966 w_k1_MSFT_s005
                                 + 0.976518 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.07243 w_k1_ADM_s005
                                 + 0.725885 w_k1_AXP_s005
                                 + 0.60545 w_k1_DIS_s005
                                 + 0.782221 w_k1_FDX_s005
                                 + 0.617357 w_k1_IBM_s005
                                 + 0.633332 w_k1_INTC_s005
                                 + 0.653171 w_k1_KO_s005
                                 + 0.727065 w_k1_MAT_s005
                                 + 0.796071 w_k1_MSFT_s005
                                 + 0.859452 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.29781 w_k1_ADM_s005
                                 + 1.60436 w_k1_AXP_s005
                                 + 1.45051 w_k1_DIS_s005
                                 + 1.36655 w_k1_FDX_s005
                                 + 1.19292 w_k1_IBM_s005
                                 + 1.31577 w_k1_INTC_s005 + 1.2561 w_k1_KO_s005
                                 + 1.29563 w_k1_MAT_s005
                                 + 1.1522 w_k1_MSFT_s005 + 1.0757 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.14502 w_k1_ADM_s005
                                 + 1.32138 w_k1_AXP_s005
                                 + 0.937631 w_k1_DIS_s005
                                 + 1.05268 w_k1_FDX_s005
                                 + 0.727551 w_k1_IBM_s005
                                 + 1.05955 w_k1_INTC_s005
                                 + 0.835133 w_k1_KO_s005
                                 + 0.796722 w_k1_MAT_s005
                                 + 1.00785 w_k1_MSFT_s005
                                 + 0.989301 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.02602 w_k1_ADM_s005
                                 + 1.27286 w_k1_AXP_s005
                                 + 1.48026 w_k1_DIS_s005
                                 + 0.981384 w_k1_FDX_s005
                                 + 1.71362 w_k1_IBM_s005
                                 + 0.982655 w_k1_INTC_s005
                                 + 1.39643 w_k1_KO_s005 + 1.83565 w_k1_MAT_s005
                                 + 1.10247 w_k1_MSFT_s005
                                 + 1.33362 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.01181 w_k1_ADM_s005
                                 + 0.730033 w_k1_AXP_s005
                                 + 0.883821 w_k1_DIS_s005
                                 + 0.915171 w_k1_FDX_s005
                                 + 0.684443 w_k1_IBM_s005
                                 + 0.856146 w_k1_INTC_s005
                                 + 0.740697 w_k1_KO_s005
                                 + 0.609814 w_k1_MAT_s005
                                 + 1.03819 w_k1_MSFT_s005
                                 + 0.919953 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.28854 w_k1_ADM_s006
                                 + 0.857822 w_k1_AXP_s006
                                 + 0.997044 w_k1_DIS_s006
                                 + 0.902705 w_k1_FDX_s006
                                 + 1.12568 w_k1_IBM_s006
                                 + 1.15918 w_k1_INTC_s006
                                 + 1.11727 w_k1_KO_s006
                                 + 0.923043 w_k1_MAT_s006
                                 + 0.94741 w_k1_MSFT_s006
                                 + 0.950989 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.54769 w_k1_ADM_s006
                                 + 1.2814 w_k1_AXP_s006 + 1.70578 w_k1_DIS_s006
                                 + 1.22016 w_k1_FDX_s006
                                 + 1.81059 w_k1_IBM_s006
                                 + 1.45462 w_k1_INTC_s006
                                 + 1.61584 w_k1_KO_s006 + 1.40793 w_k1_MAT_s006
                                 + 1.42863 w_k1_MSFT_s006
                                 + 1.50182 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.996754 w_k1_ADM_s006
                                 + 0.940415 w_k1_AXP_s006
                                 + 0.965774 w_k1_DIS_s006
                                 + 0.899023 w_k1_FDX_s006
                                 + 0.897386 w_k1_IBM_s006
                                 + 0.93398 w_k1_INTC_s006
                                 + 0.916643 w_k1_KO_s006
                                 + 0.831147 w_k1_MAT_s006
                                 + 0.915462 w_k1_MSFT_s006
                                 + 1.01188 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.64436 w_k1_ADM_s006
                                 + 1.41839 w_k1_AXP_s006
                                 + 1.08992 w_k1_DIS_s006
                                 + 1.14922 w_k1_FDX_s006
                                 + 1.00941 w_k1_IBM_s006
                                 + 1.05201 w_k1_INTC_s006
                                 + 0.970688 w_k1_KO_s006
                                 + 1.40475 w_k1_MAT_s006
                                 + 1.25668 w_k1_MSFT_s006
                                 + 1.11399 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.42046 w_k1_ADM_s006
                                 + 0.83802 w_k1_AXP_s006
                                 + 1.20646 w_k1_DIS_s006
                                 + 1.18879 w_k1_FDX_s006
                                 + 0.744547 w_k1_IBM_s006
                                 + 1.38663 w_k1_INTC_s006
                                 + 0.899615 w_k1_KO_s006
                                 + 0.653903 w_k1_MAT_s006
                                 + 1.27049 w_k1_MSFT_s006
                                 + 1.04138 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.1644 w_k1_ADM_s006
                                 + 0.777227 w_k1_AXP_s006
                                 + 0.60074 w_k1_DIS_s006
                                 + 0.794037 w_k1_FDX_s006
                                 + 0.686137 w_k1_IBM_s006
                                 + 0.769601 w_k1_INTC_s006
                                 + 0.707215 w_k1_KO_s006
                                 + 0.787739 w_k1_MAT_s006
                                 + 0.740314 w_k1_MSFT_s006
                                 + 0.766514 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.40977 w_k1_ADM_s006
                                 + 1.02009 w_k1_AXP_s006
                                 + 1.30857 w_k1_DIS_s006 + 1.1446 w_k1_FDX_s006
                                 + 1.29741 w_k1_IBM_s006
                                 + 1.0831 w_k1_INTC_s006 + 1.21183 w_k1_KO_s006
                                 + 1.42653 w_k1_MAT_s006
                                 + 1.22772 w_k1_MSFT_s006
                                 + 1.14856 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.34534 w_k1_ADM_s006
                                 + 1.14112 w_k1_AXP_s006 + 1.0349 w_k1_DIS_s006
                                 + 1.02661 w_k1_FDX_s006
                                 + 0.722615 w_k1_IBM_s006
                                 + 0.986832 w_k1_INTC_s006
                                 + 0.988848 w_k1_KO_s006
                                 + 0.893378 w_k1_MAT_s006
                                 + 1.08757 w_k1_MSFT_s006
                                 + 1.06026 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.992617 w_k1_ADM_s007
                                 + 0.766792 w_k1_AXP_s007
                                 + 0.858198 w_k1_DIS_s007
                                 + 0.783901 w_k1_FDX_s007
                                 + 1.08024 w_k1_IBM_s007
                                 + 0.927571 w_k1_INTC_s007
                                 + 0.97549 w_k1_KO_s007
                                 + 0.936447 w_k1_MAT_s007
                                 + 0.920473 w_k1_MSFT_s007
                                 + 1.07172 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.39017 w_k1_ADM_s007
                                 + 1.3089 w_k1_AXP_s007
                                 + 0.935556 w_k1_DIS_s007
                                 + 1.11687 w_k1_FDX_s007
                                 + 0.871276 w_k1_IBM_s007
                                 + 0.958467 w_k1_INTC_s007
                                 + 1.00697 w_k1_KO_s007 + 1.03439 w_k1_MAT_s007
                                 + 0.881541 w_k1_MSFT_s007
                                 + 1.10127 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.978641 w_k1_ADM_s007
                                 + 1.18269 w_k1_AXP_s007
                                 + 0.940967 w_k1_DIS_s007
                                 + 0.940715 w_k1_FDX_s007
                                 + 1.11526 w_k1_IBM_s007
                                 + 0.914674 w_k1_INTC_s007
                                 + 0.866044 w_k1_KO_s007
                                 + 0.796251 w_k1_MAT_s007
                                 + 0.882496 w_k1_MSFT_s007
                                 + 1.0045 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.975544 w_k1_ADM_s007
                                 + 1.16404 w_k1_AXP_s007 + 1.4007 w_k1_DIS_s007
                                 + 1.02843 w_k1_FDX_s007
                                 + 1.85506 w_k1_IBM_s007
                                 + 1.0754 w_k1_INTC_s007 + 1.51588 w_k1_KO_s007
                                 + 2.0819 w_k1_MAT_s007
                                 + 0.930571 w_k1_MSFT_s007
                                 + 1.27546 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.720914 w_k1_ADM_s007
                                 + 0.637227 w_k1_AXP_s007
                                 + 0.637482 w_k1_DIS_s007
                                 + 0.741683 w_k1_FDX_s007
                                 + 0.655804 w_k1_IBM_s007
                                 + 0.906211 w_k1_INTC_s007
                                 + 0.829715 w_k1_KO_s007
                                 + 0.728525 w_k1_MAT_s007
                                 + 0.745302 w_k1_MSFT_s007
                                 + 0.906652 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.4036 w_k1_ADM_s007
                                 + 1.15896 w_k1_AXP_s007 + 1.419 w_k1_DIS_s007
                                 + 1.07013 w_k1_FDX_s007
                                 + 1.15059 w_k1_IBM_s007
                                 + 1.1407 w_k1_INTC_s007 + 1.25797 w_k1_KO_s007
                                 + 1.17673 w_k1_MAT_s007
                                 + 1.06049 w_k1_MSFT_s007
                                 + 1.12325 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.55068 w_k1_ADM_s007
                                 + 1.87761 w_k1_AXP_s007
                                 + 2.05878 w_k1_DIS_s007
                                 + 1.51598 w_k1_FDX_s007
                                 + 1.51607 w_k1_IBM_s007
                                 + 1.50661 w_k1_INTC_s007
                                 + 1.52253 w_k1_KO_s007 + 1.51311 w_k1_MAT_s007
                                 + 1.32346 w_k1_MSFT_s007 + 1.346 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.985938 w_k1_ADM_s007
                                 + 0.89432 w_k1_AXP_s007
                                 + 0.864304 w_k1_DIS_s007
                                 + 0.88389 w_k1_FDX_s007
                                 + 0.697364 w_k1_IBM_s007
                                 + 0.952715 w_k1_INTC_s007
                                 + 0.792905 w_k1_KO_s007
                                 + 0.565728 w_k1_MAT_s007
                                 + 0.874581 w_k1_MSFT_s007
                                 + 0.907601 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.7534 w_k1_ADM_s010
                                 + 0.986228 w_k1_AXP_s010 + 1.073 w_k1_DIS_s010
                                 + 0.868596 w_k1_FDX_s010
                                 + 1.41256 w_k1_IBM_s010
                                 + 0.929669 w_k1_INTC_s010
                                 + 1.06531 w_k1_KO_s010
                                 + 0.977781 w_k1_MAT_s010
                                 + 0.777956 w_k1_MSFT_s010
                                 + 1.02604 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.981543 w_k1_ADM_s010
                                 + 0.736843 w_k1_AXP_s010
                                 + 1.08771 w_k1_DIS_s010
                                 + 0.895824 w_k1_FDX_s010
                                 + 0.72975 w_k1_IBM_s010
                                 + 1.05086 w_k1_INTC_s010
                                 + 0.827757 w_k1_KO_s010
                                 + 0.544377 w_k1_MAT_s010
                                 + 1.0014 w_k1_MSFT_s010
                                 + 0.867988 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.28085 w_k1_ADM_s010
                                 + 1.24344 w_k1_AXP_s010
                                 + 1.74626 w_k1_DIS_s010
                                 + 1.34024 w_k1_FDX_s010
                                 + 0.909037 w_k1_IBM_s010
                                 + 1.12576 w_k1_INTC_s010
                                 + 1.12443 w_k1_KO_s010
                                 + 0.961726 w_k1_MAT_s010
                                 + 1.30145 w_k1_MSFT_s010
                                 + 1.35553 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.32535 w_k1_ADM_s010
                                 + 1.7007 w_k1_AXP_s010 + 1.93084 w_k1_DIS_s010
                                 + 1.32597 w_k1_FDX_s010
                                 + 1.45407 w_k1_IBM_s010
                                 + 1.13493 w_k1_INTC_s010 + 1.4402 w_k1_KO_s010
                                 + 1.31816 w_k1_MAT_s010
                                 + 1.14997 w_k1_MSFT_s010
                                 + 1.30643 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.876125 w_k1_ADM_s010
                                 + 1.20513 w_k1_AXP_s010
                                 + 1.21604 w_k1_DIS_s010
                                 + 1.05065 w_k1_FDX_s010
                                 + 1.01825 w_k1_IBM_s010
                                 + 1.18227 w_k1_INTC_s010
                                 + 1.10467 w_k1_KO_s010
                                 + 0.953931 w_k1_MAT_s010
                                 + 1.00695 w_k1_MSFT_s010
                                 + 1.12124 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.91583 w_k1_ADM_s010
                                 + 0.960556 w_k1_AXP_s010
                                 + 1.57129 w_k1_DIS_s010
                                 + 1.04565 w_k1_FDX_s010
                                 + 1.28402 w_k1_IBM_s010
                                 + 1.07863 w_k1_INTC_s010
                                 + 1.32125 w_k1_KO_s010 + 1.14223 w_k1_MAT_s010
                                 + 1.17129 w_k1_MSFT_s010
                                 + 1.24711 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.775268 w_k1_ADM_s010
                                 + 0.746674 w_k1_AXP_s010
                                 + 0.820992 w_k1_DIS_s010
                                 + 0.754025 w_k1_FDX_s010
                                 + 0.9288 w_k1_IBM_s010
                                 + 0.908736 w_k1_INTC_s010
                                 + 0.878104 w_k1_KO_s010
                                 + 0.561802 w_k1_MAT_s010
                                 + 0.737976 w_k1_MSFT_s010
                                 + 0.940402 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.867339 w_k1_ADM_s010
                                 + 0.903776 w_k1_AXP_s010
                                 + 0.892763 w_k1_DIS_s010
                                 + 1.03223 w_k1_FDX_s010
                                 + 0.761699 w_k1_IBM_s010
                                 + 1.07062 w_k1_INTC_s010
                                 + 0.752532 w_k1_KO_s010
                                 + 0.66171 w_k1_MAT_s010
                                 + 0.951049 w_k1_MSFT_s010
                                 + 0.846665 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.995171 w_k1_ADM_s011
                                 + 0.798592 w_k1_AXP_s011
                                 + 0.613147 w_k1_DIS_s011
                                 + 0.853615 w_k1_FDX_s011
                                 + 0.813044 w_k1_IBM_s011
                                 + 1.09077 w_k1_INTC_s011
                                 + 0.702185 w_k1_KO_s011
                                 + 0.892276 w_k1_MAT_s011
                                 + 0.776435 w_k1_MSFT_s011
                                 + 0.862288 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.02432 w_k1_ADM_s011
                                 + 0.807007 w_k1_AXP_s011
                                 + 1.06367 w_k1_DIS_s011
                                 + 0.977798 w_k1_FDX_s011
                                 + 0.999498 w_k1_IBM_s011
                                 + 1.2816 w_k1_INTC_s011 + 1.13778 w_k1_KO_s011
                                 + 0.809993 w_k1_MAT_s011
                                 + 0.999688 w_k1_MSFT_s011
                                 + 1.0689 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.09233 w_k1_ADM_s011
                                 + 1.38239 w_k1_AXP_s011
                                 + 0.994044 w_k1_DIS_s011
                                 + 1.11681 w_k1_FDX_s011
                                 + 0.97518 w_k1_IBM_s011
                                 + 1.10939 w_k1_INTC_s011 + 1.0004 w_k1_KO_s011
                                 + 1.34572 w_k1_MAT_s011
                                 + 0.899276 w_k1_MSFT_s011
                                 + 0.996859 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.04741 w_k1_ADM_s011
                                 + 0.973286 w_k1_AXP_s011
                                 + 0.849273 w_k1_DIS_s011
                                 + 1.14616 w_k1_FDX_s011
                                 + 0.754603 w_k1_IBM_s011
                                 + 1.35215 w_k1_INTC_s011
                                 + 0.948971 w_k1_KO_s011
                                 + 0.805432 w_k1_MAT_s011
                                 + 0.998072 w_k1_MSFT_s011
                                 + 0.994024 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.21493 w_k1_ADM_s011
                                 + 1.2106 w_k1_AXP_s011 + 1.32469 w_k1_DIS_s011
                                 + 1.22578 w_k1_FDX_s011
                                 + 1.27912 w_k1_IBM_s011
                                 + 1.37986 w_k1_INTC_s011
                                 + 1.26854 w_k1_KO_s011 + 1.21607 w_k1_MAT_s011
                                 + 1.2193 w_k1_MSFT_s011 + 1.05187 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.29273 w_k1_ADM_s011
                                 + 1.18469 w_k1_AXP_s011
                                 + 1.33452 w_k1_DIS_s011
                                 + 1.28453 w_k1_FDX_s011
                                 + 0.970041 w_k1_IBM_s011
                                 + 1.23754 w_k1_INTC_s011 + 1.2451 w_k1_KO_s011
                                 + 0.962603 w_k1_MAT_s011
                                 + 1.14537 w_k1_MSFT_s011
                                 + 1.20358 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06517 w_k1_ADM_s011
                                 + 1.28798 w_k1_AXP_s011
                                 + 1.22982 w_k1_DIS_s011
                                 + 1.00972 w_k1_FDX_s011
                                 + 1.58243 w_k1_IBM_s011
                                 + 1.15909 w_k1_INTC_s011
                                 + 1.43014 w_k1_KO_s011 + 1.62148 w_k1_MAT_s011
                                 + 1.06483 w_k1_MSFT_s011
                                 + 1.22243 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.00714 w_k1_ADM_s011
                                 + 0.691797 w_k1_AXP_s011
                                 + 0.752483 w_k1_DIS_s011
                                 + 0.865105 w_k1_FDX_s011
                                 + 0.920759 w_k1_IBM_s011
                                 + 1.046 w_k1_INTC_s011 + 0.928917 w_k1_KO_s011
                                 + 0.706087 w_k1_MAT_s011
                                 + 0.876533 w_k1_MSFT_s011
                                 + 0.948716 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.23769 w_k1_ADM_s012
                                 + 1.35064 w_k1_AXP_s012
                                 + 1.79215 w_k1_DIS_s012
                                 + 1.16387 w_k1_FDX_s012
                                 + 1.53296 w_k1_IBM_s012
                                 + 1.48273 w_k1_INTC_s012
                                 + 1.57292 w_k1_KO_s012 + 1.83364 w_k1_MAT_s012
                                 + 1.17289 w_k1_MSFT_s012
                                 + 1.17633 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.765861 w_k1_ADM_s012
                                 + 1.01217 w_k1_AXP_s012
                                 + 0.97194 w_k1_DIS_s012
                                 + 0.876125 w_k1_FDX_s012
                                 + 1.73314 w_k1_IBM_s012
                                 + 1.20722 w_k1_INTC_s012
                                 + 1.26117 w_k1_KO_s012 + 2.23863 w_k1_MAT_s012
                                 + 0.97194 w_k1_MSFT_s012
                                 + 0.98539 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11085 w_k1_ADM_s012
                                 + 1.12776 w_k1_AXP_s012
                                 + 1.08804 w_k1_DIS_s012
                                 + 1.18185 w_k1_FDX_s012
                                 + 0.718701 w_k1_IBM_s012
                                 + 1.13378 w_k1_INTC_s012
                                 + 0.839724 w_k1_KO_s012
                                 + 0.909498 w_k1_MAT_s012
                                 + 0.901188 w_k1_MSFT_s012
                                 + 1.01639 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.12055 w_k1_ADM_s012
                                 + 1.33422 w_k1_AXP_s012 + 1.3607 w_k1_DIS_s012
                                 + 1.24101 w_k1_FDX_s012
                                 + 1.04619 w_k1_IBM_s012
                                 + 1.58707 w_k1_INTC_s012
                                 + 1.15068 w_k1_KO_s012
                                 + 0.956067 w_k1_MAT_s012
                                 + 1.11425 w_k1_MSFT_s012
                                 + 1.02244 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.14428 w_k1_ADM_s012
                                 + 0.919185 w_k1_AXP_s012
                                 + 1.13008 w_k1_DIS_s012
                                 + 1.06989 w_k1_FDX_s012
                                 + 0.993998 w_k1_IBM_s012
                                 + 1.15482 w_k1_INTC_s012
                                 + 1.05515 w_k1_KO_s012
                                 + 0.963768 w_k1_MAT_s012
                                 + 0.992472 w_k1_MSFT_s012
                                 + 1.02092 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.781731 w_k1_ADM_s012
                                 + 1.11661 w_k1_AXP_s012
                                 + 0.941427 w_k1_DIS_s012
                                 + 0.983289 w_k1_FDX_s012
                                 + 1.22395 w_k1_IBM_s012
                                 + 1.17509 w_k1_INTC_s012
                                 + 1.05627 w_k1_KO_s012 + 1.35592 w_k1_MAT_s012
                                 + 0.9236 w_k1_MSFT_s012
                                 + 0.946193 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.26332 w_k1_ADM_s012
                                 + 1.24813 w_k1_AXP_s012
                                 + 1.31046 w_k1_DIS_s012
                                 + 1.17908 w_k1_FDX_s012
                                 + 1.31247 w_k1_IBM_s012
                                 + 1.05642 w_k1_INTC_s012
                                 + 1.18134 w_k1_KO_s012 + 1.44797 w_k1_MAT_s012
                                 + 1.1553 w_k1_MSFT_s012 + 1.19151 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.888914 w_k1_ADM_s012
                                 + 0.757813 w_k1_AXP_s012
                                 + 0.848412 w_k1_DIS_s012
                                 + 0.877614 w_k1_FDX_s012
                                 + 0.765318 w_k1_IBM_s012
                                 + 1.01056 w_k1_INTC_s012
                                 + 0.875331 w_k1_KO_s012
                                 + 0.695384 w_k1_MAT_s012
                                 + 0.862196 w_k1_MSFT_s012
                                 + 0.847594 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.876862 w_k1_ADM_s013
                                 + 0.722751 w_k1_AXP_s013
                                 + 0.770227 w_k1_DIS_s013
                                 + 0.817606 w_k1_FDX_s013
                                 + 1.15873 w_k1_IBM_s013
                                 + 0.727353 w_k1_INTC_s013
                                 + 0.915972 w_k1_KO_s013
                                 + 1.03973 w_k1_MAT_s013
                                 + 0.903544 w_k1_MSFT_s013
                                 + 0.952375 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.899479 w_k1_ADM_s013
                                 + 0.982377 w_k1_AXP_s013
                                 + 1.01171 w_k1_DIS_s013
                                 + 1.00185 w_k1_FDX_s013
                                 + 0.950202 w_k1_IBM_s013
                                 + 0.937595 w_k1_INTC_s013
                                 + 0.838456 w_k1_KO_s013
                                 + 0.857167 w_k1_MAT_s013
                                 + 1.13715 w_k1_MSFT_s013
                                 + 0.949683 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.16252 w_k1_ADM_s013
                                 + 1.09659 w_k1_AXP_s013
                                 + 1.04724 w_k1_DIS_s013
                                 + 1.15627 w_k1_FDX_s013
                                 + 0.819128 w_k1_IBM_s013
                                 + 1.13604 w_k1_INTC_s013
                                 + 0.806002 w_k1_KO_s013
                                 + 0.806877 w_k1_MAT_s013
                                 + 1.08482 w_k1_MSFT_s013
                                 + 0.892286 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.923871 w_k1_ADM_s013
                                 + 1.49099 w_k1_AXP_s013
                                 + 1.30941 w_k1_DIS_s013
                                 + 1.31797 w_k1_FDX_s013
                                 + 1.45138 w_k1_IBM_s013
                                 + 1.23406 w_k1_INTC_s013
                                 + 1.11655 w_k1_KO_s013 + 1.84602 w_k1_MAT_s013
                                 + 1.37563 w_k1_MSFT_s013
                                 + 1.16429 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.767737 w_k1_ADM_s013
                                 + 0.775348 w_k1_AXP_s013
                                 + 0.663923 w_k1_DIS_s013
                                 + 0.767965 w_k1_FDX_s013
                                 + 0.777946 w_k1_IBM_s013
                                 + 0.903752 w_k1_INTC_s013
                                 + 0.685514 w_k1_KO_s013
                                 + 0.637801 w_k1_MAT_s013
                                 + 0.840359 w_k1_MSFT_s013
                                 + 0.78353 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.883231 w_k1_ADM_s013
                                 + 1.22264 w_k1_AXP_s013
                                 + 1.54665 w_k1_DIS_s013
                                 + 1.20642 w_k1_FDX_s013
                                 + 2.03329 w_k1_IBM_s013
                                 + 1.18094 w_k1_INTC_s013
                                 + 1.32194 w_k1_KO_s013 + 1.86807 w_k1_MAT_s013
                                 + 1.21882 w_k1_MSFT_s013
                                 + 1.13006 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.13567 w_k1_ADM_s013
                                 + 1.5963 w_k1_AXP_s013 + 1.26351 w_k1_DIS_s013
                                 + 1.2591 w_k1_FDX_s013 + 1.07786 w_k1_IBM_s013
                                 + 0.982926 w_k1_INTC_s013
                                 + 0.949162 w_k1_KO_s013
                                 + 1.73807 w_k1_MAT_s013
                                 + 1.20785 w_k1_MSFT_s013
                                 + 1.06488 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.00108 w_k1_ADM_s013
                                 + 1.10146 w_k1_AXP_s013
                                 + 1.19265 w_k1_DIS_s013
                                 + 1.09325 w_k1_FDX_s013
                                 + 1.29136 w_k1_IBM_s013
                                 + 1.03902 w_k1_INTC_s013
                                 + 1.16097 w_k1_KO_s013 + 1.445 w_k1_MAT_s013
                                 + 1.05464 w_k1_MSFT_s013
                                 + 1.09694 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.08968 w_k1_ADM_s014
                                 + 1.0294 w_k1_AXP_s014 + 1.58741 w_k1_DIS_s014
                                 + 1.31302 w_k1_FDX_s014
                                 + 1.05653 w_k1_IBM_s014
                                 + 1.29927 w_k1_INTC_s014
                                 + 1.05213 w_k1_KO_s014
                                 + 0.683537 w_k1_MAT_s014
                                 + 1.32961 w_k1_MSFT_s014
                                 + 1.02915 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.856288 w_k1_ADM_s014
                                 + 0.775991 w_k1_AXP_s014
                                 + 0.742011 w_k1_DIS_s014
                                 + 0.851052 w_k1_FDX_s014
                                 + 0.73243 w_k1_IBM_s014
                                 + 0.812641 w_k1_INTC_s014
                                 + 0.806039 w_k1_KO_s014
                                 + 0.72334 w_k1_MAT_s014
                                 + 0.852471 w_k1_MSFT_s014
                                 + 0.878245 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.36671 w_k1_ADM_s014
                                 + 1.19327 w_k1_AXP_s014
                                 + 1.14074 w_k1_DIS_s014
                                 + 1.07742 w_k1_FDX_s014 + 1.4146 w_k1_IBM_s014
                                 + 1.06199 w_k1_INTC_s014
                                 + 1.10697 w_k1_KO_s014 + 1.10216 w_k1_MAT_s014
                                 + 1.08519 w_k1_MSFT_s014
                                 + 1.06443 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.68516 w_k1_ADM_s014
                                 + 1.82139 w_k1_AXP_s014
                                 + 1.77686 w_k1_DIS_s014
                                 + 1.46418 w_k1_FDX_s014
                                 + 1.45004 w_k1_IBM_s014
                                 + 1.42671 w_k1_INTC_s014
                                 + 1.31475 w_k1_KO_s014 + 1.42997 w_k1_MAT_s014
                                 + 1.38603 w_k1_MSFT_s014
                                 + 1.14515 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.916282 w_k1_ADM_s014
                                 + 0.896834 w_k1_AXP_s014
                                 + 0.987312 w_k1_DIS_s014
                                 + 1.01703 w_k1_FDX_s014
                                 + 1.16002 w_k1_IBM_s014
                                 + 1.16659 w_k1_INTC_s014
                                 + 1.01208 w_k1_KO_s014
                                 + 0.896083 w_k1_MAT_s014
                                 + 1.1179 w_k1_MSFT_s014
                                 + 0.999196 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.26487 w_k1_ADM_s014
                                 + 1.43314 w_k1_AXP_s014
                                 + 1.03231 w_k1_DIS_s014
                                 + 1.14547 w_k1_FDX_s014
                                 + 0.914126 w_k1_IBM_s014
                                 + 1.17055 w_k1_INTC_s014
                                 + 1.05521 w_k1_KO_s014 + 1.30524 w_k1_MAT_s014
                                 + 1.18203 w_k1_MSFT_s014
                                 + 1.03716 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.93325 w_k1_ADM_s014
                                 + 0.931735 w_k1_AXP_s014
                                 + 0.726854 w_k1_DIS_s014
                                 + 0.857398 w_k1_FDX_s014
                                 + 1.08941 w_k1_IBM_s014
                                 + 0.910447 w_k1_INTC_s014
                                 + 0.930476 w_k1_KO_s014
                                 + 0.951397 w_k1_MAT_s014
                                 + 0.840967 w_k1_MSFT_s014
                                 + 1.07343 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.10375 w_k1_ADM_s014
                                 + 1.10086 w_k1_AXP_s014
                                 + 0.974853 w_k1_DIS_s014
                                 + 1.12695 w_k1_FDX_s014
                                 + 0.720294 w_k1_IBM_s014
                                 + 1.0887 w_k1_INTC_s014
                                 + 0.783936 w_k1_KO_s014
                                 + 0.702892 w_k1_MAT_s014
                                 + 1.17835 w_k1_MSFT_s014
                                 + 0.941175 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.14687 w_k1_ADM_s015
                                 + 0.932624 w_k1_AXP_s015
                                 + 1.08344 w_k1_DIS_s015
                                 + 1.13996 w_k1_FDX_s015
                                 + 0.974305 w_k1_IBM_s015
                                 + 1.02424 w_k1_INTC_s015
                                 + 1.14386 w_k1_KO_s015
                                 + 0.885338 w_k1_MAT_s015
                                 + 1.35512 w_k1_MSFT_s015
                                 + 1.05815 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.951776 w_k1_ADM_s015
                                 + 0.977222 w_k1_AXP_s015
                                 + 1.1173 w_k1_DIS_s015 + 1.02038 w_k1_FDX_s015
                                 + 1.2242 w_k1_IBM_s015
                                 + 1.04437 w_k1_INTC_s015
                                 + 1.18425 w_k1_KO_s015 + 1.05649 w_k1_MAT_s015
                                 + 1.18796 w_k1_MSFT_s015
                                 + 1.03135 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.16917 w_k1_ADM_s015
                                 + 1.40568 w_k1_AXP_s015
                                 + 1.30552 w_k1_DIS_s015
                                 + 1.40938 w_k1_FDX_s015
                                 + 0.963839 w_k1_IBM_s015
                                 + 1.35538 w_k1_INTC_s015
                                 + 1.18418 w_k1_KO_s015 + 1.16909 w_k1_MAT_s015
                                 + 1.3946 w_k1_MSFT_s015 + 1.09941 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.992994 w_k1_ADM_s015
                                 + 0.902292 w_k1_AXP_s015
                                 + 0.985545 w_k1_DIS_s015
                                 + 0.930605 w_k1_FDX_s015
                                 + 1.45547 w_k1_IBM_s015
                                 + 0.812677 w_k1_INTC_s015
                                 + 1.22052 w_k1_KO_s015 + 1.71747 w_k1_MAT_s015
                                 + 1.15134 w_k1_MSFT_s015
                                 + 1.14833 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.2156 w_k1_ADM_s015
                                 + 1.33958 w_k1_AXP_s015
                                 + 1.61308 w_k1_DIS_s015 + 1.4074 w_k1_FDX_s015
                                 + 1.39385 w_k1_IBM_s015 + 1.214 w_k1_INTC_s015
                                 + 1.53021 w_k1_KO_s015 + 1.79791 w_k1_MAT_s015
                                 + 1.57483 w_k1_MSFT_s015
                                 + 1.41665 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.948205 w_k1_ADM_s015
                                 + 0.799192 w_k1_AXP_s015
                                 + 0.908005 w_k1_DIS_s015
                                 + 0.927018 w_k1_FDX_s015
                                 + 1.00539 w_k1_IBM_s015
                                 + 0.847285 w_k1_INTC_s015
                                 + 0.992979 w_k1_KO_s015
                                 + 1.25818 w_k1_MAT_s015
                                 + 1.05047 w_k1_MSFT_s015
                                 + 0.936076 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.07043 w_k1_ADM_s015
                                 + 1.0008 w_k1_AXP_s015
                                 + 0.845054 w_k1_DIS_s015
                                 + 1.07204 w_k1_FDX_s015
                                 + 0.725023 w_k1_IBM_s015
                                 + 0.899479 w_k1_INTC_s015
                                 + 0.790658 w_k1_KO_s015
                                 + 0.866913 w_k1_MAT_s015
                                 + 1.10926 w_k1_MSFT_s015
                                 + 0.997778 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.80553 w_k1_ADM_s015
                                 + 0.667389 w_k1_AXP_s015
                                 + 0.657366 w_k1_DIS_s015
                                 + 0.764698 w_k1_FDX_s015
                                 + 0.802107 w_k1_IBM_s015
                                 + 0.909671 w_k1_INTC_s015
                                 + 0.839568 w_k1_KO_s015
                                 + 0.96703 w_k1_MAT_s015
                                 + 1.01947 w_k1_MSFT_s015
                                 + 0.868615 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.1515 w_k1_ADM_s016
                                 + 0.98404 w_k1_AXP_s016
                                 + 0.806715 w_k1_DIS_s016
                                 + 0.902505 w_k1_FDX_s016
                                 + 0.877352 w_k1_IBM_s016
                                 + 0.647781 w_k1_INTC_s016
                                 + 0.813703 w_k1_KO_s016
                                 + 0.981229 w_k1_MAT_s016
                                 + 0.663363 w_k1_MSFT_s016
                                 + 0.730894 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.654002 w_k1_ADM_s016
                                 + 0.82856 w_k1_AXP_s016
                                 + 0.854909 w_k1_DIS_s016
                                 + 0.794286 w_k1_FDX_s016
                                 + 1.51561 w_k1_IBM_s016
                                 + 0.819354 w_k1_INTC_s016
                                 + 0.977781 w_k1_KO_s016
                                 + 1.07659 w_k1_MAT_s016
                                 + 0.774434 w_k1_MSFT_s016
                                 + 0.844781 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.924938 w_k1_ADM_s016
                                 + 0.822073 w_k1_AXP_s016
                                 + 1.01109 w_k1_DIS_s016
                                 + 0.868095 w_k1_FDX_s016
                                 + 1.19607 w_k1_IBM_s016
                                 + 0.802955 w_k1_INTC_s016
                                 + 1.09678 w_k1_KO_s016
                                 + 0.755109 w_k1_MAT_s016
                                 + 0.960156 w_k1_MSFT_s016
                                 + 1.02113 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.820753 w_k1_ADM_s016
                                 + 0.647248 w_k1_AXP_s016
                                 + 0.687754 w_k1_DIS_s016
                                 + 0.708143 w_k1_FDX_s016
                                 + 0.830419 w_k1_IBM_s016
                                 + 0.722772 w_k1_INTC_s016
                                 + 0.889054 w_k1_KO_s016
                                 + 0.811273 w_k1_MAT_s016
                                 + 0.711276 w_k1_MSFT_s016
                                 + 0.780532 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.02599 w_k1_ADM_s016
                                 + 1.20309 w_k1_AXP_s016
                                 + 0.976265 w_k1_DIS_s016
                                 + 1.20232 w_k1_FDX_s016
                                 + 0.740204 w_k1_IBM_s016
                                 + 0.918507 w_k1_INTC_s016
                                 + 0.881064 w_k1_KO_s016
                                 + 0.985281 w_k1_MAT_s016
                                 + 0.904561 w_k1_MSFT_s016
                                 + 0.87461 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.14575 w_k1_ADM_s016
                                 + 0.774672 w_k1_AXP_s016
                                 + 1.05793 w_k1_DIS_s016
                                 + 0.899494 w_k1_FDX_s016
                                 + 0.715617 w_k1_IBM_s016
                                 + 0.839053 w_k1_INTC_s016
                                 + 0.955896 w_k1_KO_s016
                                 + 0.811204 w_k1_MAT_s016
                                 + 0.926755 w_k1_MSFT_s016
                                 + 0.794293 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.18436 w_k1_ADM_s016
                                 + 1.34324 w_k1_AXP_s016
                                 + 1.20118 w_k1_DIS_s016
                                 + 1.13079 w_k1_FDX_s016
                                 + 1.07213 w_k1_IBM_s016
                                 + 1.13558 w_k1_INTC_s016
                                 + 1.19806 w_k1_KO_s016 + 1.54999 w_k1_MAT_s016
                                 + 1.09698 w_k1_MSFT_s016
                                 + 0.957589 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.19245 w_k1_ADM_s016
                                 + 1.24054 w_k1_AXP_s016
                                 + 1.31905 w_k1_DIS_s016
                                 + 1.16937 w_k1_FDX_s016
                                 + 1.54123 w_k1_IBM_s016
                                 + 1.18829 w_k1_INTC_s016
                                 + 1.43255 w_k1_KO_s016 + 1.39059 w_k1_MAT_s016
                                 + 1.16785 w_k1_MSFT_s016
                                 + 0.972398 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.978886 w_k1_ADM_s017
                                 + 0.998253 w_k1_AXP_s017
                                 + 0.966878 w_k1_DIS_s017
                                 + 0.902488 w_k1_FDX_s017
                                 + 1.48331 w_k1_IBM_s017
                                 + 0.865503 w_k1_INTC_s017
                                 + 1.09775 w_k1_KO_s017 + 1.51161 w_k1_MAT_s017
                                 + 1.04917 w_k1_MSFT_s017
                                 + 1.01857 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.11367 w_k1_ADM_s017
                                 + 1.45439 w_k1_AXP_s017
                                 + 1.57882 w_k1_DIS_s017
                                 + 1.32474 w_k1_FDX_s017
                                 + 1.44889 w_k1_IBM_s017
                                 + 1.66911 w_k1_INTC_s017
                                 + 1.52824 w_k1_KO_s017 + 1.27381 w_k1_MAT_s017
                                 + 1.38901 w_k1_MSFT_s017
                                 + 1.32796 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.761317 w_k1_ADM_s017
                                 + 0.688006 w_k1_AXP_s017
                                 + 0.545749 w_k1_DIS_s017
                                 + 0.803181 w_k1_FDX_s017
                                 + 0.807786 w_k1_IBM_s017
                                 + 0.960246 w_k1_INTC_s017
                                 + 0.775244 w_k1_KO_s017
                                 + 0.906051 w_k1_MAT_s017
                                 + 0.83816 w_k1_MSFT_s017
                                 + 0.877869 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.884948 w_k1_ADM_s017
                                 + 0.885869 w_k1_AXP_s017
                                 + 0.757267 w_k1_DIS_s017
                                 + 0.87435 w_k1_FDX_s017
                                 + 0.890242 w_k1_IBM_s017
                                 + 1.17007 w_k1_INTC_s017
                                 + 0.876494 w_k1_KO_s017
                                 + 0.894694 w_k1_MAT_s017
                                 + 1.00593 w_k1_MSFT_s017
                                 + 0.977751 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.868655 w_k1_ADM_s017
                                 + 0.696422 w_k1_AXP_s017
                                 + 0.657026 w_k1_DIS_s017
                                 + 0.877897 w_k1_FDX_s017
                                 + 0.500732 w_k1_IBM_s017
                                 + 1.09484 w_k1_INTC_s017
                                 + 0.64594 w_k1_KO_s017
                                 + 0.397984 w_k1_MAT_s017
                                 + 1.03615 w_k1_MSFT_s017
                                 + 0.888605 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.856316 w_k1_ADM_s017
                                 + 0.802549 w_k1_AXP_s017
                                 + 0.976629 w_k1_DIS_s017
                                 + 0.915827 w_k1_FDX_s017
                                 + 1.11259 w_k1_IBM_s017
                                 + 1.13498 w_k1_INTC_s017
                                 + 1.10762 w_k1_KO_s017
                                 + 0.910582 w_k1_MAT_s017
                                 + 1.1003 w_k1_MSFT_s017 + 1.04618 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.902805 w_k1_ADM_s017
                                 + 0.970732 w_k1_AXP_s017
                                 + 1.15853 w_k1_DIS_s017 + 1.1027 w_k1_FDX_s017
                                 + 0.97816 w_k1_IBM_s017
                                 + 1.50928 w_k1_INTC_s017
                                 + 1.13327 w_k1_KO_s017
                                 + 0.887401 w_k1_MAT_s017
                                 + 1.3169 w_k1_MSFT_s017 + 1.09846 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.07332 w_k1_ADM_s017
                                 + 1.24327 w_k1_AXP_s017
                                 + 1.24754 w_k1_DIS_s017
                                 + 1.30216 w_k1_FDX_s017
                                 + 0.717295 w_k1_IBM_s017
                                 + 1.41489 w_k1_INTC_s017
                                 + 0.960567 w_k1_KO_s017
                                 + 0.811926 w_k1_MAT_s017
                                 + 1.21075 w_k1_MSFT_s017
                                 + 1.18416 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.711851 w_k1_ADM_s100
                                 + 0.816171 w_k1_AXP_s100
                                 + 0.77813 w_k1_DIS_s100
                                 + 0.89185 w_k1_FDX_s100
                                 + 0.648829 w_k1_IBM_s100
                                 + 0.864934 w_k1_INTC_s100
                                 + 0.858266 w_k1_KO_s100
                                 + 0.722908 w_k1_MAT_s100
                                 + 0.986361 w_k1_MSFT_s100
                                 + 0.882093 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.860015 w_k1_ADM_s100
                                 + 0.886754 w_k1_AXP_s100
                                 + 0.763474 w_k1_DIS_s100
                                 + 0.800196 w_k1_FDX_s100
                                 + 1.45848 w_k1_IBM_s100
                                 + 0.712158 w_k1_INTC_s100
                                 + 0.947639 w_k1_KO_s100
                                 + 1.76802 w_k1_MAT_s100
                                 + 0.813129 w_k1_MSFT_s100
                                 + 1.01281 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.939869 w_k1_ADM_s100
                                 + 1.27104 w_k1_AXP_s100
                                 + 1.27011 w_k1_DIS_s100
                                 + 1.29313 w_k1_FDX_s100
                                 + 0.890245 w_k1_IBM_s100
                                 + 1.33319 w_k1_INTC_s100
                                 + 1.06206 w_k1_KO_s100 + 1.07288 w_k1_MAT_s100
                                 + 1.38087 w_k1_MSFT_s100
                                 + 1.08008 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.702754 w_k1_ADM_s100
                                 + 0.846031 w_k1_AXP_s100
                                 + 0.629392 w_k1_DIS_s100
                                 + 0.739432 w_k1_FDX_s100
                                 + 0.91868 w_k1_IBM_s100
                                 + 0.847504 w_k1_INTC_s100
                                 + 0.775855 w_k1_KO_s100
                                 + 0.92948 w_k1_MAT_s100
                                 + 0.807128 w_k1_MSFT_s100
                                 + 0.81722 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.879435 w_k1_ADM_s100
                                 + 0.886486 w_k1_AXP_s100
                                 + 1.0003 w_k1_DIS_s100
                                 + 0.985734 w_k1_FDX_s100
                                 + 0.904272 w_k1_IBM_s100
                                 + 0.836834 w_k1_INTC_s100
                                 + 0.993844 w_k1_KO_s100
                                 + 1.02041 w_k1_MAT_s100
                                 + 1.10862 w_k1_MSFT_s100
                                 + 0.966873 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.16516 w_k1_ADM_s100
                                 + 1.2614 w_k1_AXP_s100 + 1.22322 w_k1_DIS_s100
                                 + 1.09568 w_k1_FDX_s100
                                 + 1.31915 w_k1_IBM_s100
                                 + 1.08191 w_k1_INTC_s100
                                 + 1.34727 w_k1_KO_s100 + 1.21807 w_k1_MAT_s100
                                 + 1.10396 w_k1_MSFT_s100
                                 + 1.05317 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.816004 w_k1_ADM_s100
                                 + 0.918638 w_k1_AXP_s100
                                 + 0.879619 w_k1_DIS_s100
                                 + 0.889577 w_k1_FDX_s100
                                 + 1.12483 w_k1_IBM_s100
                                 + 0.827962 w_k1_INTC_s100
                                 + 1.01976 w_k1_KO_s100 + 1.27424 w_k1_MAT_s100
                                 + 0.864258 w_k1_MSFT_s100
                                 + 0.913979 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.03555 w_k1_ADM_s100
                                 + 1.48957 w_k1_AXP_s100
                                 + 1.53823 w_k1_DIS_s100
                                 + 1.12961 w_k1_FDX_s100
                                 + 1.54352 w_k1_IBM_s100
                                 + 0.929275 w_k1_INTC_s100
                                 + 1.43304 w_k1_KO_s100 + 2.29665 w_k1_MAT_s100
                                 + 1.13518 w_k1_MSFT_s100
                                 + 1.30978 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.18409 w_k1_ADM_s101
                                 + 1.33158 w_k1_AXP_s101
                                 + 1.45709 w_k1_DIS_s101
                                 + 1.30309 w_k1_FDX_s101
                                 + 0.945929 w_k1_IBM_s101
                                 + 1.43329 w_k1_INTC_s101 + 1.2364 w_k1_KO_s101
                                 + 1.45759 w_k1_MAT_s101
                                 + 1.31709 w_k1_MSFT_s101
                                 + 1.06226 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.38464 w_k1_ADM_s101
                                 + 1.46174 w_k1_AXP_s101
                                 + 2.08085 w_k1_DIS_s101
                                 + 1.39809 w_k1_FDX_s101
                                 + 1.42685 w_k1_IBM_s101
                                 + 1.26242 w_k1_INTC_s101
                                 + 1.64385 w_k1_KO_s101 + 1.40532 w_k1_MAT_s101
                                 + 1.60388 w_k1_MSFT_s101
                                 + 1.21683 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.07455 w_k1_ADM_s101
                                 + 0.887447 w_k1_AXP_s101
                                 + 1.01353 w_k1_DIS_s101
                                 + 1.10807 w_k1_FDX_s101
                                 + 0.799003 w_k1_IBM_s101
                                 + 0.942517 w_k1_INTC_s101
                                 + 0.884964 w_k1_KO_s101
                                 + 1.04327 w_k1_MAT_s101
                                 + 1.05401 w_k1_MSFT_s101
                                 + 0.909305 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09692 w_k1_ADM_s101
                                 + 1.05699 w_k1_AXP_s101
                                 + 1.20055 w_k1_DIS_s101
                                 + 0.990736 w_k1_FDX_s101
                                 + 1.49514 w_k1_IBM_s101
                                 + 0.926705 w_k1_INTC_s101
                                 + 1.27222 w_k1_KO_s101 + 1.59634 w_k1_MAT_s101
                                 + 0.984017 w_k1_MSFT_s101
                                 + 1.02466 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.05877 w_k1_ADM_s101
                                 + 0.829971 w_k1_AXP_s101
                                 + 0.953345 w_k1_DIS_s101
                                 + 0.881091 w_k1_FDX_s101
                                 + 1.10851 w_k1_IBM_s101
                                 + 0.818495 w_k1_INTC_s101
                                 + 1.06475 w_k1_KO_s101 + 1.16556 w_k1_MAT_s101
                                 + 0.896268 w_k1_MSFT_s101
                                 + 0.955755 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.956124 w_k1_ADM_s101
                                 + 0.975107 w_k1_AXP_s101
                                 + 1.38108 w_k1_DIS_s101
                                 + 1.10543 w_k1_FDX_s101
                                 + 1.13027 w_k1_IBM_s101
                                 + 1.15044 w_k1_INTC_s101
                                 + 1.17853 w_k1_KO_s101 + 1.33316 w_k1_MAT_s101
                                 + 1.14143 w_k1_MSFT_s101
                                 + 1.13035 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04885 w_k1_ADM_s101
                                 + 0.88071 w_k1_AXP_s101
                                 + 0.949668 w_k1_DIS_s101
                                 + 0.997641 w_k1_FDX_s101
                                 + 0.945947 w_k1_IBM_s101
                                 + 0.918085 w_k1_INTC_s101
                                 + 0.90971 w_k1_KO_s101 + 1.24278 w_k1_MAT_s101
                                 + 1.01112 w_k1_MSFT_s101
                                 + 0.905773 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.982787 w_k1_ADM_s101
                                 + 0.727338 w_k1_AXP_s101
                                 + 0.655994 w_k1_DIS_s101
                                 + 0.779465 w_k1_FDX_s101
                                 + 0.791359 w_k1_IBM_s101
                                 + 0.807015 w_k1_INTC_s101
                                 + 0.77285 w_k1_KO_s101
                                 + 0.971038 w_k1_MAT_s101
                                 + 0.894853 w_k1_MSFT_s101
                                 + 0.895906 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.924993 w_k1_ADM_s102
                                 + 1.05924 w_k1_AXP_s102
                                 + 0.683137 w_k1_DIS_s102
                                 + 0.890301 w_k1_FDX_s102
                                 + 0.975585 w_k1_IBM_s102
                                 + 0.817995 w_k1_INTC_s102
                                 + 0.831993 w_k1_KO_s102
                                 + 0.807458 w_k1_MAT_s102
                                 + 0.917898 w_k1_MSFT_s102
                                 + 0.923451 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.31179 w_k1_ADM_s102
                                 + 0.98692 w_k1_AXP_s102
                                 + 1.61652 w_k1_DIS_s102
                                 + 1.23317 w_k1_FDX_s102
                                 + 1.53513 w_k1_IBM_s102
                                 + 1.14783 w_k1_INTC_s102
                                 + 1.58893 w_k1_KO_s102 + 1.13348 w_k1_MAT_s102
                                 + 1.41174 w_k1_MSFT_s102
                                 + 1.28572 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.893574 w_k1_ADM_s102
                                 + 0.950579 w_k1_AXP_s102
                                 + 1.12713 w_k1_DIS_s102
                                 + 1.00554 w_k1_FDX_s102
                                 + 0.967882 w_k1_IBM_s102
                                 + 0.983539 w_k1_INTC_s102
                                 + 0.953393 w_k1_KO_s102
                                 + 0.92042 w_k1_MAT_s102
                                 + 1.02877 w_k1_MSFT_s102
                                 + 1.09993 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.00512 w_k1_ADM_s102
                                 + 1.27403 w_k1_AXP_s102
                                 + 1.12658 w_k1_DIS_s102
                                 + 1.07144 w_k1_FDX_s102
                                 + 1.29326 w_k1_IBM_s102
                                 + 1.04442 w_k1_INTC_s102
                                 + 1.20576 w_k1_KO_s102 + 1.0805 w_k1_MAT_s102
                                 + 1.18887 w_k1_MSFT_s102
                                 + 1.17661 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.909938 w_k1_ADM_s102
                                 + 0.791577 w_k1_AXP_s102
                                 + 0.718553 w_k1_DIS_s102
                                 + 0.849622 w_k1_FDX_s102
                                 + 0.746811 w_k1_IBM_s102
                                 + 0.849336 w_k1_INTC_s102
                                 + 0.757886 w_k1_KO_s102
                                 + 0.576201 w_k1_MAT_s102
                                 + 0.963895 w_k1_MSFT_s102
                                 + 0.936934 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.054 w_k1_ADM_s102
                                 + 1.63336 w_k1_AXP_s102
                                 + 1.36785 w_k1_DIS_s102
                                 + 1.32287 w_k1_FDX_s102
                                 + 1.32406 w_k1_IBM_s102
                                 + 1.2237 w_k1_INTC_s102 + 1.21208 w_k1_KO_s102
                                 + 1.6532 w_k1_MAT_s102
                                 + 1.50451 w_k1_MSFT_s102 + 1.2929 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.867354 w_k1_ADM_s102
                                 + 0.999388 w_k1_AXP_s102
                                 + 1.0045 w_k1_DIS_s102
                                 + 0.859873 w_k1_FDX_s102
                                 + 1.29996 w_k1_IBM_s102
                                 + 1.02938 w_k1_INTC_s102
                                 + 1.18825 w_k1_KO_s102 + 1.29804 w_k1_MAT_s102
                                 + 1.09017 w_k1_MSFT_s102
                                 + 1.12435 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.28943 w_k1_ADM_s102
                                 + 1.27407 w_k1_AXP_s102
                                 + 1.17914 w_k1_DIS_s102
                                 + 1.22861 w_k1_FDX_s102
                                 + 0.880854 w_k1_IBM_s102
                                 + 1.07631 w_k1_INTC_s102
                                 + 0.931196 w_k1_KO_s102
                                 + 1.08557 w_k1_MAT_s102
                                 + 1.24274 w_k1_MSFT_s102
                                 + 1.07649 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.691158 w_k1_ADM_s103
                                 + 0.625973 w_k1_AXP_s103
                                 + 0.618328 w_k1_DIS_s103
                                 + 0.836767 w_k1_FDX_s103
                                 + 0.518567 w_k1_IBM_s103
                                 + 0.748185 w_k1_INTC_s103
                                 + 0.664053 w_k1_KO_s103
                                 + 0.522995 w_k1_MAT_s103
                                 + 0.77835 w_k1_MSFT_s103
                                 + 0.815245 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.08889 w_k1_ADM_s103
                                 + 0.904715 w_k1_AXP_s103
                                 + 1.25397 w_k1_DIS_s103 + 1.0635 w_k1_FDX_s103
                                 + 0.955313 w_k1_IBM_s103
                                 + 0.975311 w_k1_INTC_s103
                                 + 1.0377 w_k1_KO_s103 + 0.710361 w_k1_MAT_s103
                                 + 1.0442 w_k1_MSFT_s103 + 1.11038 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.03002 w_k1_ADM_s103
                                 + 0.875287 w_k1_AXP_s103
                                 + 0.73598 w_k1_DIS_s103
                                 + 0.826408 w_k1_FDX_s103
                                 + 0.98166 w_k1_IBM_s103
                                 + 0.86043 w_k1_INTC_s103
                                 + 0.860168 w_k1_KO_s103
                                 + 0.876847 w_k1_MAT_s103
                                 + 0.76928 w_k1_MSFT_s103
                                 + 0.895379 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.05896 w_k1_ADM_s103
                                 + 1.00467 w_k1_AXP_s103
                                 + 0.86404 w_k1_DIS_s103
                                 + 1.15781 w_k1_FDX_s103
                                 + 0.824058 w_k1_IBM_s103
                                 + 1.1451 w_k1_INTC_s103
                                 + 0.778616 w_k1_KO_s103
                                 + 0.910868 w_k1_MAT_s103
                                 + 1.00051 w_k1_MSFT_s103
                                 + 0.956043 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.05174 w_k1_ADM_s103
                                 + 1.11816 w_k1_AXP_s103
                                 + 1.14942 w_k1_DIS_s103
                                 + 1.02718 w_k1_FDX_s103 + 1.5448 w_k1_IBM_s103
                                 + 0.831597 w_k1_INTC_s103
                                 + 1.11883 w_k1_KO_s103 + 1.61915 w_k1_MAT_s103
                                 + 0.914551 w_k1_MSFT_s103
                                 + 0.995562 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.794192 w_k1_ADM_s103
                                 + 0.74632 w_k1_AXP_s103
                                 + 0.859583 w_k1_DIS_s103
                                 + 0.941543 w_k1_FDX_s103
                                 + 0.717794 w_k1_IBM_s103
                                 + 0.924689 w_k1_INTC_s103
                                 + 0.745182 w_k1_KO_s103
                                 + 0.758313 w_k1_MAT_s103
                                 + 0.885651 w_k1_MSFT_s103
                                 + 0.96859 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.722006 w_k1_ADM_s103
                                 + 0.515423 w_k1_AXP_s103
                                 + 0.649786 w_k1_DIS_s103
                                 + 0.712708 w_k1_FDX_s103
                                 + 0.803116 w_k1_IBM_s103
                                 + 0.759082 w_k1_INTC_s103
                                 + 0.731279 w_k1_KO_s103
                                 + 0.592668 w_k1_MAT_s103
                                 + 0.805241 w_k1_MSFT_s103
                                 + 0.749544 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.10276 w_k1_ADM_s103
                                 + 1.4475 w_k1_AXP_s103 + 1.70121 w_k1_DIS_s103
                                 + 1.4281 w_k1_FDX_s103 + 1.34185 w_k1_IBM_s103
                                 + 1.19962 w_k1_INTC_s103 + 1.3398 w_k1_KO_s103
                                 + 1.57381 w_k1_MAT_s103
                                 + 1.17418 w_k1_MSFT_s103
                                 + 1.31273 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.15463 w_k1_ADM_s104
                                 + 1.16215 w_k1_AXP_s104
                                 + 1.27156 w_k1_DIS_s104 + 1.2353 w_k1_FDX_s104
                                 + 0.74234 w_k1_IBM_s104
                                 + 1.1162 w_k1_INTC_s104
                                 + 0.848494 w_k1_KO_s104
                                 + 0.916249 w_k1_MAT_s104
                                 + 1.1139 w_k1_MSFT_s104
                                 + 0.973551 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.36356 w_k1_ADM_s104
                                 + 1.67127 w_k1_AXP_s104
                                 + 2.08681 w_k1_DIS_s104
                                 + 1.40351 w_k1_FDX_s104
                                 + 1.68932 w_k1_IBM_s104
                                 + 1.25237 w_k1_INTC_s104
                                 + 1.65229 w_k1_KO_s104 + 2.08223 w_k1_MAT_s104
                                 + 1.32737 w_k1_MSFT_s104
                                 + 1.17467 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.08445 w_k1_ADM_s104
                                 + 1.24261 w_k1_AXP_s104
                                 + 1.07317 w_k1_DIS_s104
                                 + 1.11691 w_k1_FDX_s104
                                 + 1.01093 w_k1_IBM_s104
                                 + 1.2845 w_k1_INTC_s104 + 1.07216 w_k1_KO_s104
                                 + 1.19295 w_k1_MAT_s104
                                 + 1.13685 w_k1_MSFT_s104
                                 + 1.07928 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.13817 w_k1_ADM_s104
                                 + 1.19863 w_k1_AXP_s104
                                 + 1.45774 w_k1_DIS_s104
                                 + 1.14534 w_k1_FDX_s104
                                 + 1.52879 w_k1_IBM_s104
                                 + 1.12138 w_k1_INTC_s104 + 1.3491 w_k1_KO_s104
                                 + 1.58595 w_k1_MAT_s104
                                 + 1.11596 w_k1_MSFT_s104
                                 + 1.02245 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.893092 w_k1_ADM_s104
                                 + 0.873897 w_k1_AXP_s104
                                 + 0.879164 w_k1_DIS_s104
                                 + 0.910121 w_k1_FDX_s104
                                 + 0.753804 w_k1_IBM_s104
                                 + 1.07952 w_k1_INTC_s104
                                 + 0.811071 w_k1_KO_s104
                                 + 0.783688 w_k1_MAT_s104
                                 + 0.965508 w_k1_MSFT_s104
                                 + 0.938422 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.760087 w_k1_ADM_s104
                                 + 0.878391 w_k1_AXP_s104
                                 + 0.750777 w_k1_DIS_s104
                                 + 0.756566 w_k1_FDX_s104
                                 + 1.05296 w_k1_IBM_s104
                                 + 0.835146 w_k1_INTC_s104
                                 + 0.828756 w_k1_KO_s104
                                 + 1.09516 w_k1_MAT_s104
                                 + 0.773869 w_k1_MSFT_s104
                                 + 0.881605 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.846283 w_k1_ADM_s104
                                 + 0.899233 w_k1_AXP_s104
                                 + 1.11836 w_k1_DIS_s104
                                 + 0.951998 w_k1_FDX_s104
                                 + 1.19035 w_k1_IBM_s104
                                 + 1.07907 w_k1_INTC_s104
                                 + 1.11608 w_k1_KO_s104 + 1.00765 w_k1_MAT_s104
                                 + 0.971786 w_k1_MSFT_s104
                                 + 1.03434 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.737389 w_k1_ADM_s104
                                 + 0.717327 w_k1_AXP_s104
                                 + 0.597422 w_k1_DIS_s104
                                 + 0.733741 w_k1_FDX_s104
                                 + 0.632209 w_k1_IBM_s104
                                 + 0.91765 w_k1_INTC_s104
                                 + 0.614093 w_k1_KO_s104
                                 + 0.46343 w_k1_MAT_s104
                                 + 0.700747 w_k1_MSFT_s104
                                 + 0.747528 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.769052 w_k1_ADM_s105
                                 + 0.916383 w_k1_AXP_s105
                                 + 0.695503 w_k1_DIS_s105
                                 + 0.816135 w_k1_FDX_s105
                                 + 0.705765 w_k1_IBM_s105
                                 + 0.729536 w_k1_INTC_s105
                                 + 0.691094 w_k1_KO_s105
                                 + 0.628927 w_k1_MAT_s105
                                 + 0.750031 w_k1_MSFT_s105
                                 + 0.82946 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.14444 w_k1_ADM_s105
                                 + 1.26839 w_k1_AXP_s105
                                 + 1.54264 w_k1_DIS_s105
                                 + 1.27394 w_k1_FDX_s105
                                 + 0.945117 w_k1_IBM_s105
                                 + 1.30941 w_k1_INTC_s105
                                 + 1.21007 w_k1_KO_s105
                                 + 0.621023 w_k1_MAT_s105
                                 + 1.2545 w_k1_MSFT_s105 + 1.09978 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.07515 w_k1_ADM_s105
                                 + 1.00548 w_k1_AXP_s105
                                 + 1.10444 w_k1_DIS_s105
                                 + 1.02629 w_k1_FDX_s105
                                 + 0.895804 w_k1_IBM_s105
                                 + 0.98921 w_k1_INTC_s105
                                 + 0.943069 w_k1_KO_s105
                                 + 0.798077 w_k1_MAT_s105
                                 + 1.05675 w_k1_MSFT_s105
                                 + 1.05574 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.824535 w_k1_ADM_s105
                                 + 1.00792 w_k1_AXP_s105
                                 + 0.926148 w_k1_DIS_s105
                                 + 0.899032 w_k1_FDX_s105
                                 + 1.30641 w_k1_IBM_s105
                                 + 0.782852 w_k1_INTC_s105
                                 + 0.880696 w_k1_KO_s105
                                 + 1.09113 w_k1_MAT_s105
                                 + 0.899274 w_k1_MSFT_s105
                                 + 1.01959 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.47578 w_k1_ADM_s105
                                 + 1.77027 w_k1_AXP_s105
                                 + 1.64854 w_k1_DIS_s105
                                 + 1.32744 w_k1_FDX_s105
                                 + 1.51419 w_k1_IBM_s105
                                 + 1.26086 w_k1_INTC_s105
                                 + 1.46602 w_k1_KO_s105 + 1.18477 w_k1_MAT_s105
                                 + 1.35553 w_k1_MSFT_s105
                                 + 1.26855 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.986563 w_k1_ADM_s105
                                 + 1.04474 w_k1_AXP_s105
                                 + 1.00459 w_k1_DIS_s105 + 1.0727 w_k1_FDX_s105
                                 + 1.00284 w_k1_IBM_s105
                                 + 1.01003 w_k1_INTC_s105
                                 + 1.05368 w_k1_KO_s105
                                 + 0.866163 w_k1_MAT_s105
                                 + 1.06851 w_k1_MSFT_s105
                                 + 1.08735 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.26385 w_k1_ADM_s105
                                 + 1.42508 w_k1_AXP_s105
                                 + 1.15791 w_k1_DIS_s105
                                 + 1.22697 w_k1_FDX_s105
                                 + 1.20272 w_k1_IBM_s105
                                 + 1.20126 w_k1_INTC_s105
                                 + 1.05637 w_k1_KO_s105 + 1.19024 w_k1_MAT_s105
                                 + 1.13348 w_k1_MSFT_s105 + 1.1252 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.987661 w_k1_ADM_s105
                                 + 0.853176 w_k1_AXP_s105
                                 + 0.98145 w_k1_DIS_s105
                                 + 0.90265 w_k1_FDX_s105
                                 + 0.973738 w_k1_IBM_s105
                                 + 0.918007 w_k1_INTC_s105
                                 + 0.955306 w_k1_KO_s105
                                 + 0.824109 w_k1_MAT_s105
                                 + 1.01487 w_k1_MSFT_s105
                                 + 0.955885 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.733242 w_k1_ADM_s106
                                 + 0.591837 w_k1_AXP_s106
                                 + 0.531186 w_k1_DIS_s106
                                 + 0.697212 w_k1_FDX_s106
                                 + 0.793607 w_k1_IBM_s106
                                 + 0.763168 w_k1_INTC_s106
                                 + 0.700178 w_k1_KO_s106
                                 + 0.654725 w_k1_MAT_s106
                                 + 0.699239 w_k1_MSFT_s106
                                 + 0.808788 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.23858 w_k1_ADM_s106
                                 + 1.34236 w_k1_AXP_s106
                                 + 1.35317 w_k1_DIS_s106 + 1.2073 w_k1_FDX_s106
                                 + 1.5083 w_k1_IBM_s106
                                 + 1.27637 w_k1_INTC_s106
                                 + 1.36029 w_k1_KO_s106 + 1.59735 w_k1_MAT_s106
                                 + 1.17637 w_k1_MSFT_s106
                                 + 1.18574 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.07089 w_k1_ADM_s106
                                 + 0.916368 w_k1_AXP_s106
                                 + 0.902944 w_k1_DIS_s106
                                 + 0.952197 w_k1_FDX_s106
                                 + 1.01744 w_k1_IBM_s106
                                 + 1.0419 w_k1_INTC_s106
                                 + 0.897045 w_k1_KO_s106
                                 + 0.956099 w_k1_MAT_s106
                                 + 1.01442 w_k1_MSFT_s106
                                 + 1.01039 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.32971 w_k1_ADM_s106
                                 + 1.26423 w_k1_AXP_s106
                                 + 1.29312 w_k1_DIS_s106
                                 + 1.30767 w_k1_FDX_s106
                                 + 0.870762 w_k1_IBM_s106
                                 + 1.34133 w_k1_INTC_s106
                                 + 1.04018 w_k1_KO_s106
                                 + 0.933777 w_k1_MAT_s106
                                 + 1.25733 w_k1_MSFT_s106
                                 + 1.10177 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.800923 w_k1_ADM_s106
                                 + 0.632489 w_k1_AXP_s106
                                 + 0.720162 w_k1_DIS_s106
                                 + 0.816236 w_k1_FDX_s106
                                 + 0.936925 w_k1_IBM_s106
                                 + 0.991208 w_k1_INTC_s106
                                 + 0.844764 w_k1_KO_s106
                                 + 0.981462 w_k1_MAT_s106
                                 + 0.91536 w_k1_MSFT_s106
                                 + 0.957838 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.861619 w_k1_ADM_s106
                                 + 0.906568 w_k1_AXP_s106
                                 + 0.93041 w_k1_DIS_s106
                                 + 0.876275 w_k1_FDX_s106
                                 + 1.35625 w_k1_IBM_s106
                                 + 0.959533 w_k1_INTC_s106
                                 + 0.90897 w_k1_KO_s106 + 1.50338 w_k1_MAT_s106
                                 + 0.833548 w_k1_MSFT_s106
                                 + 1.04523 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.875641 w_k1_ADM_s106
                                 + 0.777466 w_k1_AXP_s106
                                 + 0.736463 w_k1_DIS_s106
                                 + 0.94711 w_k1_FDX_s106
                                 + 0.648469 w_k1_IBM_s106
                                 + 0.968681 w_k1_INTC_s106
                                 + 0.662926 w_k1_KO_s106
                                 + 0.647074 w_k1_MAT_s106
                                 + 0.918881 w_k1_MSFT_s106
                                 + 0.832979 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.9597 w_k1_ADM_s106
                                 + 1.03063 w_k1_AXP_s106
                                 + 1.09311 w_k1_DIS_s106
                                 + 1.02288 w_k1_FDX_s106
                                 + 0.917448 w_k1_IBM_s106
                                 + 1.20683 w_k1_INTC_s106
                                 + 0.931817 w_k1_KO_s106
                                 + 1.00035 w_k1_MAT_s106
                                 + 1.0861 w_k1_MSFT_s106
                                 + 0.956844 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.76633 w_k1_ADM_s107
                                 + 0.820338 w_k1_AXP_s107
                                 + 0.748778 w_k1_DIS_s107
                                 + 0.798014 w_k1_FDX_s107
                                 + 1.08572 w_k1_IBM_s107
                                 + 0.888635 w_k1_INTC_s107
                                 + 0.87941 w_k1_KO_s107
                                 + 0.871015 w_k1_MAT_s107
                                 + 0.865152 w_k1_MSFT_s107
                                 + 0.93095 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.31388 w_k1_ADM_s107
                                 + 1.39309 w_k1_AXP_s107
                                 + 1.45304 w_k1_DIS_s107
                                 + 1.27801 w_k1_FDX_s107
                                 + 1.18327 w_k1_IBM_s107
                                 + 1.2429 w_k1_INTC_s107 + 1.27431 w_k1_KO_s107
                                 + 1.5579 w_k1_MAT_s107
                                 + 1.27253 w_k1_MSFT_s107 + 1.0205 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.76879 w_k1_ADM_s107
                                 + 0.753177 w_k1_AXP_s107
                                 + 0.621577 w_k1_DIS_s107
                                 + 0.795679 w_k1_FDX_s107
                                 + 0.692567 w_k1_IBM_s107
                                 + 0.774737 w_k1_INTC_s107
                                 + 0.621609 w_k1_KO_s107
                                 + 0.817817 w_k1_MAT_s107
                                 + 0.850474 w_k1_MSFT_s107
                                 + 0.82619 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.05149 w_k1_ADM_s107
                                 + 0.885186 w_k1_AXP_s107
                                 + 1.15672 w_k1_DIS_s107
                                 + 0.984415 w_k1_FDX_s107
                                 + 0.934327 w_k1_IBM_s107
                                 + 1.03144 w_k1_INTC_s107
                                 + 1.06418 w_k1_KO_s107
                                 + 0.771343 w_k1_MAT_s107
                                 + 1.12996 w_k1_MSFT_s107
                                 + 1.06433 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00058 w_k1_ADM_s107
                                 + 1.18219 w_k1_AXP_s107
                                 + 0.947257 w_k1_DIS_s107
                                 + 0.939797 w_k1_FDX_s107
                                 + 1.56295 w_k1_IBM_s107
                                 + 0.917424 w_k1_INTC_s107
                                 + 1.07701 w_k1_KO_s107 + 1.82986 w_k1_MAT_s107
                                 + 0.968774 w_k1_MSFT_s107
                                 + 1.18318 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.07554 w_k1_ADM_s107
                                 + 1.32293 w_k1_AXP_s107
                                 + 0.869312 w_k1_DIS_s107
                                 + 0.994471 w_k1_FDX_s107
                                 + 0.997179 w_k1_IBM_s107
                                 + 1.07011 w_k1_INTC_s107
                                 + 0.904984 w_k1_KO_s107
                                 + 0.959239 w_k1_MAT_s107
                                 + 1.01888 w_k1_MSFT_s107
                                 + 1.14126 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.00462 w_k1_ADM_s107
                                 + 1.12598 w_k1_AXP_s107
                                 + 1.21699 w_k1_DIS_s107
                                 + 1.05422 w_k1_FDX_s107
                                 + 1.49791 w_k1_IBM_s107
                                 + 1.03173 w_k1_INTC_s107
                                 + 1.27767 w_k1_KO_s107 + 1.83764 w_k1_MAT_s107
                                 + 1.19076 w_k1_MSFT_s107
                                 + 1.23157 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.878311 w_k1_ADM_s107
                                 + 0.838838 w_k1_AXP_s107
                                 + 0.856902 w_k1_DIS_s107
                                 + 0.883781 w_k1_FDX_s107
                                 + 0.709879 w_k1_IBM_s107
                                 + 0.813136 w_k1_INTC_s107
                                 + 0.767939 w_k1_KO_s107
                                 + 0.723584 w_k1_MAT_s107
                                 + 0.971676 w_k1_MSFT_s107
                                 + 0.906296 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.01069 w_k1_ADM_s110
                                 + 1.4426 w_k1_AXP_s110 + 1.12565 w_k1_DIS_s110
                                 + 1.10292 w_k1_FDX_s110
                                 + 1.16347 w_k1_IBM_s110
                                 + 0.961586 w_k1_INTC_s110
                                 + 1.04686 w_k1_KO_s110 + 1.40349 w_k1_MAT_s110
                                 + 1.04845 w_k1_MSFT_s110
                                 + 1.06863 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.892327 w_k1_ADM_s110
                                 + 0.651472 w_k1_AXP_s110
                                 + 0.59718 w_k1_DIS_s110
                                 + 0.679449 w_k1_FDX_s110
                                 + 0.711945 w_k1_IBM_s110
                                 + 0.679607 w_k1_INTC_s110
                                 + 0.710858 w_k1_KO_s110
                                 + 0.618679 w_k1_MAT_s110
                                 + 0.78174 w_k1_MSFT_s110
                                 + 0.731326 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.891512 w_k1_ADM_s110
                                 + 1.22165 w_k1_AXP_s110 + 1.2286 w_k1_DIS_s110
                                 + 1.03294 w_k1_FDX_s110 + 1.4435 w_k1_IBM_s110
                                 + 1.02105 w_k1_INTC_s110
                                 + 1.08177 w_k1_KO_s110 + 1.53947 w_k1_MAT_s110
                                 + 0.997386 w_k1_MSFT_s110
                                 + 0.989341 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.822271 w_k1_ADM_s110
                                 + 1.09567 w_k1_AXP_s110
                                 + 0.944798 w_k1_DIS_s110
                                 + 1.05211 w_k1_FDX_s110
                                 + 0.713534 w_k1_IBM_s110
                                 + 1.07353 w_k1_INTC_s110
                                 + 0.787483 w_k1_KO_s110
                                 + 0.790421 w_k1_MAT_s110
                                 + 1.06351 w_k1_MSFT_s110
                                 + 0.965797 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.906317 w_k1_ADM_s110
                                 + 0.759107 w_k1_AXP_s110
                                 + 0.658622 w_k1_DIS_s110
                                 + 0.75985 w_k1_FDX_s110
                                 + 0.955749 w_k1_IBM_s110
                                 + 0.593234 w_k1_INTC_s110
                                 + 0.746528 w_k1_KO_s110
                                 + 0.823853 w_k1_MAT_s110
                                 + 0.701162 w_k1_MSFT_s110
                                 + 0.878059 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.81439 w_k1_ADM_s110
                                 + 0.816811 w_k1_AXP_s110
                                 + 0.687946 w_k1_DIS_s110
                                 + 0.787709 w_k1_FDX_s110
                                 + 1.21296 w_k1_IBM_s110
                                 + 0.85501 w_k1_INTC_s110
                                 + 0.924125 w_k1_KO_s110
                                 + 1.01921 w_k1_MAT_s110
                                 + 0.721412 w_k1_MSFT_s110
                                 + 0.805445 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.961806 w_k1_ADM_s110
                                 + 1.09402 w_k1_AXP_s110
                                 + 1.24501 w_k1_DIS_s110
                                 + 1.11435 w_k1_FDX_s110
                                 + 1.74811 w_k1_IBM_s110
                                 + 1.14311 w_k1_INTC_s110
                                 + 1.35878 w_k1_KO_s110 + 1.52508 w_k1_MAT_s110
                                 + 1.08063 w_k1_MSFT_s110
                                 + 1.02698 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.24035 w_k1_ADM_s110
                                 + 1.0239 w_k1_AXP_s110 + 1.38996 w_k1_DIS_s110
                                 + 1.0678 w_k1_FDX_s110 + 1.0229 w_k1_IBM_s110
                                 + 1.10625 w_k1_INTC_s110
                                 + 1.18554 w_k1_KO_s110
                                 + 0.785926 w_k1_MAT_s110
                                 + 1.14715 w_k1_MSFT_s110
                                 + 1.12417 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.17329 w_k1_ADM_s111
                                 + 1.79904 w_k1_AXP_s111
                                 + 2.00073 w_k1_DIS_s111
                                 + 1.49271 w_k1_FDX_s111
                                 + 1.80682 w_k1_IBM_s111
                                 + 1.36981 w_k1_INTC_s111
                                 + 1.49782 w_k1_KO_s111 + 2.1129 w_k1_MAT_s111
                                 + 1.47899 w_k1_MSFT_s111
                                 + 1.49942 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.125 w_k1_ADM_s111
                                 + 1.05014 w_k1_AXP_s111
                                 + 0.930716 w_k1_DIS_s111
                                 + 1.08702 w_k1_FDX_s111
                                 + 0.80464 w_k1_IBM_s111
                                 + 0.832745 w_k1_INTC_s111
                                 + 0.81912 w_k1_KO_s111
                                 + 0.739551 w_k1_MAT_s111
                                 + 0.959676 w_k1_MSFT_s111
                                 + 0.987647 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.11023 w_k1_ADM_s111
                                 + 1.18415 w_k1_AXP_s111
                                 + 1.70722 w_k1_DIS_s111
                                 + 1.44932 w_k1_FDX_s111
                                 + 1.04196 w_k1_IBM_s111
                                 + 1.12729 w_k1_INTC_s111
                                 + 1.06033 w_k1_KO_s111 + 1.41713 w_k1_MAT_s111
                                 + 1.43685 w_k1_MSFT_s111 + 1.2653 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.845546 w_k1_ADM_s111
                                 + 1.05406 w_k1_AXP_s111
                                 + 1.07125 w_k1_DIS_s111
                                 + 0.988482 w_k1_FDX_s111
                                 + 1.55179 w_k1_IBM_s111
                                 + 0.79767 w_k1_INTC_s111
                                 + 1.13206 w_k1_KO_s111 + 1.6493 w_k1_MAT_s111
                                 + 0.965178 w_k1_MSFT_s111
                                 + 1.08205 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.15832 w_k1_ADM_s111
                                 + 1.46895 w_k1_AXP_s111 + 1.3242 w_k1_DIS_s111
                                 + 1.24883 w_k1_FDX_s111 + 1.3994 w_k1_IBM_s111
                                 + 1.2452 w_k1_INTC_s111 + 1.06697 w_k1_KO_s111
                                 + 1.25533 w_k1_MAT_s111
                                 + 1.24253 w_k1_MSFT_s111
                                 + 1.24853 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.787427 w_k1_ADM_s111
                                 + 0.699945 w_k1_AXP_s111
                                 + 0.940666 w_k1_DIS_s111
                                 + 0.827632 w_k1_FDX_s111
                                 + 0.96234 w_k1_IBM_s111
                                 + 0.856284 w_k1_INTC_s111
                                 + 0.793916 w_k1_KO_s111
                                 + 0.821556 w_k1_MAT_s111
                                 + 0.896723 w_k1_MSFT_s111
                                 + 0.98416 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.853662 w_k1_ADM_s111
                                 + 0.814366 w_k1_AXP_s111
                                 + 0.833245 w_k1_DIS_s111
                                 + 0.779474 w_k1_FDX_s111
                                 + 1.10601 w_k1_IBM_s111
                                 + 0.774481 w_k1_INTC_s111
                                 + 0.81799 w_k1_KO_s111 + 1.0791 w_k1_MAT_s111
                                 + 0.829885 w_k1_MSFT_s111
                                 + 1.01281 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.767035 w_k1_ADM_s111
                                 + 0.729534 w_k1_AXP_s111
                                 + 0.640472 w_k1_DIS_s111
                                 + 0.796087 w_k1_FDX_s111
                                 + 0.547544 w_k1_IBM_s111
                                 + 0.759327 w_k1_INTC_s111
                                 + 0.604967 w_k1_KO_s111
                                 + 0.662902 w_k1_MAT_s111
                                 + 0.826515 w_k1_MSFT_s111
                                 + 0.859399 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.18643 w_k1_ADM_s112
                                 + 1.1798 w_k1_AXP_s112 + 1.06159 w_k1_DIS_s112
                                 + 1.13773 w_k1_FDX_s112
                                 + 0.739583 w_k1_IBM_s112
                                 + 0.763621 w_k1_INTC_s112
                                 + 1.07551 w_k1_KO_s112
                                 + 0.911719 w_k1_MAT_s112
                                 + 1.13444 w_k1_MSFT_s112
                                 + 0.981616 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.02593 w_k1_ADM_s112
                                 + 0.95879 w_k1_AXP_s112
                                 + 1.46806 w_k1_DIS_s112
                                 + 1.16396 w_k1_FDX_s112
                                 + 1.07696 w_k1_IBM_s112
                                 + 1.14199 w_k1_INTC_s112
                                 + 1.39517 w_k1_KO_s112
                                 + 0.839991 w_k1_MAT_s112
                                 + 1.27871 w_k1_MSFT_s112
                                 + 1.01815 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.09808 w_k1_ADM_s112
                                 + 1.24376 w_k1_AXP_s112
                                 + 1.14567 w_k1_DIS_s112
                                 + 1.13255 w_k1_FDX_s112
                                 + 1.03659 w_k1_IBM_s112
                                 + 0.953111 w_k1_INTC_s112
                                 + 1.09585 w_k1_KO_s112 + 1.11301 w_k1_MAT_s112
                                 + 1.15943 w_k1_MSFT_s112
                                 + 1.19393 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.791493 w_k1_ADM_s112
                                 + 0.768761 w_k1_AXP_s112
                                 + 0.973274 w_k1_DIS_s112
                                 + 0.80533 w_k1_FDX_s112
                                 + 1.37333 w_k1_IBM_s112
                                 + 0.768917 w_k1_INTC_s112
                                 + 0.991783 w_k1_KO_s112 + 1.0979 w_k1_MAT_s112
                                 + 0.767062 w_k1_MSFT_s112
                                 + 0.938446 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.877069 w_k1_ADM_s112
                                 + 0.724247 w_k1_AXP_s112
                                 + 0.746871 w_k1_DIS_s112
                                 + 0.761665 w_k1_FDX_s112
                                 + 0.908102 w_k1_IBM_s112
                                 + 0.560377 w_k1_INTC_s112
                                 + 0.912007 w_k1_KO_s112
                                 + 0.853551 w_k1_MAT_s112
                                 + 0.840617 w_k1_MSFT_s112
                                 + 0.940871 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.907834 w_k1_ADM_s112
                                 + 0.85833 w_k1_AXP_s112
                                 + 0.759944 w_k1_DIS_s112
                                 + 0.823727 w_k1_FDX_s112
                                 + 0.690771 w_k1_IBM_s112
                                 + 0.638468 w_k1_INTC_s112
                                 + 0.791071 w_k1_KO_s112
                                 + 0.69454 w_k1_MAT_s112
                                 + 0.759362 w_k1_MSFT_s112
                                 + 0.90764 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.26202 w_k1_ADM_s112
                                 + 1.47677 w_k1_AXP_s112 + 1.1697 w_k1_DIS_s112
                                 + 1.11745 w_k1_FDX_s112
                                 + 1.67521 w_k1_IBM_s112
                                 + 0.977037 w_k1_INTC_s112
                                 + 1.36935 w_k1_KO_s112 + 1.47449 w_k1_MAT_s112
                                 + 1.03864 w_k1_MSFT_s112
                                 + 1.23952 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.45358 w_k1_ADM_s112
                                 + 1.78304 w_k1_AXP_s112
                                 + 1.89377 w_k1_DIS_s112
                                 + 1.53048 w_k1_FDX_s112
                                 + 1.23258 w_k1_IBM_s112
                                 + 1.5573 w_k1_INTC_s112 + 1.46093 w_k1_KO_s112
                                 + 1.05402 w_k1_MAT_s112
                                 + 1.57029 w_k1_MSFT_s112
                                 + 1.36895 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.2009 w_k1_ADM_s113
                                 + 0.692517 w_k1_AXP_s113
                                 + 0.61965 w_k1_DIS_s113
                                 + 0.789792 w_k1_FDX_s113
                                 + 0.684997 w_k1_IBM_s113
                                 + 0.96277 w_k1_INTC_s113
                                 + 0.689855 w_k1_KO_s113
                                 + 0.556241 w_k1_MAT_s113
                                 + 0.851707 w_k1_MSFT_s113
                                 + 0.896281 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.32626 w_k1_ADM_s113
                                 + 0.913704 w_k1_AXP_s113
                                 + 1.15668 w_k1_DIS_s113
                                 + 1.05758 w_k1_FDX_s113
                                 + 1.03896 w_k1_IBM_s113
                                 + 1.22567 w_k1_INTC_s113 + 1.0977 w_k1_KO_s113
                                 + 0.886131 w_k1_MAT_s113
                                 + 1.19249 w_k1_MSFT_s113
                                 + 0.959521 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.03632 w_k1_ADM_s113
                                 + 1.05721 w_k1_AXP_s113 + 1.1596 w_k1_DIS_s113
                                 + 1.17711 w_k1_FDX_s113
                                 + 0.71066 w_k1_IBM_s113
                                 + 0.897359 w_k1_INTC_s113
                                 + 1.03589 w_k1_KO_s113
                                 + 0.918669 w_k1_MAT_s113
                                 + 1.11872 w_k1_MSFT_s113
                                 + 1.01738 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.14311 w_k1_ADM_s113
                                 + 0.801844 w_k1_AXP_s113
                                 + 0.862918 w_k1_DIS_s113
                                 + 0.755813 w_k1_FDX_s113
                                 + 1.38332 w_k1_IBM_s113
                                 + 0.916379 w_k1_INTC_s113
                                 + 1.27388 w_k1_KO_s113 + 1.32709 w_k1_MAT_s113
                                 + 1.00665 w_k1_MSFT_s113 + 0.9808 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.42504 w_k1_ADM_s113
                                 + 1.27105 w_k1_AXP_s113
                                 + 1.40828 w_k1_DIS_s113
                                 + 1.30514 w_k1_FDX_s113 + 1.1747 w_k1_IBM_s113
                                 + 1.44218 w_k1_INTC_s113
                                 + 1.40313 w_k1_KO_s113 + 1.40753 w_k1_MAT_s113
                                 + 1.45632 w_k1_MSFT_s113
                                 + 1.22595 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.13988 w_k1_ADM_s113
                                 + 0.638321 w_k1_AXP_s113
                                 + 0.737652 w_k1_DIS_s113
                                 + 0.797994 w_k1_FDX_s113
                                 + 0.905207 w_k1_IBM_s113
                                 + 1.0036 w_k1_INTC_s113
                                 + 0.913176 w_k1_KO_s113
                                 + 0.726122 w_k1_MAT_s113
                                 + 1.03151 w_k1_MSFT_s113
                                 + 0.955278 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.38862 w_k1_ADM_s113
                                 + 1.0914 w_k1_AXP_s113 + 1.33653 w_k1_DIS_s113
                                 + 1.11149 w_k1_FDX_s113
                                 + 1.47857 w_k1_IBM_s113
                                 + 1.3362 w_k1_INTC_s113 + 1.4759 w_k1_KO_s113
                                 + 1.15561 w_k1_MAT_s113
                                 + 1.31559 w_k1_MSFT_s113
                                 + 1.12611 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.59458 w_k1_ADM_s113
                                 + 1.37516 w_k1_AXP_s113
                                 + 1.10966 w_k1_DIS_s113
                                 + 1.30191 w_k1_FDX_s113
                                 + 0.894419 w_k1_IBM_s113
                                 + 1.32231 w_k1_INTC_s113
                                 + 1.22108 w_k1_KO_s113
                                 + 0.969525 w_k1_MAT_s113
                                 + 1.23702 w_k1_MSFT_s113
                                 + 1.01664 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.971283 w_k1_ADM_s114
                                 + 0.975618 w_k1_AXP_s114
                                 + 0.863101 w_k1_DIS_s114
                                 + 0.910763 w_k1_FDX_s114
                                 + 0.994471 w_k1_IBM_s114
                                 + 0.784028 w_k1_INTC_s114
                                 + 0.782513 w_k1_KO_s114
                                 + 0.925426 w_k1_MAT_s114
                                 + 0.802308 w_k1_MSFT_s114
                                 + 0.915907 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.26679 w_k1_ADM_s114
                                 + 1.07493 w_k1_AXP_s114
                                 + 1.15175 w_k1_DIS_s114
                                 + 1.13851 w_k1_FDX_s114
                                 + 1.01859 w_k1_IBM_s114
                                 + 0.998627 w_k1_INTC_s114
                                 + 0.994997 w_k1_KO_s114
                                 + 0.739372 w_k1_MAT_s114
                                 + 0.954103 w_k1_MSFT_s114
                                 + 1.01198 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.932797 w_k1_ADM_s114
                                 + 0.79797 w_k1_AXP_s114
                                 + 1.09209 w_k1_DIS_s114
                                 + 1.02574 w_k1_FDX_s114
                                 + 1.47615 w_k1_IBM_s114
                                 + 1.09999 w_k1_INTC_s114
                                 + 1.12862 w_k1_KO_s114 + 1.11836 w_k1_MAT_s114
                                 + 1.01515 w_k1_MSFT_s114
                                 + 1.08449 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.00429 w_k1_ADM_s114
                                 + 1.0047 w_k1_AXP_s114
                                 + 0.859238 w_k1_DIS_s114
                                 + 1.09397 w_k1_FDX_s114
                                 + 0.721249 w_k1_IBM_s114
                                 + 0.900751 w_k1_INTC_s114
                                 + 0.784533 w_k1_KO_s114
                                 + 0.559331 w_k1_MAT_s114
                                 + 0.858654 w_k1_MSFT_s114
                                 + 0.883245 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.622096 w_k1_ADM_s114
                                 + 0.657799 w_k1_AXP_s114
                                 + 0.619288 w_k1_DIS_s114
                                 + 0.727485 w_k1_FDX_s114 + 1.104 w_k1_IBM_s114
                                 + 0.734503 w_k1_INTC_s114
                                 + 0.802719 w_k1_KO_s114
                                 + 0.780585 w_k1_MAT_s114
                                 + 0.744755 w_k1_MSFT_s114
                                 + 0.908521 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.933733 w_k1_ADM_s114
                                 + 0.696118 w_k1_AXP_s114
                                 + 0.647405 w_k1_DIS_s114
                                 + 0.828399 w_k1_FDX_s114
                                 + 0.845239 w_k1_IBM_s114
                                 + 0.890089 w_k1_INTC_s114
                                 + 0.734499 w_k1_KO_s114
                                 + 0.621432 w_k1_MAT_s114
                                 + 0.773786 w_k1_MSFT_s114
                                 + 0.861027 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.31183 w_k1_ADM_s114
                                 + 1.4363 w_k1_AXP_s114 + 1.55511 w_k1_DIS_s114
                                 + 1.45015 w_k1_FDX_s114
                                 + 0.972584 w_k1_IBM_s114
                                 + 1.18068 w_k1_INTC_s114
                                 + 1.06359 w_k1_KO_s114 + 1.05492 w_k1_MAT_s114
                                 + 1.30852 w_k1_MSFT_s114
                                 + 1.11633 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.11196 w_k1_ADM_s114
                                 + 1.42674 w_k1_AXP_s114
                                 + 1.20261 w_k1_DIS_s114
                                 + 1.25199 w_k1_FDX_s114
                                 + 1.22263 w_k1_IBM_s114
                                 + 1.0364 w_k1_INTC_s114 + 1.00463 w_k1_KO_s114
                                 + 1.18459 w_k1_MAT_s114
                                 + 1.03252 w_k1_MSFT_s114 + 1.1773 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.37855 w_k1_ADM_s115
                                 + 1.26466 w_k1_AXP_s115
                                 + 1.34025 w_k1_DIS_s115
                                 + 1.36111 w_k1_FDX_s115
                                 + 0.853571 w_k1_IBM_s115
                                 + 1.18137 w_k1_INTC_s115
                                 + 1.04379 w_k1_KO_s115
                                 + 0.815318 w_k1_MAT_s115
                                 + 1.26054 w_k1_MSFT_s115 + 1.1237 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.916414 w_k1_ADM_s115
                                 + 0.924498 w_k1_AXP_s115
                                 + 0.894583 w_k1_DIS_s115
                                 + 0.92146 w_k1_FDX_s115
                                 + 0.714279 w_k1_IBM_s115
                                 + 0.73737 w_k1_INTC_s115
                                 + 0.951184 w_k1_KO_s115
                                 + 0.70414 w_k1_MAT_s115
                                 + 0.961691 w_k1_MSFT_s115
                                 + 1.06082 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.09616 w_k1_ADM_s115
                                 + 1.09926 w_k1_AXP_s115
                                 + 1.14512 w_k1_DIS_s115
                                 + 1.08999 w_k1_FDX_s115
                                 + 0.981588 w_k1_IBM_s115
                                 + 1.15683 w_k1_INTC_s115
                                 + 1.18486 w_k1_KO_s115 + 1.0871 w_k1_MAT_s115
                                 + 1.44297 w_k1_MSFT_s115
                                 + 1.11539 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.624371 w_k1_ADM_s115
                                 + 0.596943 w_k1_AXP_s115
                                 + 0.6036 w_k1_DIS_s115
                                 + 0.656744 w_k1_FDX_s115
                                 + 0.784546 w_k1_IBM_s115
                                 + 0.720777 w_k1_INTC_s115
                                 + 0.799701 w_k1_KO_s115
                                 + 0.766659 w_k1_MAT_s115
                                 + 0.846554 w_k1_MSFT_s115
                                 + 0.813746 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04009 w_k1_ADM_s115
                                 + 1.2087 w_k1_AXP_s115 + 1.24302 w_k1_DIS_s115
                                 + 1.14513 w_k1_FDX_s115
                                 + 1.23034 w_k1_IBM_s115
                                 + 0.860127 w_k1_INTC_s115
                                 + 1.25231 w_k1_KO_s115 + 1.57186 w_k1_MAT_s115
                                 + 1.1314 w_k1_MSFT_s115 + 1.21738 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.31985 w_k1_ADM_s115
                                 + 1.28829 w_k1_AXP_s115
                                 + 1.74104 w_k1_DIS_s115
                                 + 1.24352 w_k1_FDX_s115
                                 + 1.38253 w_k1_IBM_s115
                                 + 1.13142 w_k1_INTC_s115
                                 + 1.82453 w_k1_KO_s115 + 1.80002 w_k1_MAT_s115
                                 + 1.56401 w_k1_MSFT_s115
                                 + 1.42575 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.775077 w_k1_ADM_s115
                                 + 0.581643 w_k1_AXP_s115
                                 + 0.803577 w_k1_DIS_s115
                                 + 0.734792 w_k1_FDX_s115
                                 + 1.31941 w_k1_IBM_s115
                                 + 0.664894 w_k1_INTC_s115
                                 + 1.27871 w_k1_KO_s115 + 1.23233 w_k1_MAT_s115
                                 + 1.02814 w_k1_MSFT_s115
                                 + 0.942879 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.812207 w_k1_ADM_s115
                                 + 0.886665 w_k1_AXP_s115
                                 + 0.77396 w_k1_DIS_s115
                                 + 0.82463 w_k1_FDX_s115
                                 + 0.972511 w_k1_IBM_s115
                                 + 0.888331 w_k1_INTC_s115
                                 + 1.02907 w_k1_KO_s115
                                 + 0.999204 w_k1_MAT_s115
                                 + 0.937782 w_k1_MSFT_s115
                                 + 1.00116 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.617967 w_k1_ADM_s116
                                 + 0.530555 w_k1_AXP_s116
                                 + 0.718989 w_k1_DIS_s116
                                 + 0.661133 w_k1_FDX_s116
                                 + 0.973371 w_k1_IBM_s116
                                 + 0.7284 w_k1_INTC_s116
                                 + 0.854682 w_k1_KO_s116
                                 + 0.877781 w_k1_MAT_s116
                                 + 0.765808 w_k1_MSFT_s116
                                 + 0.863771 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.75839 w_k1_ADM_s116
                                 + 0.907537 w_k1_AXP_s116
                                 + 0.909129 w_k1_DIS_s116
                                 + 0.817985 w_k1_FDX_s116
                                 + 1.26933 w_k1_IBM_s116
                                 + 0.919525 w_k1_INTC_s116
                                 + 1.08888 w_k1_KO_s116 + 1.07971 w_k1_MAT_s116
                                 + 0.789949 w_k1_MSFT_s116
                                 + 0.903539 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.33882 w_k1_ADM_s116
                                 + 1.29137 w_k1_AXP_s116
                                 + 1.71381 w_k1_DIS_s116
                                 + 1.46754 w_k1_FDX_s116
                                 + 1.20181 w_k1_IBM_s116
                                 + 1.28042 w_k1_INTC_s116
                                 + 1.44217 w_k1_KO_s116 + 1.07499 w_k1_MAT_s116
                                 + 1.32916 w_k1_MSFT_s116 + 1.3353 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.1088 w_k1_ADM_s116
                                 + 1.10959 w_k1_AXP_s116
                                 + 1.22709 w_k1_DIS_s116
                                 + 1.03823 w_k1_FDX_s116
                                 + 1.29446 w_k1_IBM_s116
                                 + 1.0944 w_k1_INTC_s116 + 1.23548 w_k1_KO_s116
                                 + 1.38341 w_k1_MAT_s116
                                 + 0.897713 w_k1_MSFT_s116
                                 + 1.07745 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.781577 w_k1_ADM_s116
                                 + 0.655814 w_k1_AXP_s116
                                 + 0.777037 w_k1_DIS_s116
                                 + 0.836807 w_k1_FDX_s116
                                 + 0.644794 w_k1_IBM_s116
                                 + 0.958163 w_k1_INTC_s116
                                 + 0.674322 w_k1_KO_s116
                                 + 0.424957 w_k1_MAT_s116
                                 + 0.76367 w_k1_MSFT_s116
                                 + 0.790785 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.883732 w_k1_ADM_s116
                                 + 1.01236 w_k1_AXP_s116
                                 + 0.997354 w_k1_DIS_s116
                                 + 1.06687 w_k1_FDX_s116
                                 + 0.766871 w_k1_IBM_s116
                                 + 1.00207 w_k1_INTC_s116
                                 + 0.944209 w_k1_KO_s116
                                 + 0.94584 w_k1_MAT_s116
                                 + 0.954574 w_k1_MSFT_s116
                                 + 0.903629 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.98351 w_k1_ADM_s116
                                 + 1.05015 w_k1_AXP_s116
                                 + 1.49837 w_k1_DIS_s116
                                 + 1.25261 w_k1_FDX_s116
                                 + 1.12569 w_k1_IBM_s116
                                 + 1.16992 w_k1_INTC_s116
                                 + 1.25514 w_k1_KO_s116
                                 + 0.738519 w_k1_MAT_s116
                                 + 1.084 w_k1_MSFT_s116 + 1.17556 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.941258 w_k1_ADM_s116
                                 + 0.846797 w_k1_AXP_s116
                                 + 0.613141 w_k1_DIS_s116
                                 + 0.771877 w_k1_FDX_s116
                                 + 0.744577 w_k1_IBM_s116
                                 + 0.733889 w_k1_INTC_s116
                                 + 0.715586 w_k1_KO_s116
                                 + 0.731401 w_k1_MAT_s116
                                 + 0.678902 w_k1_MSFT_s116
                                 + 0.857098 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.01902 w_k1_ADM_s117
                                 + 1.1152 w_k1_AXP_s117 + 1.23808 w_k1_DIS_s117
                                 + 1.14341 w_k1_FDX_s117
                                 + 1.21969 w_k1_IBM_s117
                                 + 1.15855 w_k1_INTC_s117
                                 + 1.15699 w_k1_KO_s117 + 1.17173 w_k1_MAT_s117
                                 + 1.16638 w_k1_MSFT_s117
                                 + 1.02812 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.883257 w_k1_ADM_s117
                                 + 1.35172 w_k1_AXP_s117 + 1.4535 w_k1_DIS_s117
                                 + 1.23684 w_k1_FDX_s117
                                 + 1.49029 w_k1_IBM_s117
                                 + 1.35703 w_k1_INTC_s117
                                 + 1.35061 w_k1_KO_s117 + 2.0623 w_k1_MAT_s117
                                 + 1.45632 w_k1_MSFT_s117
                                 + 1.18263 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.809461 w_k1_ADM_s117
                                 + 0.758203 w_k1_AXP_s117
                                 + 0.64618 w_k1_DIS_s117
                                 + 0.748721 w_k1_FDX_s117
                                 + 1.03418 w_k1_IBM_s117
                                 + 0.847108 w_k1_INTC_s117
                                 + 0.756449 w_k1_KO_s117
                                 + 1.19138 w_k1_MAT_s117
                                 + 0.837087 w_k1_MSFT_s117
                                 + 0.870567 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.1103 w_k1_ADM_s117
                                 + 1.24013 w_k1_AXP_s117
                                 + 0.863852 w_k1_DIS_s117
                                 + 1.17551 w_k1_FDX_s117
                                 + 0.781488 w_k1_IBM_s117
                                 + 1.15233 w_k1_INTC_s117
                                 + 0.813974 w_k1_KO_s117
                                 + 0.858328 w_k1_MAT_s117
                                 + 1.04226 w_k1_MSFT_s117 + 1.0312 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.870087 w_k1_ADM_s117
                                 + 0.929279 w_k1_AXP_s117
                                 + 0.774161 w_k1_DIS_s117
                                 + 0.951333 w_k1_FDX_s117
                                 + 0.976639 w_k1_IBM_s117
                                 + 0.998201 w_k1_INTC_s117
                                 + 0.834748 w_k1_KO_s117
                                 + 1.19848 w_k1_MAT_s117
                                 + 0.973336 w_k1_MSFT_s117
                                 + 0.929842 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.25324 w_k1_ADM_s117
                                 + 0.759317 w_k1_AXP_s117
                                 + 1.05605 w_k1_DIS_s117
                                 + 0.979349 w_k1_FDX_s117
                                 + 0.768098 w_k1_IBM_s117
                                 + 1.09892 w_k1_INTC_s117
                                 + 0.954659 w_k1_KO_s117
                                 + 0.778869 w_k1_MAT_s117
                                 + 1.15639 w_k1_MSFT_s117
                                 + 0.866327 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.873124 w_k1_ADM_s117
                                 + 0.792896 w_k1_AXP_s117
                                 + 0.916516 w_k1_DIS_s117
                                 + 0.852399 w_k1_FDX_s117
                                 + 1.35696 w_k1_IBM_s117
                                 + 0.951725 w_k1_INTC_s117
                                 + 1.08016 w_k1_KO_s117 + 1.43942 w_k1_MAT_s117
                                 + 0.893933 w_k1_MSFT_s117
                                 + 0.861632 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.680108 w_k1_ADM_s117
                                 + 0.712722 w_k1_AXP_s117
                                 + 0.651486 w_k1_DIS_s117
                                 + 0.741407 w_k1_FDX_s117
                                 + 0.731638 w_k1_IBM_s117
                                 + 1.0289 w_k1_INTC_s117
                                 + 0.737655 w_k1_KO_s117
                                 + 0.679777 w_k1_MAT_s117
                                 + 0.930278 w_k1_MSFT_s117
                                 + 0.765005 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.023820756706516 w_k1_ADM_s000
                                 + 0.0209145516568725 w_k1_ADM_s001
                                 + 0.0178921290191554 w_k1_ADM_s002
                                 + 0.0701806691878409 w_k1_ADM_s003
                                 + 0.0336062606756502 w_k1_ADM_s004
                                 + 0.036126837192014 w_k1_ADM_s005
                                 + 0.0387122285935638 w_k1_ADM_s006
                                 + 0.0187342448439816 w_k1_ADM_s007
                                 + 0.014710132322148 w_k1_ADM_s010
                                 + 0.0170429537584706 w_k1_ADM_s011
                                 + 0.049759599992738 w_k1_ADM_s012
                                 + 0.0100701755419017 w_k1_ADM_s013
                                 + 0.0226081245000451 w_k1_ADM_s014
                                 + 0.0132709740215182 w_k1_ADM_s015
                                 + 0.0107641139715616 w_k1_ADM_s016
                                 + 0.0201694508162953 w_k1_ADM_s017
                                 + 0.0214506709286213 w_k1_ADM_s100
                                 + 0.0265194995768938 w_k1_ADM_s101
                                 + 0.0716063172554931 w_k1_ADM_s102
                                 + 0.0425007386696385 w_k1_ADM_s103
                                 + 0.118102441292849 w_k1_ADM_s104
                                 + 0.030773595096822 w_k1_ADM_s105
                                 + 0.0612629043895813 w_k1_ADM_s106
                                 + 0.113416224758335 w_k1_ADM_s107
                                 + 0.0174849565367353 w_k1_ADM_s110
                                 + 0.0149162039230843 w_k1_ADM_s111
                                 + 0.012021464277768 w_k1_ADM_s112
                                 + 0.0163512771223688 w_k1_ADM_s113
                                 + 0.0185854087413304 w_k1_ADM_s114
                                 + 0.0130598450771086 w_k1_ADM_s115
                                 + 0.00654855571061793 w_k1_ADM_s116
                                 + 0.0105166294480889 w_k1_ADM_s117
                                 + 0.026813614030593 w_k1_AXP_s000
                                 + 0.0195912518255378 w_k1_AXP_s001
                                 + 0.0190310106141625 w_k1_AXP_s002
                                 + 0.0749694526741946 w_k1_AXP_s003
                                 + 0.0329273163963743 w_k1_AXP_s004
                                 + 0.0359766127383285 w_k1_AXP_s005
                                 + 0.0284488979821784 w_k1_AXP_s006
                                 + 0.0181873005011424 w_k1_AXP_s007
                                 + 0.0164379818097171 w_k1_AXP_s010
                                 + 0.0154996943773045 w_k1_AXP_s011
                                 + 0.0525078539786445 w_k1_AXP_s012
                                 + 0.0112658731808697 w_k1_AXP_s013
                                 + 0.0219412966987264 w_k1_AXP_s014
                                 + 0.0122868651394796 w_k1_AXP_s015
                                 + 0.0106578357366856 w_k1_AXP_s016
                                 + 0.0213481801094494 w_k1_AXP_s017
                                 + 0.0247814354388048 w_k1_AXP_s100
                                 + 0.0237440886171479 w_k1_AXP_s101
                                 + 0.0776423564016388 w_k1_AXP_s102
                                 + 0.0418796399231361 w_k1_AXP_s103
                                 + 0.125395834045724 w_k1_AXP_s104
                                 + 0.0342051163442246 w_k1_AXP_s105
                                 + 0.0586137278064498 w_k1_AXP_s106
                                 + 0.117077750464547 w_k1_AXP_s107
                                 + 0.0183639109417874 w_k1_AXP_s110
                                 + 0.0160577946561021 w_k1_AXP_s111
                                 + 0.0121409228488822 w_k1_AXP_s112
                                 + 0.0126225655789782 w_k1_AXP_s113
                                 + 0.0180086271473077 w_k1_AXP_s114
                                 + 0.0130028031078577 w_k1_AXP_s115
                                 + 0.00657085452190052 w_k1_AXP_s116
                                 + 0.0106933789119126 w_k1_AXP_s117
                                 + 0.0296416315957403 w_k1_DIS_s000
                                 + 0.0226254092302794 w_k1_DIS_s001
                                 + 0.0183323526816891 w_k1_DIS_s002
                                 + 0.0720602192754013 w_k1_DIS_s003
                                 + 0.0329187341737161 w_k1_DIS_s004
                                 + 0.0343376859105364 w_k1_DIS_s005
                                 + 0.031299924707077 w_k1_DIS_s006
                                 + 0.0180600582818295 w_k1_DIS_s007
                                 + 0.0189634355426618 w_k1_DIS_s010
                                 + 0.015505619558477 w_k1_DIS_s011
                                 + 0.0545338257841604 w_k1_DIS_s012
                                 + 0.0109624680165577 w_k1_DIS_s013
                                 + 0.0210382707492212 w_k1_DIS_s014
                                 + 0.0133004673818523 w_k1_DIS_s015
                                 + 0.0104832700916924 w_k1_DIS_s016
                                 + 0.0213502479198461 w_k1_DIS_s017
                                 + 0.0244458983185286 w_k1_DIS_s100
                                 + 0.0274889392175543 w_k1_DIS_s101
                                 + 0.0729531309430129 w_k1_DIS_s102
                                 + 0.0467824258218993 w_k1_DIS_s103
                                 + 0.133336435324359 w_k1_DIS_s104
                                 + 0.033005861112678 w_k1_DIS_s105
                                 + 0.0591521070332928 w_k1_DIS_s106
                                 + 0.117297974107325 w_k1_DIS_s107
                                 + 0.0188785535945136 w_k1_DIS_s110
                                 + 0.0164954363095887 w_k1_DIS_s111
                                 + 0.0122482120754785 w_k1_DIS_s112
                                 + 0.0140026899294419 w_k1_DIS_s113
                                 + 0.0179206379654439 w_k1_DIS_s114
                                 + 0.013991087840526 w_k1_DIS_s115
                                 + 0.0074515221356711 w_k1_DIS_s116
                                 + 0.0106555447038981 w_k1_DIS_s117
                                 + 0.0264323944363711 w_k1_FDX_s000
                                 + 0.0218825886707597 w_k1_FDX_s001
                                 + 0.0179802488124678 w_k1_FDX_s002
                                 + 0.0727609996416533 w_k1_FDX_s003
                                 + 0.0352813139080239 w_k1_FDX_s004
                                 + 0.0330122984076787 w_k1_FDX_s005
                                 + 0.0292514523172483 w_k1_FDX_s006
                                 + 0.0165720605617911 w_k1_FDX_s007
                                 + 0.0160150868863327 w_k1_FDX_s010
                                 + 0.0161121894730719 w_k1_FDX_s011
                                 + 0.0520638410123182 w_k1_FDX_s012
                                 + 0.0109980304608179 w_k1_FDX_s013
                                 + 0.0218109253453019 w_k1_FDX_s014
                                 + 0.0135295309020685 w_k1_FDX_s015
                                 + 0.0101856597804134 w_k1_FDX_s016
                                 + 0.0218908555052763 w_k1_FDX_s017
                                 + 0.0240536744973765 w_k1_FDX_s100
                                 + 0.0251729483467928 w_k1_FDX_s101
                                 + 0.0730816844723686 w_k1_FDX_s102
                                 + 0.0459794230882427 w_k1_FDX_s103
                                 + 0.122655996774292 w_k1_FDX_s104
                                 + 0.0312733557401424 w_k1_FDX_s105
                                 + 0.0605047903642451 w_k1_FDX_s106
                                 + 0.111336862799236 w_k1_FDX_s107
                                 + 0.0177830852431067 w_k1_FDX_s110
                                 + 0.0159025152635731 w_k1_FDX_s111
                                 + 0.0116989303462834 w_k1_FDX_s112
                                 + 0.0135519982991711 w_k1_FDX_s113
                                 + 0.0191528857519964 w_k1_FDX_s114
                                 + 0.0130558470861284 w_k1_FDX_s115
                                 + 0.00696534517491726 w_k1_FDX_s116
                                 + 0.0109864561461481 w_k1_FDX_s117
                                 + 0.0286183743179376 w_k1_IBM_s000
                                 + 0.022739172945846 w_k1_IBM_s001
                                 + 0.0179470570104989 w_k1_IBM_s002
                                 + 0.0713963896072764 w_k1_IBM_s003
                                 + 0.0337861943585508 w_k1_IBM_s004
                                 + 0.0328403844060624 w_k1_IBM_s005
                                 + 0.0305927964569402 w_k1_IBM_s006
                                 + 0.0179991576677141 w_k1_IBM_s007
                                 + 0.0166370281714108 w_k1_IBM_s010
                                 + 0.0163420242230369 w_k1_IBM_s011
                                 + 0.0526891243231501 w_k1_IBM_s012
                                 + 0.0115381153538908 w_k1_IBM_s013
                                 + 0.0215972016142795 w_k1_IBM_s014
                                 + 0.0140554607338198 w_k1_IBM_s015
                                 + 0.0107133289928624 w_k1_IBM_s016
                                 + 0.0219050205518833 w_k1_IBM_s017
                                 + 0.0251209753061034 w_k1_IBM_s100
                                 + 0.0263540135252148 w_k1_IBM_s101
                                 + 0.0758123720973048 w_k1_IBM_s102
                                 + 0.0437562912654634 w_k1_IBM_s103
                                 + 0.129029488658911 w_k1_IBM_s104
                                 + 0.0313781007699774 w_k1_IBM_s105
                                 + 0.0573634804435562 w_k1_IBM_s106
                                 + 0.116353241654139 w_k1_IBM_s107
                                 + 0.0183573377954582 w_k1_IBM_s110
                                 + 0.0164106840904083 w_k1_IBM_s111
                                 + 0.0118779220579574 w_k1_IBM_s112
                                 + 0.0133263717721228 w_k1_IBM_s113
                                 + 0.0190362965370546 w_k1_IBM_s114
                                 + 0.0133718232652627 w_k1_IBM_s115
                                 + 0.00712177221407796 w_k1_IBM_s116
                                 + 0.0112419869308678 w_k1_IBM_s117
                                 + 0.0235669085214703 w_k1_INTC_s000
                                 + 0.0201588275674839 w_k1_INTC_s001
                                 + 0.0169206582843981 w_k1_INTC_s002
                                 + 0.0718759621227838 w_k1_INTC_s003
                                 + 0.033847004750252 w_k1_INTC_s004
                                 + 0.0322001644200342 w_k1_INTC_s005
                                 + 0.0316060267291939 w_k1_INTC_s006
                                 + 0.0172761934833907 w_k1_INTC_s007
                                 + 0.0169870964591843 w_k1_INTC_s010
                                 + 0.0188402511515902 w_k1_INTC_s011
                                 + 0.0593238466914235 w_k1_INTC_s012
                                 + 0.010539127336093 w_k1_INTC_s013
                                 + 0.0221362395433602 w_k1_INTC_s014
                                 + 0.0129261905035078 w_k1_INTC_s015
                                 + 0.00944862655840542 w_k1_INTC_s016
                                 + 0.026271049158705 w_k1_INTC_s017
                                 + 0.0226109333255333 w_k1_INTC_s100
                                 + 0.0244475113766335 w_k1_INTC_s101
                                 + 0.0708533754882376 w_k1_INTC_s102
                                 + 0.0419299096188996 w_k1_INTC_s103
                                 + 0.133225043439311 w_k1_INTC_s104
                                 + 0.0298525511053798 w_k1_INTC_s105
                                 + 0.065209910824215 w_k1_INTC_s106
                                 + 0.111305120986349 w_k1_INTC_s107
                                 + 0.0177599338007609 w_k1_INTC_s110
                                 + 0.0140455351469899 w_k1_INTC_s111
                                 + 0.0096779742937856 w_k1_INTC_s112
                                 + 0.0148370981322523 w_k1_INTC_s113
                                 + 0.017532709332475 w_k1_INTC_s114
                                 + 0.0122032557686532 w_k1_INTC_s115
                                 + 0.0068736709859884 w_k1_INTC_s116
                                 + 0.0118916596078109 w_k1_INTC_s117
                                 + 0.0252638829909468 w_k1_KO_s000
                                 + 0.0219273499764399 w_k1_KO_s001
                                 + 0.0190998390636975 w_k1_KO_s002
                                 + 0.07156274444023 w_k1_KO_s003
                                 + 0.0353768970590412 w_k1_KO_s004
                                 + 0.0320127484259495 w_k1_KO_s005
                                 + 0.0305189932894064 w_k1_KO_s006
                                 + 0.0177208851790545 w_k1_KO_s007
                                 + 0.0164745985947708 w_k1_KO_s010
                                 + 0.0166586626846206 w_k1_KO_s011
                                 + 0.0524849794993035 w_k1_KO_s012
                                 + 0.00977166682968095 w_k1_KO_s013
                                 + 0.0201163874947908 w_k1_KO_s014
                                 + 0.01431561229503 w_k1_KO_s015
                                 + 0.0107666221660186 w_k1_KO_s016
                                 + 0.0220046230847279 w_k1_KO_s017
                                 + 0.0253114838856959 w_k1_KO_s100
                                 + 0.026537539455898 w_k1_KO_s101
                                 + 0.0722173671559656 w_k1_KO_s102
                                 + 0.0418566287329556 w_k1_KO_s103
                                 + 0.12410328862732 w_k1_KO_s104
                                 + 0.0300281245938736 w_k1_KO_s105
                                 + 0.0552010982717217 w_k1_KO_s106
                                 + 0.11293814272142 w_k1_KO_s107
                                 + 0.0176244616004284 w_k1_KO_s110
                                 + 0.0140196345878362 w_k1_KO_s111
                                 + 0.0124699672658425 w_k1_KO_s112
                                 + 0.0147707190387915 w_k1_KO_s113
                                 + 0.016670350211151 w_k1_KO_s114
                                 + 0.0153834301047249 w_k1_KO_s115
                                 + 0.00729302112954592 w_k1_KO_s116
                                 + 0.0105813408934566 w_k1_KO_s117
                                 + 0.0240466782753696 w_k1_MAT_s000
                                 + 0.0264306581525747 w_k1_MAT_s001
                                 + 0.0235022951334097 w_k1_MAT_s002
                                 + 0.0597281146487589 w_k1_MAT_s003
                                 + 0.030086725135031 w_k1_MAT_s004
                                 + 0.0350382882875093 w_k1_MAT_s005
                                 + 0.0300827668578583 w_k1_MAT_s006
                                 + 0.0173022761259732 w_k1_MAT_s007
                                 + 0.0135403275453128 w_k1_MAT_s010
                                 + 0.0159113744253162 w_k1_MAT_s011
                                 + 0.0568727593353075 w_k1_MAT_s012
                                 + 0.0119468650121813 w_k1_MAT_s013
                                 + 0.019056793867463 w_k1_MAT_s014
                                 + 0.0158998981954666 w_k1_MAT_s015
                                 + 0.011450472974439 w_k1_MAT_s016
                                 + 0.0214601885164675 w_k1_MAT_s017
                                 + 0.02910209512045 w_k1_MAT_s100
                                 + 0.0310667263705147 w_k1_MAT_s101
                                 + 0.0711955798877498 w_k1_MAT_s102
                                 + 0.0436893510559508 w_k1_MAT_s103
                                 + 0.131445053256564 w_k1_MAT_s104
                                 + 0.0261727729636964 w_k1_MAT_s105
                                 + 0.0588275093100059 w_k1_MAT_s106
                                 + 0.127147243577867 w_k1_MAT_s107
                                 + 0.0170324985423365 w_k1_MAT_s110
                                 + 0.0166870829477299 w_k1_MAT_s111
                                 + 0.0114066268652046 w_k1_MAT_s112
                                 + 0.0127779118405399 w_k1_MAT_s113
                                 + 0.0156189413243601 w_k1_MAT_s114
                                 + 0.0146300528915722 w_k1_MAT_s115
                                 + 0.00661910533134713 w_k1_MAT_s116
                                 + 0.0124270068565942 w_k1_MAT_s117
                                 + 0.0273448851041441 w_k1_MSFT_s000
                                 + 0.0205492669834498 w_k1_MSFT_s001
                                 + 0.0187435595866457 w_k1_MSFT_s002
                                 + 0.0704968752512826 w_k1_MSFT_s003
                                 + 0.0336325665692467 w_k1_MSFT_s004
                                 + 0.03179722090982 w_k1_MSFT_s005
                                 + 0.0310140442025901 w_k1_MSFT_s006
                                 + 0.015839336555113 w_k1_MSFT_s007
                                 + 0.015481011060638 w_k1_MSFT_s010
                                 + 0.0154518518855008 w_k1_MSFT_s011
                                 + 0.0486495714895844 w_k1_MSFT_s012
                                 + 0.0114269667853752 w_k1_MSFT_s013
                                 + 0.0222772194310642 w_k1_MSFT_s014
                                 + 0.0156679923728431 w_k1_MSFT_s015
                                 + 0.00947728173314764 w_k1_MSFT_s016
                                 + 0.023883998921782 w_k1_MSFT_s017
                                 + 0.0256440264070339 w_k1_MSFT_s100
                                 + 0.0261101721602352 w_k1_MSFT_s101
                                 + 0.0797439651635673 w_k1_MSFT_s102
                                 + 0.0419429206346822 w_k1_MSFT_s103
                                 + 0.122337548224315 w_k1_MSFT_s104
                                 + 0.0310008872603019 w_k1_MSFT_s105
                                 + 0.0606932473584059 w_k1_MSFT_s106
                                 + 0.119546088451361 w_k1_MSFT_s107
                                 + 0.0180104219392997 w_k1_MSFT_s110
                                 + 0.0155882170193452 w_k1_MSFT_s111
                                 + 0.0119636951768318 w_k1_MSFT_s112
                                 + 0.0150946428651787 w_k1_MSFT_s113
                                 + 0.017113782547952 w_k1_MSFT_s114
                                 + 0.0151227556175636 w_k1_MSFT_s115
                                 + 0.00639183209645643 w_k1_MSFT_s116
                                 + 0.0117169895902484 w_k1_MSFT_s117
                                 + 0.0257945158864433 w_k1_PG_s000
                                 + 0.0227693260762085 w_k1_PG_s001
                                 + 0.0190128754532 w_k1_PG_s002
                                 + 0.0672827917830033 w_k1_PG_s003
                                 + 0.0352893705092332 w_k1_PG_s004
                                 + 0.0325588462808055 w_k1_PG_s005
                                 + 0.03008158572892 w_k1_PG_s006
                                 + 0.0181239977807696 w_k1_PG_s007
                                 + 0.0170842278241446 w_k1_PG_s010
                                 + 0.0161787348720052 w_k1_PG_s011
                                 + 0.0492303113058103 w_k1_PG_s012
                                 + 0.0103441884690195 w_k1_PG_s013
                                 + 0.0211139042722103 w_k1_PG_s014
                                 + 0.0136717693006019 w_k1_PG_s015
                                 + 0.0088717700473128 w_k1_PG_s016
                                 + 0.0227502298848365 w_k1_PG_s017
                                 + 0.024315141407067 w_k1_PG_s100
                                 + 0.0247246102197458 w_k1_PG_s101
                                 + 0.0776147351579388 w_k1_PG_s102
                                 + 0.044821499014879 w_k1_PG_s103
                                 + 0.120148670083333 w_k1_PG_s104
                                 + 0.0308895002674318 w_k1_PG_s105
                                 + 0.0583692857727105 w_k1_PG_s106
                                 + 0.115031564915379 w_k1_PG_s107
                                 + 0.0176673169550882 w_k1_PG_s110
                                 + 0.0162942131569178 w_k1_PG_s111
                                 + 0.0122180264890511 w_k1_PG_s112
                                 + 0.0131944601432467 w_k1_PG_s113
                                 + 0.0181738575280176 w_k1_PG_s114
                                 + 0.0143443357912324 w_k1_PG_s115
                                 + 0.00701771325361784 w_k1_PG_s116
                                 + 0.0103786537047883 w_k1_PG_s117 >= 1.14
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
