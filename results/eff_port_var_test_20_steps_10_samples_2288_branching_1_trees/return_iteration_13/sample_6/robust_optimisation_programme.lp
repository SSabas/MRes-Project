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
 w_k1_ADM_s0_return_constraint:  1.04653 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.958819 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08832 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.972117 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00374 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15076 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.07097 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.825096 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.43104 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.686658 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.05861 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.750607 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.927492 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.777566 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.50251 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.0097 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.672135 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.09028 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.20788 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.932036 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.756991 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.865871 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.689381 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.5375 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.975615 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.819917 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.1279 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.44396 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.975473 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.1746 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.996026 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.04346 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.833306 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.00752 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.961492 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.700057 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.515 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.797112 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.0428 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.980322 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15563 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.79808 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.01932 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23609 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.996057 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.847536 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.83103 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.608307 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.30427 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.74539 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.13078 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.850747 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.61511 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.891771 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.701811 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.15919 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.33912 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.08459 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.678028 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.92116 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.495386 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.56577 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.12658 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.738922 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.01287 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.3256 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.923883 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.57354 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.29294 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.03138 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.878731 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.0449 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.16905 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.639999 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.79686 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.584001 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.08347 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.989848 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.26059 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.852939 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.978735 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.28579 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.940097 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.971987 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.81629 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.454601 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.24631 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.581322 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.903555 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.647046 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 2.09018 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.86754 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.543974 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.19306 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.32804 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.03443 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.539997 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.861275 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.476504 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.80805 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.903312 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.608572 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.04335 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.31772 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.844925 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.82506 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.13227 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.17184 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.821814 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.07542 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.20499 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.619189 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 2.11245 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.437912 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.084 w_0_FDX - w_k1_FDX_s0 + 0.99 b_k1_FDX_s0
                                 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.967429 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1133 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.900271 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.00075 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.02696 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.10048 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.868828 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.54931 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.59816 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.12621 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.788209 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.00995 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.795721 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.53972 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.995281 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.630243 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.11401 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.29282 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.904893 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.731395 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.899175 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.547213 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.48945 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.903204 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.743025 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.05218 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.37747 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.834897 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.33973 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.18997 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.07026 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.916946 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.942344 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.17675 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.699573 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.57892 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.673434 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.903241 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.16238 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.23124 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.760294 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.893223 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.51129 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.756964 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.12847 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.36036 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.761757 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.3607 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.650496 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.992704 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.980173 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.38754 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.743524 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.04535 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.08121 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.23863 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.5797 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.690738 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.11245 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.878952 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.24288 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.15687 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.680144 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.919945 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.816944 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.43825 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.65555 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.14926 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27591 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.740457 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.6435 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.902738 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.967687 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.46711 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.657686 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02935 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.970442 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.06795 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.929738 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.976666 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.08125 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.0253 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.03386 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.44712 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.740192 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.09539 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.81325 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.973344 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.01249 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.48446 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.98469 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.751527 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.0241 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.20076 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.0257 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.869626 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.984098 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.701596 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.44777 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.971629 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.752427 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.04826 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.33089 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.974992 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.45585 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13669 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.04113 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.10666 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.18066 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.1554 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.838332 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.35607 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.801705 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.971513 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.06038 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.22404 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.830178 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.934848 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.2892 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.839994 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.12793 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.3648 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.617756 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.13879 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.623285 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.990411 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.76581 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.62397 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.973029 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.821733 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.16344 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.32372 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.25089 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.703169 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11726 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.747641 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.43606 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.00145 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.719287 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.962126 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00608 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.18789 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.52204 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.03177 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.16717 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.754204 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.28438 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.992632 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.863119 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.70054 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.585972 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.945643 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.20614 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.41124 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.580885 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.03319 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.47055 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.622719 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.969929 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.129 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.57329 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.35272 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.596071 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.990096 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.923187 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.6214 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.991652 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23871 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.35543 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.51323 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.9077 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.802717 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.15092 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.564301 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.03183 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.951765 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.579345 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.653088 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.69424 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.28962 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.31446 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.13809 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.20897 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.799882 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.33879 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.901235 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.975364 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.87355 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.610101 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.03516 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.01178 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.03547 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.925662 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.960401 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03565 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.17681 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13094 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.50328 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.631665 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.17372 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.772829 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.06702 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.90206 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.42597 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.980302 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.675351 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.0965 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.22232 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.02157 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.735205 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.961131 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.645391 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.47256 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.877971 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.728781 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.01198 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.28785 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.914203 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.42125 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.08597 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.9799 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.954602 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.06152 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17177 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.792989 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.55358 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.72294 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.042 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.00474 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08584 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.876634 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.02186 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11067 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.949349 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.933942 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.20579 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.718255 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.07493 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.774427 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05333 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.851909 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.28962 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.928622 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.822168 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.06012 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.16309 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.07627 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.772272 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.03469 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.685431 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.2316 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.882631 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.749306 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00814 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.07759 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.976629 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.26107 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08784 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.028 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.911601 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.15785 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.976114 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.834312 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.4329 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.714249 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.540998419669551 z_k1_s0000
                                 - 0.428000164258165 z_k1_s0001
                                 - 0.997002246084351 z_k1_s0002
                                 - 0.295000272220994 z_k1_s0003
                                 - 0.723002394879924 z_k1_s0004
                                 - 0.778997968788598 z_k1_s0005
                                 - 0.814998270873977 z_k1_s0006
                                 - 0.31800022041989 z_k1_s0007
                                 - 0.533000282311798 z_k1_s0010
                                 - 0.366002129298363 z_k1_s0011
                                 - 0.295000880641406 z_k1_s0012
                                 - 0.515002189548701 z_k1_s0013
                                 - 0.42100146142238 z_k1_s0014
                                 - 0.193000521052358 z_k1_s0015
                                 - 0.552000111900265 z_k1_s0016
                                 - 0.442001447652231 z_k1_s0017
                                 - 0.547000368712526 z_k1_s0020
                                 - 0.739999050846091 z_k1_s0021
                                 - 0.23399966588383 z_k1_s0022
                                 - 0.73399907553109 z_k1_s0023
                                 - 0.287999443718843 z_k1_s0024
                                 - 0.15699977314172 z_k1_s0025
                                 - 0.247999503519037 z_k1_s0026
                                 - 0.194999512036933 z_k1_s0027
                                 - 0.165000298296726 z_k1_s0030
                                 - 0.197000268789917 z_k1_s0031
                                 - 0.332999568910226 z_k1_s0032
                                 - 0.235000272919381 z_k1_s0033
                                 - 0.233001097308381 z_k1_s0034
                                 - 0.519000193834185 z_k1_s0035
                                 - 0.205000313638282 z_k1_s0036
                                 - 0.202000296820145 z_k1_s0037
                                 - 0.227999684246546 z_k1_s0040
                                 - 0.865001447069184 z_k1_s0041
                                 - 0.595000915856902 z_k1_s0042
                                 - 0.596000663046784 z_k1_s0043
                                 - 0.98799786740232 z_k1_s0044
                                 - 0.554997270272788 z_k1_s0045
                                 - 0.497997922460686 z_k1_s0046
                                 - 0.260999596305949 z_k1_s0047
                                 - 0.178999763586154 z_k1_s0050
                                 - 0.103000201672977 z_k1_s0051
                                 - 0.288000424095109 z_k1_s0052
                                 - 0.153000225300527 z_k1_s0053
                                 - 0.214000815126963 z_k1_s0054
                                 - 0.174000256224128 z_k1_s0055
                                 - 0.289000925568398 z_k1_s0056
                                 - 0.10000019725532 z_k1_s0057
                                 - 0.375000794288993 z_k1_s0060
                                 - 0.258001222087776 z_k1_s0061
                                 - 0.120000175612368 z_k1_s0062
                                 - 0.33200091794149 z_k1_s0063
                                 - 0.138000260874305 z_k1_s0064
                                 - 0.135000246663982 z_k1_s0065
                                 - 0.485000660666583 z_k1_s0066
                                 - 0.1210002458159 z_k1_s0067
                                 - 0.325999856810129 z_k1_s0070
                                 - 0.47199892845383 z_k1_s0071
                                 - 0.453001369262109 z_k1_s0072
                                 - 0.277999858472439 z_k1_s0073
                                 - 0.463000709415718 z_k1_s0074
                                 - 0.299999989610562 z_k1_s0075
                                 - 1.18099844225007 z_k1_s0076
                                 - 0.484000510838435 z_k1_s0077
                                 - 0.278998888371581 z_k1_s0100
                                 - 0.287000615177672 z_k1_s0101
                                 - 0.300998919089035 z_k1_s0102
                                 - 0.213999955669837 z_k1_s0103
                                 - 0.394000121212661 z_k1_s0104
                                 - 0.328000029060297 z_k1_s0105
                                 - 0.275000383968183 z_k1_s0106
                                 - 0.280999320073103 z_k1_s0107
                                 - 0.353000196592152 z_k1_s0110
                                 - 0.749999253070566 z_k1_s0111
                                 - 0.405000183958398 z_k1_s0112
                                 - 0.575998956517961 z_k1_s0113
                                 - 0.395000137984096 z_k1_s0114
                                 - 0.340000304625643 z_k1_s0115
                                 - 0.5030017252071 z_k1_s0116
                                 - 0.873000909255023 z_k1_s0117
                                 - 0.35099881661157 z_k1_s0120
                                 - 0.31999908151174 z_k1_s0121
                                 - 0.314999258753508 z_k1_s0122
                                 - 0.327999214924307 z_k1_s0123
                                 - 0.155999682549285 z_k1_s0124
                                 - 0.36300005922891 z_k1_s0125
                                 - 0.120999880743171 z_k1_s0126
                                 - 0.130999734759333 z_k1_s0127
                                 - 0.538999772550069 z_k1_s0130
                                 - 0.334000417630382 z_k1_s0131
                                 - 0.238000045827426 z_k1_s0132
                                 - 0.938001505556261 z_k1_s0133
                                 - 0.381001287892823 z_k1_s0134
                                 - 0.326000261513364 z_k1_s0135
                                 - 0.404002065204526 z_k1_s0136
                                 - 0.593999672728583 z_k1_s0137
                                 - 0.376999426320201 z_k1_s0140
                                 - 0.525001596575772 z_k1_s0141
                                 - 0.155000417446063 z_k1_s0142
                                 - 0.444000848754421 z_k1_s0143
                                 - 0.159000346028364 z_k1_s0144
                                 - 0.344001218693833 z_k1_s0145
                                 - 0.613000591587615 z_k1_s0146
                                 - 0.214000567212609 z_k1_s0147
                                 - 0.104999879761293 z_k1_s0150
                                 - 0.149000174083876 z_k1_s0151
                                 - 0.298999806457291 z_k1_s0152
                                 - 0.154999665819163 z_k1_s0153
                                 - 0.316000194372639 z_k1_s0154
                                 - 0.119999980098537 z_k1_s0155
                                 - 0.13099991657928 z_k1_s0156
                                 - 0.0959999566588493 z_k1_s0157
                                 - 0.210999186831058 z_k1_s0160
                                 - 0.201999318829344 z_k1_s0161
                                 - 0.42499794406423 z_k1_s0162
                                 - 0.269998932039852 z_k1_s0163
                                 - 0.747998348105179 z_k1_s0164
                                 - 0.344999760046424 z_k1_s0165
                                 - 0.444998936062898 z_k1_s0166
                                 - 0.245998898436824 z_k1_s0167
                                 - 0.703999306574328 z_k1_s0170
                                 - 0.406000012978033 z_k1_s0171
                                 - 0.211999423805581 z_k1_s0172
                                 - 0.64399725914367 z_k1_s0173
                                 - 0.200999443909087 z_k1_s0174
                                 - 0.190999597002885 z_k1_s0175
                                 - 0.37900057463096 z_k1_s0176
                                 - 0.228999608445141 z_k1_s0177
                                 - 0.528000622077064 z_k1_s1000
                                 - 0.263000973354796 z_k1_s1001
                                 - 0.243000044677468 z_k1_s1002
                                 - 0.124000087798391 z_k1_s1003
                                 - 0.139000075556256 z_k1_s1004
                                 - 0.228999192103301 z_k1_s1005
                                 - 0.159999936917246 z_k1_s1006
                                 - 0.338999237327674 z_k1_s1007
                                 - 0.816000963298392 z_k1_s1010
                                 - 0.426001307986139 z_k1_s1011
                                 - 0.416998245819384 z_k1_s1012
                                 - 0.302999840880988 z_k1_s1013
                                 - 0.506001056152872 z_k1_s1014
                                 - 0.287999799799782 z_k1_s1015
                                 - 0.367998841566973 z_k1_s1016
                                 - 0.40799871565034 z_k1_s1017
                                 - 0.469998661124733 z_k1_s1020
                                 - 1.30499378585803 z_k1_s1021
                                 - 0.647996290255766 z_k1_s1022
                                 - 0.580998243516877 z_k1_s1023
                                 - 1.14299622153895 z_k1_s1024
                                 - 0.388998672369303 z_k1_s1025
                                 - 1.16699541380996 z_k1_s1026
                                 - 0.329999149791528 z_k1_s1027
                                 - 0.658000834554048 z_k1_s1030
                                 - 0.552000150606712 z_k1_s1031
                                 - 0.651000034330655 z_k1_s1032
                                 - 0.807000650041593 z_k1_s1033
                                 - 2.24000183933232 z_k1_s1034
                                 - 2.5389976837084 z_k1_s1035
                                 - 0.623000599839356 z_k1_s1036
                                 - 1.34599918000478 z_k1_s1037
                                 - 0.788000994630942 z_k1_s1040
                                 - 1.84399823072065 z_k1_s1041
                                 - 0.963001300870141 z_k1_s1042
                                 - 1.17799975550474 z_k1_s1043
                                 - 1.00499791548371 z_k1_s1044
                                 - 0.772000774471716 z_k1_s1045
                                 - 2.19100114387179 z_k1_s1046
                                 - 0.867000940715853 z_k1_s1047
                                 - 0.152999966405295 z_k1_s1050
                                 - 0.586999123330551 z_k1_s1051
                                 - 0.287999064857531 z_k1_s1052
                                 - 0.200999964948127 z_k1_s1053
                                 - 0.415999678571392 z_k1_s1054
                                 - 0.250000474910304 z_k1_s1055
                                 - 0.278000898660043 z_k1_s1056
                                 - 0.142999947408882 z_k1_s1057
                                 - 0.349998341780729 z_k1_s1060
                                 - 0.484000288082429 z_k1_s1061
                                 - 0.259999843664467 z_k1_s1062
                                 - 0.799997676354292 z_k1_s1063
                                 - 0.449999334573563 z_k1_s1064
                                 - 0.430001189212672 z_k1_s1065
                                 - 0.210999596735707 z_k1_s1066
                                 - 0.436999171290533 z_k1_s1067
                                 - 0.88800183804593 z_k1_s1070
                                 - 0.511998821284048 z_k1_s1071
                                 - 0.684001705252072 z_k1_s1072
                                 - 0.613002856036579 z_k1_s1073
                                 - 0.548999790370768 z_k1_s1074
                                 - 0.433999904511798 z_k1_s1075
                                 - 0.3600005367426 z_k1_s1076
                                 - 0.24400033523216 z_k1_s1077
                                 - 0.294000947926184 z_k1_s1100
                                 - 0.245000789938486 z_k1_s1101
                                 - 0.189000790667798 z_k1_s1102
                                 - 0.139000655353959 z_k1_s1103
                                 - 0.0750004567721101 z_k1_s1104
                                 - 0.0610002666034241 z_k1_s1105
                                 - 0.196000631950789 z_k1_s1106
                                 - 0.0700003163400503 z_k1_s1107
                                 - 0.235999032321335 z_k1_s1110
                                 - 0.074999846064541 z_k1_s1111
                                 - 0.205999038975622 z_k1_s1112
                                 - 0.282999479562087 z_k1_s1113
                                 - 0.128999696787083 z_k1_s1114
                                 - 0.191999957412561 z_k1_s1115
                                 - 0.0629998487262559 z_k1_s1116
                                 - 0.18899995807799 z_k1_s1117
                                 - 0.309002064787447 z_k1_s1120
                                 - 0.658003156158215 z_k1_s1121
                                 - 0.139000496789706 z_k1_s1122
                                 - 0.22600066161588 z_k1_s1123
                                 - 0.185000518364251 z_k1_s1124
                                 - 0.144000375112622 z_k1_s1125
                                 - 0.438001663927598 z_k1_s1126
                                 - 0.183000567035085 z_k1_s1127
                                 - 0.100000309141459 z_k1_s1130
                                 - 0.0860002913417129 z_k1_s1131
                                 - 0.0410000885279115 z_k1_s1132
                                 - 0.0430000819707118 z_k1_s1133
                                 - 0.0749999133553711 z_k1_s1134
                                 - 0.092000207969969 z_k1_s1135
                                 - 0.139000085725847 z_k1_s1136
                                 - 0.0610001503562034 z_k1_s1137
                                 - 0.599999992321819 z_k1_s1140
                                 - 0.227999949002287 z_k1_s1141
                                 - 0.253999451842852 z_k1_s1142
                                 - 0.272000605532615 z_k1_s1143
                                 - 0.305000316630288 z_k1_s1144
                                 - 0.163999933794624 z_k1_s1145
                                 - 0.192000045622987 z_k1_s1146
                                 - 0.389000892515397 z_k1_s1147
                                 - 0.135000134548859 z_k1_s1150
                                 - 0.27699991444514 z_k1_s1151
                                 - 0.0830001216515527 z_k1_s1152
                                 - 0.271000455518973 z_k1_s1153
                                 - 0.149000393344497 z_k1_s1154
                                 - 0.323999825036853 z_k1_s1155
                                 - 0.075000013282649 z_k1_s1156
                                 - 0.069000001156028 z_k1_s1157
                                 - 0.156000379612797 z_k1_s1160
                                 - 0.0560001067122218 z_k1_s1161
                                 - 0.149000772374653 z_k1_s1162
                                 - 0.106999984938947 z_k1_s1163
                                 - 0.0980002653465592 z_k1_s1164
                                 - 0.063000133151278 z_k1_s1165
                                 - 0.242000494835032 z_k1_s1166
                                 - 0.177000249329509 z_k1_s1167
                                 - 0.0820001980407347 z_k1_s1170
                                 - 0.13100013230129 z_k1_s1171
                                 - 0.120000186108183 z_k1_s1172
                                 - 0.0970002213042576 z_k1_s1173
                                 - 0.352000404183718 z_k1_s1174
                                 - 0.114999824019627 z_k1_s1175
                                 - 0.0900001395811373 z_k1_s1176
                                 - 0.0760001887353256 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.04341 w_k1_ADM_s000
                                 + 1.47403 w_k1_AXP_s000
                                 + 1.15658 w_k1_DIS_s000
                                 + 1.08729 w_k1_FDX_s000
                                 + 1.17545 w_k1_IBM_s000
                                 + 0.990827 w_k1_INTC_s000
                                 + 1.16199 w_k1_KO_s000 + 1.80636 w_k1_MAT_s000
                                 + 1.01107 w_k1_MSFT_s000
                                 + 1.11156 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.39773 w_k1_ADM_s000
                                 + 1.55687 w_k1_AXP_s000
                                 + 1.93179 w_k1_DIS_s000
                                 + 1.39051 w_k1_FDX_s000 + 1.201 w_k1_IBM_s000
                                 + 1.23223 w_k1_INTC_s000
                                 + 1.53525 w_k1_KO_s000 + 1.83324 w_k1_MAT_s000
                                 + 1.36151 w_k1_MSFT_s000
                                 + 1.17482 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.886646 w_k1_ADM_s000
                                 + 1.01498 w_k1_AXP_s000
                                 + 0.903482 w_k1_DIS_s000
                                 + 0.978719 w_k1_FDX_s000
                                 + 0.837923 w_k1_IBM_s000
                                 + 1.09825 w_k1_INTC_s000
                                 + 0.84326 w_k1_KO_s000
                                 + 0.860084 w_k1_MAT_s000
                                 + 0.972272 w_k1_MSFT_s000
                                 + 0.883902 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.857297 w_k1_ADM_s000
                                 + 0.8921 w_k1_AXP_s000
                                 + 0.605431 w_k1_DIS_s000
                                 + 0.837081 w_k1_FDX_s000
                                 + 0.687925 w_k1_IBM_s000
                                 + 1.06244 w_k1_INTC_s000
                                 + 0.749974 w_k1_KO_s000
                                 + 0.756857 w_k1_MAT_s000
                                 + 0.864727 w_k1_MSFT_s000
                                 + 0.873977 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.16155 w_k1_ADM_s000
                                 + 1.26799 w_k1_AXP_s000
                                 + 1.38418 w_k1_DIS_s000
                                 + 1.25782 w_k1_FDX_s000
                                 + 0.917319 w_k1_IBM_s000
                                 + 1.1524 w_k1_INTC_s000 + 1.07666 w_k1_KO_s000
                                 + 1.00587 w_k1_MAT_s000
                                 + 1.07971 w_k1_MSFT_s000
                                 + 1.05578 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.616333 w_k1_ADM_s000
                                 + 0.721182 w_k1_AXP_s000
                                 + 0.714773 w_k1_DIS_s000
                                 + 0.741772 w_k1_FDX_s000
                                 + 0.830929 w_k1_IBM_s000
                                 + 0.871118 w_k1_INTC_s000
                                 + 0.790092 w_k1_KO_s000
                                 + 0.97508 w_k1_MAT_s000
                                 + 0.766623 w_k1_MSFT_s000
                                 + 0.793525 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.01611 w_k1_ADM_s000
                                 + 0.878098 w_k1_AXP_s000
                                 + 1.08879 w_k1_DIS_s000
                                 + 0.922826 w_k1_FDX_s000
                                 + 1.14699 w_k1_IBM_s000
                                 + 0.947548 w_k1_INTC_s000
                                 + 1.23136 w_k1_KO_s000 + 1.12336 w_k1_MAT_s000
                                 + 0.930618 w_k1_MSFT_s000
                                 + 1.04846 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.16182 w_k1_ADM_s000
                                 + 1.18405 w_k1_AXP_s000
                                 + 1.53708 w_k1_DIS_s000
                                 + 1.18915 w_k1_FDX_s000 + 1.4787 w_k1_IBM_s000
                                 + 1.06799 w_k1_INTC_s000
                                 + 1.40816 w_k1_KO_s000 + 1.46811 w_k1_MAT_s000
                                 + 1.09751 w_k1_MSFT_s000
                                 + 1.16682 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.14849 w_k1_ADM_s001
                                 + 1.10784 w_k1_AXP_s001
                                 + 1.31405 w_k1_DIS_s001
                                 + 1.12182 w_k1_FDX_s001
                                 + 1.43165 w_k1_IBM_s001
                                 + 0.990731 w_k1_INTC_s001
                                 + 1.10172 w_k1_KO_s001 + 1.36081 w_k1_MAT_s001
                                 + 1.06376 w_k1_MSFT_s001
                                 + 1.17338 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.23452 w_k1_ADM_s001
                                 + 1.09122 w_k1_AXP_s001
                                 + 0.824818 w_k1_DIS_s001
                                 + 0.825211 w_k1_FDX_s001
                                 + 1.40246 w_k1_IBM_s001
                                 + 0.717853 w_k1_INTC_s001
                                 + 0.994707 w_k1_KO_s001
                                 + 1.43681 w_k1_MAT_s001
                                 + 0.866662 w_k1_MSFT_s001
                                 + 1.09161 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.03049 w_k1_ADM_s001
                                 + 0.830701 w_k1_AXP_s001
                                 + 0.918788 w_k1_DIS_s001
                                 + 1.01504 w_k1_FDX_s001
                                 + 0.701577 w_k1_IBM_s001
                                 + 0.814241 w_k1_INTC_s001
                                 + 0.820593 w_k1_KO_s001
                                 + 0.89088 w_k1_MAT_s001
                                 + 0.912281 w_k1_MSFT_s001
                                 + 1.06101 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.933975 w_k1_ADM_s001
                                 + 0.698341 w_k1_AXP_s001
                                 + 0.679681 w_k1_DIS_s001
                                 + 0.84937 w_k1_FDX_s001
                                 + 0.765421 w_k1_IBM_s001
                                 + 0.826909 w_k1_INTC_s001
                                 + 0.738568 w_k1_KO_s001
                                 + 0.561395 w_k1_MAT_s001
                                 + 0.746964 w_k1_MSFT_s001
                                 + 0.8902 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.00176 w_k1_ADM_s001
                                 + 0.953448 w_k1_AXP_s001
                                 + 0.813866 w_k1_DIS_s001
                                 + 0.995583 w_k1_FDX_s001
                                 + 0.965999 w_k1_IBM_s001
                                 + 0.989609 w_k1_INTC_s001
                                 + 0.876751 w_k1_KO_s001
                                 + 0.853289 w_k1_MAT_s001
                                 + 0.956034 w_k1_MSFT_s001
                                 + 0.913172 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.820341 w_k1_ADM_s001
                                 + 0.71516 w_k1_AXP_s001
                                 + 0.646477 w_k1_DIS_s001
                                 + 0.69507 w_k1_FDX_s001
                                 + 1.14005 w_k1_IBM_s001
                                 + 0.794148 w_k1_INTC_s001
                                 + 0.849526 w_k1_KO_s001
                                 + 0.864833 w_k1_MAT_s001
                                 + 0.695903 w_k1_MSFT_s001
                                 + 0.832509 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.34754 w_k1_ADM_s001
                                 + 1.29535 w_k1_AXP_s001
                                 + 1.57296 w_k1_DIS_s001
                                 + 1.28138 w_k1_FDX_s001
                                 + 1.05568 w_k1_IBM_s001
                                 + 1.07788 w_k1_INTC_s001
                                 + 1.18682 w_k1_KO_s001
                                 + 0.901642 w_k1_MAT_s001
                                 + 1.16914 w_k1_MSFT_s001
                                 + 1.10743 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.23113 w_k1_ADM_s001
                                 + 1.06714 w_k1_AXP_s001
                                 + 1.11844 w_k1_DIS_s001
                                 + 1.23182 w_k1_FDX_s001
                                 + 0.888714 w_k1_IBM_s001
                                 + 1.00643 w_k1_INTC_s001
                                 + 0.866042 w_k1_KO_s001
                                 + 0.848459 w_k1_MAT_s001
                                 + 1.07386 w_k1_MSFT_s001
                                 + 1.07567 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.998181 w_k1_ADM_s002
                                 + 0.850596 w_k1_AXP_s002
                                 + 0.934458 w_k1_DIS_s002
                                 + 0.916033 w_k1_FDX_s002
                                 + 0.746214 w_k1_IBM_s002
                                 + 1.01509 w_k1_INTC_s002
                                 + 0.949366 w_k1_KO_s002
                                 + 0.782279 w_k1_MAT_s002
                                 + 1.0546 w_k1_MSFT_s002
                                 + 0.957214 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.952662 w_k1_ADM_s002
                                 + 1.20165 w_k1_AXP_s002 + 1.6303 w_k1_DIS_s002
                                 + 1.31791 w_k1_FDX_s002
                                 + 1.26534 w_k1_IBM_s002
                                 + 1.19462 w_k1_INTC_s002
                                 + 1.21103 w_k1_KO_s002 + 1.19866 w_k1_MAT_s002
                                 + 1.29623 w_k1_MSFT_s002 + 1.3 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.36943 w_k1_ADM_s002
                                 + 1.25323 w_k1_AXP_s002
                                 + 1.02859 w_k1_DIS_s002 + 1.0109 w_k1_FDX_s002
                                 + 1.36659 w_k1_IBM_s002
                                 + 1.14074 w_k1_INTC_s002
                                 + 1.14714 w_k1_KO_s002 + 1.35086 w_k1_MAT_s002
                                 + 1.00283 w_k1_MSFT_s002
                                 + 1.11922 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.01722 w_k1_ADM_s002
                                 + 0.818364 w_k1_AXP_s002
                                 + 1.02039 w_k1_DIS_s002
                                 + 0.835796 w_k1_FDX_s002
                                 + 1.29624 w_k1_IBM_s002
                                 + 0.913524 w_k1_INTC_s002 + 1.268 w_k1_KO_s002
                                 + 1.14468 w_k1_MAT_s002
                                 + 0.981053 w_k1_MSFT_s002
                                 + 1.2001 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.872473 w_k1_ADM_s002
                                 + 0.657522 w_k1_AXP_s002
                                 + 0.732197 w_k1_DIS_s002
                                 + 0.858729 w_k1_FDX_s002
                                 + 0.53103 w_k1_IBM_s002
                                 + 0.96043 w_k1_INTC_s002
                                 + 0.723549 w_k1_KO_s002
                                 + 0.528676 w_k1_MAT_s002
                                 + 0.921779 w_k1_MSFT_s002
                                 + 0.892474 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.998123 w_k1_ADM_s002
                                 + 1.25282 w_k1_AXP_s002
                                 + 0.962108 w_k1_DIS_s002
                                 + 1.07437 w_k1_FDX_s002
                                 + 0.833388 w_k1_IBM_s002
                                 + 1.08043 w_k1_INTC_s002
                                 + 0.888664 w_k1_KO_s002
                                 + 1.01433 w_k1_MAT_s002
                                 + 0.952603 w_k1_MSFT_s002
                                 + 0.95518 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.694904 w_k1_ADM_s002
                                 + 0.624892 w_k1_AXP_s002
                                 + 0.666836 w_k1_DIS_s002
                                 + 0.674056 w_k1_FDX_s002
                                 + 0.84095 w_k1_IBM_s002
                                 + 0.740459 w_k1_INTC_s002
                                 + 0.792207 w_k1_KO_s002
                                 + 0.875629 w_k1_MAT_s002
                                 + 0.75202 w_k1_MSFT_s002
                                 + 0.789982 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.39373 w_k1_ADM_s002
                                 + 1.0968 w_k1_AXP_s002 + 1.15546 w_k1_DIS_s002
                                 + 1.28435 w_k1_FDX_s002 + 0.7081 w_k1_IBM_s002
                                 + 1.15887 w_k1_INTC_s002
                                 + 1.04666 w_k1_KO_s002
                                 + 0.948192 w_k1_MAT_s002
                                 + 1.11831 w_k1_MSFT_s002
                                 + 0.992852 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.725995 w_k1_ADM_s003
                                 + 0.643278 w_k1_AXP_s003
                                 + 0.679022 w_k1_DIS_s003
                                 + 0.759076 w_k1_FDX_s003
                                 + 0.936532 w_k1_IBM_s003
                                 + 0.847493 w_k1_INTC_s003
                                 + 0.909408 w_k1_KO_s003
                                 + 0.808339 w_k1_MAT_s003
                                 + 0.889663 w_k1_MSFT_s003
                                 + 0.949502 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.917972 w_k1_ADM_s003
                                 + 0.9739 w_k1_AXP_s003
                                 + 0.921284 w_k1_DIS_s003
                                 + 0.929872 w_k1_FDX_s003
                                 + 0.916231 w_k1_IBM_s003
                                 + 0.917845 w_k1_INTC_s003
                                 + 0.952414 w_k1_KO_s003
                                 + 0.99998 w_k1_MAT_s003
                                 + 0.960184 w_k1_MSFT_s003
                                 + 0.909921 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.952426 w_k1_ADM_s003
                                 + 0.967008 w_k1_AXP_s003
                                 + 0.886657 w_k1_DIS_s003
                                 + 0.982502 w_k1_FDX_s003
                                 + 0.748135 w_k1_IBM_s003
                                 + 0.993825 w_k1_INTC_s003
                                 + 0.880554 w_k1_KO_s003
                                 + 0.68729 w_k1_MAT_s003
                                 + 1.06129 w_k1_MSFT_s003
                                 + 1.08963 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.13688 w_k1_ADM_s003
                                 + 1.45036 w_k1_AXP_s003
                                 + 1.25811 w_k1_DIS_s003
                                 + 1.24976 w_k1_FDX_s003
                                 + 1.12403 w_k1_IBM_s003
                                 + 1.2328 w_k1_INTC_s003 + 1.34834 w_k1_KO_s003
                                 + 1.45194 w_k1_MAT_s003
                                 + 1.27457 w_k1_MSFT_s003
                                 + 1.14834 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.12895 w_k1_ADM_s003
                                 + 1.20776 w_k1_AXP_s003
                                 + 1.31697 w_k1_DIS_s003
                                 + 1.33565 w_k1_FDX_s003
                                 + 0.913575 w_k1_IBM_s003
                                 + 1.28289 w_k1_INTC_s003
                                 + 1.05865 w_k1_KO_s003
                                 + 0.941145 w_k1_MAT_s003
                                 + 1.43227 w_k1_MSFT_s003
                                 + 1.15765 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.1332 w_k1_ADM_s003
                                 + 1.20121 w_k1_AXP_s003
                                 + 1.34046 w_k1_DIS_s003
                                 + 0.996303 w_k1_FDX_s003
                                 + 1.42705 w_k1_IBM_s003
                                 + 1.10518 w_k1_INTC_s003
                                 + 1.34342 w_k1_KO_s003 + 1.38253 w_k1_MAT_s003
                                 + 1.29318 w_k1_MSFT_s003
                                 + 1.11896 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.983617 w_k1_ADM_s003
                                 + 0.718227 w_k1_AXP_s003
                                 + 0.67803 w_k1_DIS_s003
                                 + 0.782396 w_k1_FDX_s003
                                 + 0.638265 w_k1_IBM_s003
                                 + 0.669955 w_k1_INTC_s003
                                 + 0.729909 w_k1_KO_s003
                                 + 0.694458 w_k1_MAT_s003
                                 + 0.872798 w_k1_MSFT_s003
                                 + 0.82042 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.845851 w_k1_ADM_s003
                                 + 0.899573 w_k1_AXP_s003
                                 + 1.15367 w_k1_DIS_s003
                                 + 0.938536 w_k1_FDX_s003
                                 + 1.0072 w_k1_IBM_s003
                                 + 1.01829 w_k1_INTC_s003
                                 + 1.07226 w_k1_KO_s003 + 1.07737 w_k1_MAT_s003
                                 + 1.24593 w_k1_MSFT_s003
                                 + 1.00338 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.50803 w_k1_ADM_s004
                                 + 1.34877 w_k1_AXP_s004
                                 + 1.60276 w_k1_DIS_s004
                                 + 1.52181 w_k1_FDX_s004
                                 + 1.17888 w_k1_IBM_s004
                                 + 1.31598 w_k1_INTC_s004
                                 + 1.40998 w_k1_KO_s004 + 1.15806 w_k1_MAT_s004
                                 + 1.34563 w_k1_MSFT_s004
                                 + 1.39514 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.00433 w_k1_ADM_s004
                                 + 0.917614 w_k1_AXP_s004
                                 + 0.75733 w_k1_DIS_s004
                                 + 0.950094 w_k1_FDX_s004
                                 + 0.720322 w_k1_IBM_s004
                                 + 0.808933 w_k1_INTC_s004
                                 + 0.785881 w_k1_KO_s004
                                 + 0.766403 w_k1_MAT_s004
                                 + 0.827897 w_k1_MSFT_s004
                                 + 0.863453 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.0152 w_k1_ADM_s004
                                 + 0.890575 w_k1_AXP_s004
                                 + 1.24558 w_k1_DIS_s004
                                 + 1.04586 w_k1_FDX_s004
                                 + 1.06269 w_k1_IBM_s004
                                 + 1.2139 w_k1_INTC_s004 + 1.06712 w_k1_KO_s004
                                 + 0.722422 w_k1_MAT_s004
                                 + 1.15795 w_k1_MSFT_s004
                                 + 0.96441 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.839237 w_k1_ADM_s004
                                 + 0.731241 w_k1_AXP_s004
                                 + 0.826523 w_k1_DIS_s004
                                 + 0.777186 w_k1_FDX_s004
                                 + 1.41085 w_k1_IBM_s004
                                 + 0.842708 w_k1_INTC_s004
                                 + 1.09841 w_k1_KO_s004 + 1.02024 w_k1_MAT_s004
                                 + 0.905614 w_k1_MSFT_s004
                                 + 0.948015 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.23554 w_k1_ADM_s004
                                 + 1.26327 w_k1_AXP_s004
                                 + 0.985746 w_k1_DIS_s004
                                 + 1.13263 w_k1_FDX_s004
                                 + 1.14666 w_k1_IBM_s004
                                 + 1.13888 w_k1_INTC_s004
                                 + 1.02358 w_k1_KO_s004 + 1.07071 w_k1_MAT_s004
                                 + 1.05903 w_k1_MSFT_s004
                                 + 1.06234 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.95909 w_k1_ADM_s004
                                 + 0.73003 w_k1_AXP_s004
                                 + 0.814877 w_k1_DIS_s004
                                 + 0.816335 w_k1_FDX_s004
                                 + 0.82202 w_k1_IBM_s004
                                 + 0.935478 w_k1_INTC_s004
                                 + 0.970473 w_k1_KO_s004
                                 + 0.641332 w_k1_MAT_s004
                                 + 0.981038 w_k1_MSFT_s004
                                 + 0.946115 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.23814 w_k1_ADM_s004
                                 + 1.00122 w_k1_AXP_s004
                                 + 1.28034 w_k1_DIS_s004
                                 + 1.24215 w_k1_FDX_s004
                                 + 0.756901 w_k1_IBM_s004
                                 + 1.15054 w_k1_INTC_s004
                                 + 1.07408 w_k1_KO_s004
                                 + 0.733389 w_k1_MAT_s004
                                 + 1.29498 w_k1_MSFT_s004
                                 + 1.20411 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.43515 w_k1_ADM_s004
                                 + 1.29791 w_k1_AXP_s004
                                 + 1.49954 w_k1_DIS_s004
                                 + 1.19817 w_k1_FDX_s004
                                 + 1.81383 w_k1_IBM_s004
                                 + 1.1716 w_k1_INTC_s004 + 1.65561 w_k1_KO_s004
                                 + 1.69528 w_k1_MAT_s004
                                 + 1.2645 w_k1_MSFT_s004 + 1.29957 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.833667 w_k1_ADM_s005
                                 + 0.707553 w_k1_AXP_s005
                                 + 0.667918 w_k1_DIS_s005
                                 + 0.844781 w_k1_FDX_s005
                                 + 0.650452 w_k1_IBM_s005
                                 + 0.866164 w_k1_INTC_s005
                                 + 0.761667 w_k1_KO_s005
                                 + 0.545932 w_k1_MAT_s005
                                 + 1.0155 w_k1_MSFT_s005
                                 + 0.887462 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07529 w_k1_ADM_s005
                                 + 0.918242 w_k1_AXP_s005
                                 + 1.4397 w_k1_DIS_s005 + 1.21987 w_k1_FDX_s005
                                 + 0.94868 w_k1_IBM_s005
                                 + 1.10846 w_k1_INTC_s005
                                 + 1.24572 w_k1_KO_s005 + 1.11639 w_k1_MAT_s005
                                 + 1.4134 w_k1_MSFT_s005 + 1.12188 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.05473 w_k1_ADM_s005
                                 + 0.923671 w_k1_AXP_s005
                                 + 0.888438 w_k1_DIS_s005
                                 + 1.02061 w_k1_FDX_s005
                                 + 0.857559 w_k1_IBM_s005
                                 + 0.935836 w_k1_INTC_s005
                                 + 1.02971 w_k1_KO_s005
                                 + 0.741946 w_k1_MAT_s005
                                 + 1.01929 w_k1_MSFT_s005
                                 + 1.03625 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.29781 w_k1_ADM_s005
                                 + 1.37744 w_k1_AXP_s005
                                 + 1.36057 w_k1_DIS_s005
                                 + 1.38054 w_k1_FDX_s005
                                 + 1.05602 w_k1_IBM_s005
                                 + 1.07013 w_k1_INTC_s005
                                 + 1.20981 w_k1_KO_s005 + 1.72397 w_k1_MAT_s005
                                 + 1.33257 w_k1_MSFT_s005 + 1.215 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.951222 w_k1_ADM_s005
                                 + 1.01996 w_k1_AXP_s005
                                 + 0.744253 w_k1_DIS_s005
                                 + 0.928046 w_k1_FDX_s005
                                 + 0.810155 w_k1_IBM_s005
                                 + 0.935054 w_k1_INTC_s005
                                 + 0.797501 w_k1_KO_s005
                                 + 0.85663 w_k1_MAT_s005
                                 + 0.898561 w_k1_MSFT_s005
                                 + 0.929205 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.20963 w_k1_ADM_s005
                                 + 1.11578 w_k1_AXP_s005
                                 + 1.25862 w_k1_DIS_s005
                                 + 1.09397 w_k1_FDX_s005
                                 + 1.26033 w_k1_IBM_s005
                                 + 1.30832 w_k1_INTC_s005
                                 + 1.22226 w_k1_KO_s005 + 1.27604 w_k1_MAT_s005
                                 + 1.13496 w_k1_MSFT_s005
                                 + 1.09467 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.04513 w_k1_ADM_s005
                                 + 0.903698 w_k1_AXP_s005
                                 + 0.970608 w_k1_DIS_s005
                                 + 0.88561 w_k1_FDX_s005
                                 + 1.16023 w_k1_IBM_s005
                                 + 0.985501 w_k1_INTC_s005
                                 + 1.10409 w_k1_KO_s005
                                 + 0.775627 w_k1_MAT_s005
                                 + 0.944362 w_k1_MSFT_s005
                                 + 0.948285 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.01225 w_k1_ADM_s005
                                 + 1.34116 w_k1_AXP_s005
                                 + 1.45494 w_k1_DIS_s005 + 1.144 w_k1_FDX_s005
                                 + 1.6073 w_k1_IBM_s005
                                 + 1.21035 w_k1_INTC_s005
                                 + 1.49418 w_k1_KO_s005 + 2.1329 w_k1_MAT_s005
                                 + 1.4131 w_k1_MSFT_s005 + 1.15262 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.07344 w_k1_ADM_s006
                                 + 1.05753 w_k1_AXP_s006
                                 + 1.01616 w_k1_DIS_s006
                                 + 1.06227 w_k1_FDX_s006
                                 + 0.736907 w_k1_IBM_s006
                                 + 0.991389 w_k1_INTC_s006
                                 + 0.829447 w_k1_KO_s006
                                 + 0.877582 w_k1_MAT_s006
                                 + 1.07954 w_k1_MSFT_s006
                                 + 0.917596 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.22606 w_k1_ADM_s006
                                 + 1.31878 w_k1_AXP_s006
                                 + 1.05745 w_k1_DIS_s006
                                 + 0.998709 w_k1_FDX_s006
                                 + 1.24523 w_k1_IBM_s006
                                 + 1.0529 w_k1_INTC_s006
                                 + 0.996947 w_k1_KO_s006
                                 + 1.63434 w_k1_MAT_s006
                                 + 0.945352 w_k1_MSFT_s006
                                 + 0.90225 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.57255 w_k1_ADM_s006
                                 + 1.25751 w_k1_AXP_s006 + 1.3334 w_k1_DIS_s006
                                 + 1.29639 w_k1_FDX_s006
                                 + 1.22952 w_k1_IBM_s006
                                 + 1.25586 w_k1_INTC_s006 + 1.2065 w_k1_KO_s006
                                 + 1.09325 w_k1_MAT_s006
                                 + 1.36945 w_k1_MSFT_s006
                                 + 1.06541 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.997143 w_k1_ADM_s006
                                 + 1.0013 w_k1_AXP_s006 + 0.99043 w_k1_DIS_s006
                                 + 0.95649 w_k1_FDX_s006
                                 + 1.10838 w_k1_IBM_s006
                                 + 1.07761 w_k1_INTC_s006 + 0.9866 w_k1_KO_s006
                                 + 1.19749 w_k1_MAT_s006
                                 + 1.01439 w_k1_MSFT_s006
                                 + 0.967978 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.09322 w_k1_ADM_s006
                                 + 1.18336 w_k1_AXP_s006 + 1.2998 w_k1_DIS_s006
                                 + 1.02391 w_k1_FDX_s006
                                 + 1.01622 w_k1_IBM_s006
                                 + 1.26039 w_k1_INTC_s006
                                 + 0.990908 w_k1_KO_s006
                                 + 0.862016 w_k1_MAT_s006
                                 + 1.17748 w_k1_MSFT_s006
                                 + 0.935115 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.54443 w_k1_ADM_s006
                                 + 1.2345 w_k1_AXP_s006 + 1.30291 w_k1_DIS_s006
                                 + 1.22636 w_k1_FDX_s006
                                 + 0.908134 w_k1_IBM_s006
                                 + 0.936541 w_k1_INTC_s006
                                 + 1.03471 w_k1_KO_s006
                                 + 0.745325 w_k1_MAT_s006
                                 + 1.04816 w_k1_MSFT_s006
                                 + 1.11946 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.897934 w_k1_ADM_s006
                                 + 0.715577 w_k1_AXP_s006
                                 + 0.707648 w_k1_DIS_s006
                                 + 0.806864 w_k1_FDX_s006
                                 + 0.834186 w_k1_IBM_s006
                                 + 0.851552 w_k1_INTC_s006
                                 + 0.824257 w_k1_KO_s006
                                 + 0.60936 w_k1_MAT_s006
                                 + 0.791345 w_k1_MSFT_s006
                                 + 0.866008 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.8169 w_k1_ADM_s006
                                 + 1.66441 w_k1_AXP_s006
                                 + 1.55242 w_k1_DIS_s006
                                 + 1.31253 w_k1_FDX_s006
                                 + 1.20593 w_k1_IBM_s006
                                 + 1.21503 w_k1_INTC_s006 + 1.2177 w_k1_KO_s006
                                 + 1.50909 w_k1_MAT_s006
                                 + 1.27882 w_k1_MSFT_s006
                                 + 1.07749 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.60039 w_k1_ADM_s007
                                 + 1.57073 w_k1_AXP_s007
                                 + 1.36354 w_k1_DIS_s007
                                 + 1.24429 w_k1_FDX_s007
                                 + 1.01552 w_k1_IBM_s007
                                 + 1.09762 w_k1_INTC_s007
                                 + 1.20345 w_k1_KO_s007
                                 + 0.972928 w_k1_MAT_s007
                                 + 1.04866 w_k1_MSFT_s007
                                 + 1.13587 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.19876 w_k1_ADM_s007
                                 + 0.971075 w_k1_AXP_s007
                                 + 1.07155 w_k1_DIS_s007
                                 + 0.90621 w_k1_FDX_s007
                                 + 1.24709 w_k1_IBM_s007
                                 + 0.667932 w_k1_INTC_s007
                                 + 1.13921 w_k1_KO_s007 + 1.12599 w_k1_MAT_s007
                                 + 0.916769 w_k1_MSFT_s007
                                 + 1.04504 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.90251 w_k1_ADM_s007
                                 + 1.19393 w_k1_AXP_s007
                                 + 0.913676 w_k1_DIS_s007
                                 + 1.04254 w_k1_FDX_s007
                                 + 0.792805 w_k1_IBM_s007
                                 + 0.881131 w_k1_INTC_s007
                                 + 0.956578 w_k1_KO_s007
                                 + 0.977995 w_k1_MAT_s007
                                 + 0.964439 w_k1_MSFT_s007
                                 + 0.952191 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.39478 w_k1_ADM_s007
                                 + 1.30733 w_k1_AXP_s007 + 1.0133 w_k1_DIS_s007
                                 + 1.05355 w_k1_FDX_s007
                                 + 1.12764 w_k1_IBM_s007
                                 + 0.789098 w_k1_INTC_s007
                                 + 1.08427 w_k1_KO_s007
                                 + 0.993442 w_k1_MAT_s007
                                 + 0.937741 w_k1_MSFT_s007
                                 + 1.04128 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.17364 w_k1_ADM_s007
                                 + 1.35022 w_k1_AXP_s007
                                 + 1.31599 w_k1_DIS_s007 + 1.1186 w_k1_FDX_s007
                                 + 1.62525 w_k1_IBM_s007
                                 + 1.01708 w_k1_INTC_s007
                                 + 1.30558 w_k1_KO_s007 + 1.4132 w_k1_MAT_s007
                                 + 1.00564 w_k1_MSFT_s007
                                 + 1.09361 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.29743 w_k1_ADM_s007
                                 + 1.12413 w_k1_AXP_s007
                                 + 1.24601 w_k1_DIS_s007
                                 + 1.29118 w_k1_FDX_s007
                                 + 0.722949 w_k1_IBM_s007
                                 + 1.08291 w_k1_INTC_s007
                                 + 1.05696 w_k1_KO_s007
                                 + 0.803762 w_k1_MAT_s007
                                 + 1.12535 w_k1_MSFT_s007
                                 + 1.02896 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.05488 w_k1_ADM_s007
                                 + 0.680903 w_k1_AXP_s007
                                 + 0.83605 w_k1_DIS_s007
                                 + 0.822653 w_k1_FDX_s007
                                 + 0.794183 w_k1_IBM_s007
                                 + 0.65674 w_k1_INTC_s007
                                 + 0.925034 w_k1_KO_s007
                                 + 0.670722 w_k1_MAT_s007
                                 + 0.855383 w_k1_MSFT_s007
                                 + 0.869303 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.21882 w_k1_ADM_s007
                                 + 1.07045 w_k1_AXP_s007 + 1.2697 w_k1_DIS_s007
                                 + 1.06028 w_k1_FDX_s007 + 1.0343 w_k1_IBM_s007
                                 + 0.943937 w_k1_INTC_s007
                                 + 1.26649 w_k1_KO_s007
                                 + 0.842901 w_k1_MAT_s007
                                 + 1.08482 w_k1_MSFT_s007
                                 + 1.05254 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.839459 w_k1_ADM_s010
                                 + 0.910178 w_k1_AXP_s010
                                 + 0.640926 w_k1_DIS_s010
                                 + 0.818471 w_k1_FDX_s010
                                 + 0.906974 w_k1_IBM_s010
                                 + 0.860375 w_k1_INTC_s010
                                 + 0.755081 w_k1_KO_s010
                                 + 0.841978 w_k1_MAT_s010
                                 + 0.854446 w_k1_MSFT_s010
                                 + 0.957496 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.661669 w_k1_ADM_s010
                                 + 0.623652 w_k1_AXP_s010
                                 + 0.639638 w_k1_DIS_s010
                                 + 0.755206 w_k1_FDX_s010
                                 + 0.673058 w_k1_IBM_s010
                                 + 0.725713 w_k1_INTC_s010
                                 + 0.704913 w_k1_KO_s010
                                 + 0.792457 w_k1_MAT_s010
                                 + 0.820132 w_k1_MSFT_s010
                                 + 0.817723 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.757542 w_k1_ADM_s010
                                 + 0.979704 w_k1_AXP_s010
                                 + 0.765184 w_k1_DIS_s010
                                 + 0.845041 w_k1_FDX_s010
                                 + 1.40256 w_k1_IBM_s010
                                 + 0.722551 w_k1_INTC_s010
                                 + 0.941389 w_k1_KO_s010
                                 + 1.71461 w_k1_MAT_s010
                                 + 0.823655 w_k1_MSFT_s010
                                 + 0.929096 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.91106 w_k1_ADM_s010
                                 + 0.913803 w_k1_AXP_s010
                                 + 1.32219 w_k1_DIS_s010
                                 + 1.11817 w_k1_FDX_s010
                                 + 0.897771 w_k1_IBM_s010
                                 + 1.07185 w_k1_INTC_s010
                                 + 1.17607 w_k1_KO_s010
                                 + 0.799782 w_k1_MAT_s010
                                 + 1.30846 w_k1_MSFT_s010 + 1.1106 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.992086 w_k1_ADM_s010
                                 + 1.12068 w_k1_AXP_s010
                                 + 1.01722 w_k1_DIS_s010
                                 + 0.976386 w_k1_FDX_s010
                                 + 1.25044 w_k1_IBM_s010
                                 + 0.80013 w_k1_INTC_s010
                                 + 1.01363 w_k1_KO_s010 + 1.68325 w_k1_MAT_s010
                                 + 1.04656 w_k1_MSFT_s010
                                 + 1.06592 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.35328 w_k1_ADM_s010
                                 + 1.67579 w_k1_AXP_s010
                                 + 1.77057 w_k1_DIS_s010
                                 + 1.37023 w_k1_FDX_s010
                                 + 1.56249 w_k1_IBM_s010
                                 + 1.21002 w_k1_INTC_s010
                                 + 1.52828 w_k1_KO_s010 + 1.58938 w_k1_MAT_s010
                                 + 1.41335 w_k1_MSFT_s010
                                 + 1.24995 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.06634 w_k1_ADM_s010
                                 + 1.53308 w_k1_AXP_s010 + 1.1593 w_k1_DIS_s010
                                 + 1.37116 w_k1_FDX_s010
                                 + 0.833655 w_k1_IBM_s010
                                 + 1.064 w_k1_INTC_s010 + 1.00773 w_k1_KO_s010
                                 + 1.10332 w_k1_MAT_s010
                                 + 1.19499 w_k1_MSFT_s010
                                 + 1.18384 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.889261 w_k1_ADM_s010
                                 + 0.888479 w_k1_AXP_s010
                                 + 0.857654 w_k1_DIS_s010
                                 + 1.00423 w_k1_FDX_s010
                                 + 0.640494 w_k1_IBM_s010
                                 + 0.801641 w_k1_INTC_s010
                                 + 0.83164 w_k1_KO_s010
                                 + 0.829715 w_k1_MAT_s010
                                 + 1.03154 w_k1_MSFT_s010
                                 + 1.01101 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.74869 w_k1_ADM_s011
                                 + 0.63803 w_k1_AXP_s011
                                 + 0.973654 w_k1_DIS_s011
                                 + 0.854238 w_k1_FDX_s011
                                 + 1.08143 w_k1_IBM_s011
                                 + 0.816062 w_k1_INTC_s011
                                 + 0.86021 w_k1_KO_s011
                                 + 0.639543 w_k1_MAT_s011
                                 + 1.00985 w_k1_MSFT_s011 + 1.03 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.935221 w_k1_ADM_s011
                                 + 1.36597 w_k1_AXP_s011 + 1.1778 w_k1_DIS_s011
                                 + 1.07274 w_k1_FDX_s011
                                 + 1.31552 w_k1_IBM_s011
                                 + 1.02775 w_k1_INTC_s011
                                 + 1.01161 w_k1_KO_s011 + 1.4311 w_k1_MAT_s011
                                 + 1.14403 w_k1_MSFT_s011
                                 + 1.08764 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.99842 w_k1_ADM_s011
                                 + 1.09021 w_k1_AXP_s011
                                 + 0.86986 w_k1_DIS_s011
                                 + 1.01994 w_k1_FDX_s011
                                 + 0.787964 w_k1_IBM_s011
                                 + 1.07528 w_k1_INTC_s011
                                 + 0.79645 w_k1_KO_s011
                                 + 0.808293 w_k1_MAT_s011
                                 + 1.06776 w_k1_MSFT_s011
                                 + 0.94665 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.984501 w_k1_ADM_s011
                                 + 1.06411 w_k1_AXP_s011
                                 + 1.05282 w_k1_DIS_s011
                                 + 0.943286 w_k1_FDX_s011
                                 + 1.14737 w_k1_IBM_s011
                                 + 1.10186 w_k1_INTC_s011
                                 + 0.987671 w_k1_KO_s011
                                 + 1.01629 w_k1_MAT_s011
                                 + 1.00484 w_k1_MSFT_s011
                                 + 0.996557 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.837691 w_k1_ADM_s011
                                 + 0.997771 w_k1_AXP_s011
                                 + 0.730652 w_k1_DIS_s011
                                 + 0.791693 w_k1_FDX_s011
                                 + 1.30248 w_k1_IBM_s011
                                 + 0.722461 w_k1_INTC_s011
                                 + 0.783406 w_k1_KO_s011
                                 + 1.18572 w_k1_MAT_s011
                                 + 0.766653 w_k1_MSFT_s011
                                 + 0.910597 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.18179 w_k1_ADM_s011
                                 + 1.25316 w_k1_AXP_s011
                                 + 0.990348 w_k1_DIS_s011
                                 + 1.18508 w_k1_FDX_s011
                                 + 0.829881 w_k1_IBM_s011
                                 + 1.01248 w_k1_INTC_s011
                                 + 0.868468 w_k1_KO_s011
                                 + 0.78033 w_k1_MAT_s011
                                 + 1.16439 w_k1_MSFT_s011
                                 + 1.09755 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.02221 w_k1_ADM_s011
                                 + 1.35806 w_k1_AXP_s011
                                 + 1.67964 w_k1_DIS_s011
                                 + 1.30161 w_k1_FDX_s011
                                 + 1.04035 w_k1_IBM_s011
                                 + 1.21927 w_k1_INTC_s011
                                 + 0.992662 w_k1_KO_s011
                                 + 1.01234 w_k1_MAT_s011
                                 + 1.17338 w_k1_MSFT_s011
                                 + 1.11919 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.695367 w_k1_ADM_s011
                                 + 0.761961 w_k1_AXP_s011
                                 + 0.730203 w_k1_DIS_s011
                                 + 0.834876 w_k1_FDX_s011
                                 + 0.760066 w_k1_IBM_s011
                                 + 0.821597 w_k1_INTC_s011
                                 + 0.725894 w_k1_KO_s011
                                 + 0.569737 w_k1_MAT_s011
                                 + 0.77232 w_k1_MSFT_s011
                                 + 0.83934 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.02003 w_k1_ADM_s012
                                 + 0.994279 w_k1_AXP_s012
                                 + 1.0585 w_k1_DIS_s012
                                 + 0.989051 w_k1_FDX_s012
                                 + 1.06725 w_k1_IBM_s012
                                 + 0.979055 w_k1_INTC_s012
                                 + 0.93371 w_k1_KO_s012
                                 + 0.797537 w_k1_MAT_s012
                                 + 1.03895 w_k1_MSFT_s012
                                 + 0.938706 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.19608 w_k1_ADM_s012
                                 + 1.23726 w_k1_AXP_s012
                                 + 1.07647 w_k1_DIS_s012
                                 + 1.24979 w_k1_FDX_s012
                                 + 0.800871 w_k1_IBM_s012
                                 + 0.986274 w_k1_INTC_s012
                                 + 0.980965 w_k1_KO_s012
                                 + 0.74307 w_k1_MAT_s012
                                 + 1.12535 w_k1_MSFT_s012
                                 + 1.03502 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.921515 w_k1_ADM_s012
                                 + 0.727277 w_k1_AXP_s012
                                 + 0.71056 w_k1_DIS_s012
                                 + 0.810931 w_k1_FDX_s012
                                 + 0.633822 w_k1_IBM_s012
                                 + 0.781648 w_k1_INTC_s012
                                 + 0.782159 w_k1_KO_s012
                                 + 0.551603 w_k1_MAT_s012
                                 + 0.86368 w_k1_MSFT_s012
                                 + 0.902725 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.00023 w_k1_ADM_s012
                                 + 1.26913 w_k1_AXP_s012
                                 + 1.26347 w_k1_DIS_s012
                                 + 1.18446 w_k1_FDX_s012
                                 + 1.16084 w_k1_IBM_s012
                                 + 1.22494 w_k1_INTC_s012
                                 + 1.10116 w_k1_KO_s012 + 1.00306 w_k1_MAT_s012
                                 + 1.0389 w_k1_MSFT_s012 + 1.0039 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.10782 w_k1_ADM_s012
                                 + 1.27784 w_k1_AXP_s012
                                 + 1.71823 w_k1_DIS_s012
                                 + 1.23598 w_k1_FDX_s012
                                 + 1.42015 w_k1_IBM_s012
                                 + 1.45588 w_k1_INTC_s012
                                 + 1.44705 w_k1_KO_s012 + 1.48395 w_k1_MAT_s012
                                 + 1.34329 w_k1_MSFT_s012
                                 + 1.03967 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.01965 w_k1_ADM_s012
                                 + 0.820936 w_k1_AXP_s012
                                 + 0.750941 w_k1_DIS_s012
                                 + 0.880485 w_k1_FDX_s012
                                 + 1.10021 w_k1_IBM_s012
                                 + 0.792308 w_k1_INTC_s012
                                 + 0.911126 w_k1_KO_s012
                                 + 0.731564 w_k1_MAT_s012
                                 + 0.806221 w_k1_MSFT_s012
                                 + 0.87315 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.537029 w_k1_ADM_s012
                                 + 0.535294 w_k1_AXP_s012
                                 + 0.63996 w_k1_DIS_s012
                                 + 0.687384 w_k1_FDX_s012
                                 + 0.904637 w_k1_IBM_s012
                                 + 0.802573 w_k1_INTC_s012
                                 + 0.797392 w_k1_KO_s012
                                 + 0.705244 w_k1_MAT_s012
                                 + 0.747077 w_k1_MSFT_s012
                                 + 0.799705 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.52358 w_k1_ADM_s012
                                 + 1.5341 w_k1_AXP_s012 + 1.80529 w_k1_DIS_s012
                                 + 1.54191 w_k1_FDX_s012 + 1.4439 w_k1_IBM_s012
                                 + 1.09307 w_k1_INTC_s012
                                 + 1.34962 w_k1_KO_s012 + 1.53988 w_k1_MAT_s012
                                 + 1.45807 w_k1_MSFT_s012 + 1.2956 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.99236 w_k1_ADM_s013
                                 + 0.759835 w_k1_AXP_s013
                                 + 1.06125 w_k1_DIS_s013
                                 + 0.939018 w_k1_FDX_s013
                                 + 1.1665 w_k1_IBM_s013
                                 + 1.01824 w_k1_INTC_s013
                                 + 1.24306 w_k1_KO_s013 + 1.1259 w_k1_MAT_s013
                                 + 1.1052 w_k1_MSFT_s013 + 1.0884 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.956783 w_k1_ADM_s013
                                 + 0.785482 w_k1_AXP_s013
                                 + 0.711392 w_k1_DIS_s013
                                 + 0.793576 w_k1_FDX_s013
                                 + 0.950736 w_k1_IBM_s013
                                 + 0.932604 w_k1_INTC_s013
                                 + 0.875252 w_k1_KO_s013
                                 + 1.12914 w_k1_MAT_s013
                                 + 0.785239 w_k1_MSFT_s013
                                 + 0.84031 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.21913 w_k1_ADM_s013
                                 + 1.62158 w_k1_AXP_s013
                                 + 1.60491 w_k1_DIS_s013
                                 + 1.53068 w_k1_FDX_s013 + 1.1677 w_k1_IBM_s013
                                 + 1.32643 w_k1_INTC_s013
                                 + 1.23478 w_k1_KO_s013 + 1.0177 w_k1_MAT_s013
                                 + 1.22091 w_k1_MSFT_s013
                                 + 1.38999 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.945036 w_k1_ADM_s013
                                 + 0.944532 w_k1_AXP_s013
                                 + 0.842268 w_k1_DIS_s013
                                 + 1.00058 w_k1_FDX_s013
                                 + 0.727223 w_k1_IBM_s013
                                 + 1.1937 w_k1_INTC_s013
                                 + 0.825961 w_k1_KO_s013
                                 + 0.703892 w_k1_MAT_s013
                                 + 1.03571 w_k1_MSFT_s013
                                 + 0.974301 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.22064 w_k1_ADM_s013
                                 + 1.20386 w_k1_AXP_s013
                                 + 1.18187 w_k1_DIS_s013
                                 + 1.25064 w_k1_FDX_s013
                                 + 0.915203 w_k1_IBM_s013
                                 + 1.10601 w_k1_INTC_s013
                                 + 1.03266 w_k1_KO_s013 + 1.35219 w_k1_MAT_s013
                                 + 1.10948 w_k1_MSFT_s013
                                 + 1.16196 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.777035 w_k1_ADM_s013
                                 + 0.649914 w_k1_AXP_s013
                                 + 0.707147 w_k1_DIS_s013
                                 + 0.750815 w_k1_FDX_s013
                                 + 0.784568 w_k1_IBM_s013
                                 + 0.815007 w_k1_INTC_s013
                                 + 0.845755 w_k1_KO_s013
                                 + 0.506835 w_k1_MAT_s013
                                 + 0.857465 w_k1_MSFT_s013
                                 + 0.7801 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.21422 w_k1_ADM_s013
                                 + 1.13494 w_k1_AXP_s013
                                 + 1.02294 w_k1_DIS_s013
                                 + 0.960188 w_k1_FDX_s013
                                 + 1.50543 w_k1_IBM_s013
                                 + 0.916872 w_k1_INTC_s013
                                 + 1.28026 w_k1_KO_s013 + 1.67403 w_k1_MAT_s013
                                 + 0.920955 w_k1_MSFT_s013
                                 + 1.01007 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.11195 w_k1_ADM_s013
                                 + 1.18851 w_k1_AXP_s013
                                 + 1.44919 w_k1_DIS_s013
                                 + 1.16859 w_k1_FDX_s013
                                 + 1.21863 w_k1_IBM_s013
                                 + 1.50909 w_k1_INTC_s013
                                 + 1.26815 w_k1_KO_s013 + 1.08398 w_k1_MAT_s013
                                 + 1.34037 w_k1_MSFT_s013
                                 + 1.17878 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.14895 w_k1_ADM_s014
                                 + 1.3126 w_k1_AXP_s014 + 1.49221 w_k1_DIS_s014
                                 + 1.16721 w_k1_FDX_s014
                                 + 1.40636 w_k1_IBM_s014
                                 + 1.20817 w_k1_INTC_s014
                                 + 1.23254 w_k1_KO_s014 + 1.274 w_k1_MAT_s014
                                 + 1.19458 w_k1_MSFT_s014
                                 + 1.07772 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.95208 w_k1_ADM_s014
                                 + 0.820015 w_k1_AXP_s014
                                 + 0.722069 w_k1_DIS_s014
                                 + 0.807557 w_k1_FDX_s014
                                 + 0.846891 w_k1_IBM_s014
                                 + 0.781355 w_k1_INTC_s014
                                 + 0.765754 w_k1_KO_s014
                                 + 0.656352 w_k1_MAT_s014
                                 + 0.85844 w_k1_MSFT_s014
                                 + 0.917481 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.09098 w_k1_ADM_s014
                                 + 1.11509 w_k1_AXP_s014
                                 + 1.66112 w_k1_DIS_s014
                                 + 1.25826 w_k1_FDX_s014
                                 + 0.881194 w_k1_IBM_s014
                                 + 1.06294 w_k1_INTC_s014
                                 + 1.09338 w_k1_KO_s014
                                 + 0.732247 w_k1_MAT_s014
                                 + 1.24884 w_k1_MSFT_s014
                                 + 1.18511 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.04351 w_k1_ADM_s014
                                 + 1.02076 w_k1_AXP_s014
                                 + 0.876116 w_k1_DIS_s014
                                 + 0.856855 w_k1_FDX_s014
                                 + 1.12894 w_k1_IBM_s014
                                 + 0.869541 w_k1_INTC_s014
                                 + 0.811333 w_k1_KO_s014
                                 + 1.08561 w_k1_MAT_s014
                                 + 0.893415 w_k1_MSFT_s014
                                 + 0.967794 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.5151 w_k1_ADM_s014
                                 + 1.67744 w_k1_AXP_s014
                                 + 1.59826 w_k1_DIS_s014
                                 + 1.39173 w_k1_FDX_s014
                                 + 0.919743 w_k1_IBM_s014
                                 + 1.16393 w_k1_INTC_s014
                                 + 1.22295 w_k1_KO_s014 + 1.0399 w_k1_MAT_s014
                                 + 1.185 w_k1_MSFT_s014 + 1.1553 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.00797 w_k1_ADM_s014
                                 + 1.11407 w_k1_AXP_s014
                                 + 1.21759 w_k1_DIS_s014
                                 + 0.963224 w_k1_FDX_s014
                                 + 1.48324 w_k1_IBM_s014
                                 + 0.90656 w_k1_INTC_s014
                                 + 1.25963 w_k1_KO_s014 + 1.35507 w_k1_MAT_s014
                                 + 0.954294 w_k1_MSFT_s014
                                 + 1.03345 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.920791 w_k1_ADM_s014
                                 + 0.959811 w_k1_AXP_s014
                                 + 0.956742 w_k1_DIS_s014
                                 + 1.02164 w_k1_FDX_s014
                                 + 0.71184 w_k1_IBM_s014
                                 + 0.951319 w_k1_INTC_s014
                                 + 0.88621 w_k1_KO_s014
                                 + 0.946366 w_k1_MAT_s014
                                 + 1.01771 w_k1_MSFT_s014 + 1.0206 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.29238 w_k1_ADM_s014
                                 + 1.05538 w_k1_AXP_s014
                                 + 1.19248 w_k1_DIS_s014 + 1.2148 w_k1_FDX_s014
                                 + 0.756605 w_k1_IBM_s014
                                 + 0.913157 w_k1_INTC_s014
                                 + 0.88531 w_k1_KO_s014 + 0.72403 w_k1_MAT_s014
                                 + 0.993623 w_k1_MSFT_s014
                                 + 1.10126 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.13879 w_k1_ADM_s015
                                 + 1.03322 w_k1_AXP_s015
                                 + 0.973097 w_k1_DIS_s015
                                 + 1.1284 w_k1_FDX_s015
                                 + 0.733087 w_k1_IBM_s015
                                 + 1.01086 w_k1_INTC_s015
                                 + 0.931438 w_k1_KO_s015
                                 + 0.716288 w_k1_MAT_s015
                                 + 1.07162 w_k1_MSFT_s015
                                 + 0.978401 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.896464 w_k1_ADM_s015
                                 + 0.652535 w_k1_AXP_s015
                                 + 0.713821 w_k1_DIS_s015
                                 + 0.887761 w_k1_FDX_s015
                                 + 0.664723 w_k1_IBM_s015
                                 + 0.806565 w_k1_INTC_s015
                                 + 0.764957 w_k1_KO_s015
                                 + 0.641295 w_k1_MAT_s015
                                 + 0.777368 w_k1_MSFT_s015
                                 + 0.885157 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.883416 w_k1_ADM_s015
                                 + 0.854355 w_k1_AXP_s015
                                 + 0.981518 w_k1_DIS_s015
                                 + 0.940845 w_k1_FDX_s015
                                 + 1.30832 w_k1_IBM_s015
                                 + 0.875245 w_k1_INTC_s015
                                 + 1.07761 w_k1_KO_s015
                                 + 0.997421 w_k1_MAT_s015
                                 + 0.953837 w_k1_MSFT_s015
                                 + 0.934526 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.7751 w_k1_ADM_s015
                                 + 0.789241 w_k1_AXP_s015
                                 + 0.629511 w_k1_DIS_s015
                                 + 0.751385 w_k1_FDX_s015
                                 + 0.797762 w_k1_IBM_s015
                                 + 0.833093 w_k1_INTC_s015
                                 + 0.824636 w_k1_KO_s015
                                 + 0.861575 w_k1_MAT_s015
                                 + 0.849431 w_k1_MSFT_s015
                                 + 0.876083 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.3372 w_k1_ADM_s015
                                 + 1.20238 w_k1_AXP_s015
                                 + 1.21287 w_k1_DIS_s015
                                 + 1.18528 w_k1_FDX_s015
                                 + 1.09224 w_k1_IBM_s015
                                 + 1.13886 w_k1_INTC_s015
                                 + 1.17902 w_k1_KO_s015 + 1.22527 w_k1_MAT_s015
                                 + 1.09216 w_k1_MSFT_s015
                                 + 1.20649 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.786168 w_k1_ADM_s015
                                 + 0.67512 w_k1_AXP_s015
                                 + 0.900117 w_k1_DIS_s015
                                 + 0.956048 w_k1_FDX_s015
                                 + 0.886474 w_k1_IBM_s015
                                 + 0.930838 w_k1_INTC_s015
                                 + 0.954429 w_k1_KO_s015
                                 + 0.638156 w_k1_MAT_s015
                                 + 0.949073 w_k1_MSFT_s015
                                 + 0.960526 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.745292 w_k1_ADM_s015
                                 + 0.889571 w_k1_AXP_s015
                                 + 0.836763 w_k1_DIS_s015
                                 + 0.795053 w_k1_FDX_s015
                                 + 1.0154 w_k1_IBM_s015
                                 + 0.934017 w_k1_INTC_s015
                                 + 0.965618 w_k1_KO_s015 + 1.0215 w_k1_MAT_s015
                                 + 0.77741 w_k1_MSFT_s015
                                 + 0.916181 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.488 w_k1_ADM_s015
                                 + 1.29632 w_k1_AXP_s015
                                 + 1.43295 w_k1_DIS_s015
                                 + 1.07366 w_k1_FDX_s015
                                 + 1.76894 w_k1_IBM_s015
                                 + 0.827347 w_k1_INTC_s015
                                 + 1.35716 w_k1_KO_s015 + 2.21215 w_k1_MAT_s015
                                 + 1.11082 w_k1_MSFT_s015
                                 + 1.24482 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.31091 w_k1_ADM_s016
                                 + 1.49079 w_k1_AXP_s016
                                 + 1.55026 w_k1_DIS_s016
                                 + 1.25094 w_k1_FDX_s016
                                 + 1.75965 w_k1_IBM_s016
                                 + 1.15327 w_k1_INTC_s016
                                 + 1.26774 w_k1_KO_s016 + 1.36938 w_k1_MAT_s016
                                 + 1.33848 w_k1_MSFT_s016
                                 + 1.13218 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.993204 w_k1_ADM_s016
                                 + 0.977053 w_k1_AXP_s016
                                 + 1.06677 w_k1_DIS_s016
                                 + 0.950675 w_k1_FDX_s016
                                 + 1.50523 w_k1_IBM_s016
                                 + 0.842045 w_k1_INTC_s016
                                 + 1.15397 w_k1_KO_s016 + 1.69353 w_k1_MAT_s016
                                 + 1.00559 w_k1_MSFT_s016
                                 + 1.13866 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.07977 w_k1_ADM_s016
                                 + 1.39744 w_k1_AXP_s016
                                 + 1.63573 w_k1_DIS_s016
                                 + 1.33999 w_k1_FDX_s016
                                 + 1.16826 w_k1_IBM_s016
                                 + 1.09347 w_k1_INTC_s016
                                 + 1.16944 w_k1_KO_s016 + 1.56688 w_k1_MAT_s016
                                 + 1.27835 w_k1_MSFT_s016
                                 + 1.19858 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.60787 w_k1_ADM_s016
                                 + 0.611841 w_k1_AXP_s016
                                 + 0.52622 w_k1_DIS_s016
                                 + 0.642539 w_k1_FDX_s016
                                 + 0.85853 w_k1_IBM_s016
                                 + 0.64257 w_k1_INTC_s016
                                 + 0.640062 w_k1_KO_s016
                                 + 0.818121 w_k1_MAT_s016
                                 + 0.739426 w_k1_MSFT_s016
                                 + 0.761117 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.902524 w_k1_ADM_s016
                                 + 0.940316 w_k1_AXP_s016
                                 + 0.893684 w_k1_DIS_s016
                                 + 0.936693 w_k1_FDX_s016
                                 + 0.990944 w_k1_IBM_s016
                                 + 0.845606 w_k1_INTC_s016
                                 + 0.971174 w_k1_KO_s016
                                 + 0.808124 w_k1_MAT_s016
                                 + 1.02764 w_k1_MSFT_s016
                                 + 0.931406 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.96822 w_k1_ADM_s016
                                 + 0.985219 w_k1_AXP_s016
                                 + 1.00419 w_k1_DIS_s016
                                 + 1.03169 w_k1_FDX_s016
                                 + 0.632321 w_k1_IBM_s016
                                 + 0.809523 w_k1_INTC_s016
                                 + 0.746188 w_k1_KO_s016
                                 + 0.751981 w_k1_MAT_s016
                                 + 1.06247 w_k1_MSFT_s016
                                 + 0.982245 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.954083 w_k1_ADM_s016
                                 + 1.21974 w_k1_AXP_s016
                                 + 1.13455 w_k1_DIS_s016
                                 + 1.08607 w_k1_FDX_s016
                                 + 1.03346 w_k1_IBM_s016 + 1.009 w_k1_INTC_s016
                                 + 0.998471 w_k1_KO_s016
                                 + 0.956257 w_k1_MAT_s016
                                 + 1.08129 w_k1_MSFT_s016
                                 + 1.08067 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.73532 w_k1_ADM_s016
                                 + 0.684387 w_k1_AXP_s016
                                 + 0.714688 w_k1_DIS_s016
                                 + 0.77736 w_k1_FDX_s016
                                 + 0.737406 w_k1_IBM_s016
                                 + 0.784436 w_k1_INTC_s016
                                 + 0.70488 w_k1_KO_s016
                                 + 0.683216 w_k1_MAT_s016
                                 + 0.790573 w_k1_MSFT_s016
                                 + 0.923588 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.19592 w_k1_ADM_s017
                                 + 0.920923 w_k1_AXP_s017
                                 + 0.995309 w_k1_DIS_s017
                                 + 1.05704 w_k1_FDX_s017
                                 + 0.796944 w_k1_IBM_s017
                                 + 1.02894 w_k1_INTC_s017
                                 + 0.82881 w_k1_KO_s017
                                 + 0.945072 w_k1_MAT_s017
                                 + 1.04882 w_k1_MSFT_s017
                                 + 0.994326 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.14191 w_k1_ADM_s017
                                 + 1.54482 w_k1_AXP_s017
                                 + 1.57741 w_k1_DIS_s017
                                 + 1.29468 w_k1_FDX_s017
                                 + 1.29859 w_k1_IBM_s017
                                 + 1.15994 w_k1_INTC_s017
                                 + 1.12362 w_k1_KO_s017 + 1.33789 w_k1_MAT_s017
                                 + 1.1345 w_k1_MSFT_s017 + 1.05019 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.972451 w_k1_ADM_s017
                                 + 0.928681 w_k1_AXP_s017
                                 + 1.06073 w_k1_DIS_s017
                                 + 0.884082 w_k1_FDX_s017
                                 + 1.5888 w_k1_IBM_s017
                                 + 0.932181 w_k1_INTC_s017
                                 + 1.12276 w_k1_KO_s017 + 1.30019 w_k1_MAT_s017
                                 + 0.998036 w_k1_MSFT_s017
                                 + 0.963578 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.935944 w_k1_ADM_s017
                                 + 0.773929 w_k1_AXP_s017
                                 + 0.722715 w_k1_DIS_s017
                                 + 0.753511 w_k1_FDX_s017
                                 + 0.833316 w_k1_IBM_s017
                                 + 0.661767 w_k1_INTC_s017
                                 + 0.761307 w_k1_KO_s017
                                 + 1.04979 w_k1_MAT_s017
                                 + 0.777952 w_k1_MSFT_s017
                                 + 0.799015 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.94444 w_k1_ADM_s017
                                 + 0.775789 w_k1_AXP_s017
                                 + 1.17827 w_k1_DIS_s017
                                 + 0.942097 w_k1_FDX_s017
                                 + 0.99619 w_k1_IBM_s017
                                 + 0.948088 w_k1_INTC_s017
                                 + 0.945587 w_k1_KO_s017
                                 + 0.902756 w_k1_MAT_s017
                                 + 1.04041 w_k1_MSFT_s017
                                 + 0.928984 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.901755 w_k1_ADM_s017
                                 + 0.869098 w_k1_AXP_s017
                                 + 0.968982 w_k1_DIS_s017
                                 + 0.941141 w_k1_FDX_s017
                                 + 0.906293 w_k1_IBM_s017
                                 + 0.992984 w_k1_INTC_s017
                                 + 0.858848 w_k1_KO_s017
                                 + 0.993504 w_k1_MAT_s017
                                 + 0.999091 w_k1_MSFT_s017
                                 + 0.889606 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.872513 w_k1_ADM_s017
                                 + 0.950539 w_k1_AXP_s017
                                 + 0.981584 w_k1_DIS_s017
                                 + 0.982308 w_k1_FDX_s017
                                 + 1.13841 w_k1_IBM_s017
                                 + 0.916584 w_k1_INTC_s017
                                 + 1.0447 w_k1_KO_s017 + 1.3546 w_k1_MAT_s017
                                 + 0.920866 w_k1_MSFT_s017
                                 + 0.975622 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.33371 w_k1_ADM_s017
                                 + 1.41073 w_k1_AXP_s017
                                 + 1.21918 w_k1_DIS_s017
                                 + 1.07354 w_k1_FDX_s017
                                 + 1.14154 w_k1_IBM_s017
                                 + 0.982664 w_k1_INTC_s017
                                 + 1.22925 w_k1_KO_s017 + 1.61034 w_k1_MAT_s017
                                 + 0.94567 w_k1_MSFT_s017
                                 + 1.00414 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.08343 w_k1_ADM_s100
                                 + 0.837332 w_k1_AXP_s100
                                 + 0.702925 w_k1_DIS_s100
                                 + 0.888741 w_k1_FDX_s100
                                 + 0.859987 w_k1_IBM_s100
                                 + 0.776155 w_k1_INTC_s100
                                 + 0.791447 w_k1_KO_s100
                                 + 1.00651 w_k1_MAT_s100
                                 + 0.843588 w_k1_MSFT_s100
                                 + 0.94585 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.08942 w_k1_ADM_s100
                                 + 1.50752 w_k1_AXP_s100
                                 + 1.59861 w_k1_DIS_s100
                                 + 1.25283 w_k1_FDX_s100
                                 + 1.66166 w_k1_IBM_s100
                                 + 1.09443 w_k1_INTC_s100
                                 + 1.21481 w_k1_KO_s100 + 1.87113 w_k1_MAT_s100
                                 + 1.11222 w_k1_MSFT_s100
                                 + 1.10678 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.08982 w_k1_ADM_s100
                                 + 1.29754 w_k1_AXP_s100
                                 + 1.10474 w_k1_DIS_s100
                                 + 1.20223 w_k1_FDX_s100
                                 + 1.16047 w_k1_IBM_s100
                                 + 1.02741 w_k1_INTC_s100
                                 + 1.11284 w_k1_KO_s100 + 1.05939 w_k1_MAT_s100
                                 + 0.993523 w_k1_MSFT_s100
                                 + 1.08795 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.39878 w_k1_ADM_s100
                                 + 1.43207 w_k1_AXP_s100
                                 + 1.48315 w_k1_DIS_s100
                                 + 1.36879 w_k1_FDX_s100
                                 + 1.07009 w_k1_IBM_s100
                                 + 0.996833 w_k1_INTC_s100
                                 + 0.992437 w_k1_KO_s100
                                 + 1.12297 w_k1_MAT_s100
                                 + 1.16083 w_k1_MSFT_s100
                                 + 1.28073 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.633947 w_k1_ADM_s100
                                 + 1.0896 w_k1_AXP_s100 + 1.00163 w_k1_DIS_s100
                                 + 0.966743 w_k1_FDX_s100
                                 + 1.39334 w_k1_IBM_s100
                                 + 1.04034 w_k1_INTC_s100
                                 + 1.16582 w_k1_KO_s100 + 1.59485 w_k1_MAT_s100
                                 + 0.926105 w_k1_MSFT_s100
                                 + 1.06069 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.00972 w_k1_ADM_s100
                                 + 0.95396 w_k1_AXP_s100
                                 + 1.12509 w_k1_DIS_s100
                                 + 1.04893 w_k1_FDX_s100
                                 + 1.00328 w_k1_IBM_s100
                                 + 0.881917 w_k1_INTC_s100
                                 + 1.12023 w_k1_KO_s100 + 1.11202 w_k1_MAT_s100
                                 + 1.17794 w_k1_MSFT_s100
                                 + 1.04987 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.842981 w_k1_ADM_s100
                                 + 1.06167 w_k1_AXP_s100
                                 + 0.671826 w_k1_DIS_s100
                                 + 0.714654 w_k1_FDX_s100
                                 + 1.20537 w_k1_IBM_s100
                                 + 0.774623 w_k1_INTC_s100
                                 + 0.813116 w_k1_KO_s100
                                 + 1.28527 w_k1_MAT_s100
                                 + 0.69068 w_k1_MSFT_s100
                                 + 0.900718 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.656678 w_k1_ADM_s100
                                 + 0.888837 w_k1_AXP_s100
                                 + 0.819675 w_k1_DIS_s100
                                 + 0.909853 w_k1_FDX_s100
                                 + 0.647322 w_k1_IBM_s100
                                 + 1.03911 w_k1_INTC_s100
                                 + 0.812043 w_k1_KO_s100
                                 + 0.734335 w_k1_MAT_s100
                                 + 0.884843 w_k1_MSFT_s100
                                 + 0.846633 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.835094 w_k1_ADM_s101
                                 + 0.699003 w_k1_AXP_s101
                                 + 0.716898 w_k1_DIS_s101
                                 + 0.842339 w_k1_FDX_s101
                                 + 0.720132 w_k1_IBM_s101
                                 + 0.866122 w_k1_INTC_s101
                                 + 0.76248 w_k1_KO_s101
                                 + 0.825655 w_k1_MAT_s101
                                 + 0.865904 w_k1_MSFT_s101
                                 + 0.886704 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.830767 w_k1_ADM_s101
                                 + 1.14354 w_k1_AXP_s101
                                 + 0.795776 w_k1_DIS_s101
                                 + 0.898068 w_k1_FDX_s101
                                 + 0.840236 w_k1_IBM_s101
                                 + 0.843519 w_k1_INTC_s101
                                 + 0.777633 w_k1_KO_s101
                                 + 0.945617 w_k1_MAT_s101
                                 + 0.840702 w_k1_MSFT_s101
                                 + 0.870933 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.643244 w_k1_ADM_s101
                                 + 0.759753 w_k1_AXP_s101
                                 + 0.818826 w_k1_DIS_s101
                                 + 0.83366 w_k1_FDX_s101
                                 + 1.17712 w_k1_IBM_s101
                                 + 0.83189 w_k1_INTC_s101
                                 + 0.898083 w_k1_KO_s101
                                 + 1.11188 w_k1_MAT_s101
                                 + 0.81178 w_k1_MSFT_s101 + 1.0107 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.18188 w_k1_ADM_s101
                                 + 1.40426 w_k1_AXP_s101
                                 + 1.79684 w_k1_DIS_s101
                                 + 1.50131 w_k1_FDX_s101
                                 + 1.20818 w_k1_IBM_s101
                                 + 1.29664 w_k1_INTC_s101
                                 + 1.41086 w_k1_KO_s101 + 1.58529 w_k1_MAT_s101
                                 + 1.50589 w_k1_MSFT_s101
                                 + 1.31183 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.877529 w_k1_ADM_s101
                                 + 0.951085 w_k1_AXP_s101
                                 + 0.915772 w_k1_DIS_s101
                                 + 1.06227 w_k1_FDX_s101
                                 + 0.934328 w_k1_IBM_s101
                                 + 0.891613 w_k1_INTC_s101
                                 + 0.787658 w_k1_KO_s101
                                 + 0.839225 w_k1_MAT_s101
                                 + 1.05647 w_k1_MSFT_s101
                                 + 0.903596 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.898433 w_k1_ADM_s101
                                 + 1.32046 w_k1_AXP_s101
                                 + 1.14469 w_k1_DIS_s101
                                 + 1.05914 w_k1_FDX_s101
                                 + 1.39586 w_k1_IBM_s101
                                 + 0.837848 w_k1_INTC_s101
                                 + 1.03517 w_k1_KO_s101 + 1.2776 w_k1_MAT_s101
                                 + 0.960994 w_k1_MSFT_s101
                                 + 1.0308 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.14766 w_k1_ADM_s101
                                 + 1.31729 w_k1_AXP_s101
                                 + 1.17169 w_k1_DIS_s101
                                 + 1.31449 w_k1_FDX_s101
                                 + 0.98813 w_k1_IBM_s101
                                 + 1.11037 w_k1_INTC_s101
                                 + 1.08606 w_k1_KO_s101 + 1.61784 w_k1_MAT_s101
                                 + 1.18739 w_k1_MSFT_s101 + 1.1676 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.989363 w_k1_ADM_s101
                                 + 0.898757 w_k1_AXP_s101
                                 + 1.34848 w_k1_DIS_s101
                                 + 1.09226 w_k1_FDX_s101
                                 + 1.10586 w_k1_IBM_s101
                                 + 1.14988 w_k1_INTC_s101
                                 + 1.20945 w_k1_KO_s101 + 1.19244 w_k1_MAT_s101
                                 + 1.2146 w_k1_MSFT_s101 + 1.17599 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.856297 w_k1_ADM_s102
                                 + 0.822774 w_k1_AXP_s102
                                 + 0.846051 w_k1_DIS_s102
                                 + 0.989345 w_k1_FDX_s102
                                 + 1.02781 w_k1_IBM_s102
                                 + 1.07242 w_k1_INTC_s102
                                 + 0.995348 w_k1_KO_s102
                                 + 1.10349 w_k1_MAT_s102
                                 + 0.954166 w_k1_MSFT_s102
                                 + 1.06984 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.14916 w_k1_ADM_s102
                                 + 0.881328 w_k1_AXP_s102
                                 + 1.25435 w_k1_DIS_s102
                                 + 1.06258 w_k1_FDX_s102
                                 + 1.14064 w_k1_IBM_s102
                                 + 0.926324 w_k1_INTC_s102
                                 + 1.21156 w_k1_KO_s102
                                 + 0.970101 w_k1_MAT_s102
                                 + 1.26522 w_k1_MSFT_s102
                                 + 1.18535 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.14455 w_k1_ADM_s102
                                 + 1.07234 w_k1_AXP_s102
                                 + 1.33574 w_k1_DIS_s102
                                 + 1.36877 w_k1_FDX_s102
                                 + 0.941843 w_k1_IBM_s102
                                 + 1.32154 w_k1_INTC_s102
                                 + 1.10639 w_k1_KO_s102 + 1.22316 w_k1_MAT_s102
                                 + 1.31849 w_k1_MSFT_s102
                                 + 1.14619 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.818081 w_k1_ADM_s102
                                 + 0.607422 w_k1_AXP_s102
                                 + 0.626308 w_k1_DIS_s102
                                 + 0.777308 w_k1_FDX_s102
                                 + 0.653262 w_k1_IBM_s102
                                 + 0.749688 w_k1_INTC_s102
                                 + 0.674044 w_k1_KO_s102
                                 + 0.648019 w_k1_MAT_s102
                                 + 0.709136 w_k1_MSFT_s102
                                 + 0.930088 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.973099 w_k1_ADM_s102
                                 + 0.77569 w_k1_AXP_s102
                                 + 0.783084 w_k1_DIS_s102
                                 + 0.853562 w_k1_FDX_s102
                                 + 1.20867 w_k1_IBM_s102
                                 + 0.888574 w_k1_INTC_s102
                                 + 1.13238 w_k1_KO_s102 + 1.33985 w_k1_MAT_s102
                                 + 0.914317 w_k1_MSFT_s102
                                 + 1.0625 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.832889 w_k1_ADM_s102
                                 + 0.64717 w_k1_AXP_s102
                                 + 0.866348 w_k1_DIS_s102
                                 + 0.861682 w_k1_FDX_s102
                                 + 0.809703 w_k1_IBM_s102
                                 + 0.785131 w_k1_INTC_s102
                                 + 0.865101 w_k1_KO_s102
                                 + 0.812333 w_k1_MAT_s102
                                 + 0.936841 w_k1_MSFT_s102
                                 + 0.946101 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.0706 w_k1_ADM_s102
                                 + 1.31778 w_k1_AXP_s102
                                 + 1.14831 w_k1_DIS_s102
                                 + 1.14305 w_k1_FDX_s102 + 1.0652 w_k1_IBM_s102
                                 + 1.17087 w_k1_INTC_s102
                                 + 1.03479 w_k1_KO_s102 + 1.39818 w_k1_MAT_s102
                                 + 1.25074 w_k1_MSFT_s102
                                 + 1.11572 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.16963 w_k1_ADM_s102
                                 + 0.787066 w_k1_AXP_s102
                                 + 0.896505 w_k1_DIS_s102
                                 + 0.937679 w_k1_FDX_s102
                                 + 0.72196 w_k1_IBM_s102
                                 + 0.98465 w_k1_INTC_s102
                                 + 0.820136 w_k1_KO_s102
                                 + 0.639393 w_k1_MAT_s102
                                 + 1.01433 w_k1_MSFT_s102
                                 + 0.938301 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.31451 w_k1_ADM_s103
                                 + 1.48127 w_k1_AXP_s103
                                 + 1.74294 w_k1_DIS_s103
                                 + 1.41463 w_k1_FDX_s103
                                 + 1.20212 w_k1_IBM_s103
                                 + 1.42386 w_k1_INTC_s103
                                 + 1.43908 w_k1_KO_s103 + 1.39808 w_k1_MAT_s103
                                 + 1.3567 w_k1_MSFT_s103 + 1.2972 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.17734 w_k1_ADM_s103
                                 + 1.26752 w_k1_AXP_s103
                                 + 1.32428 w_k1_DIS_s103
                                 + 1.14666 w_k1_FDX_s103
                                 + 1.70538 w_k1_IBM_s103
                                 + 1.2957 w_k1_INTC_s103 + 1.40368 w_k1_KO_s103
                                 + 1.73991 w_k1_MAT_s103
                                 + 1.24327 w_k1_MSFT_s103
                                 + 1.21578 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.10232 w_k1_ADM_s103
                                 + 1.25208 w_k1_AXP_s103
                                 + 1.43149 w_k1_DIS_s103
                                 + 1.25381 w_k1_FDX_s103
                                 + 1.25028 w_k1_IBM_s103
                                 + 1.24552 w_k1_INTC_s103
                                 + 1.15122 w_k1_KO_s103 + 1.40204 w_k1_MAT_s103
                                 + 1.30844 w_k1_MSFT_s103
                                 + 1.24391 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.00777 w_k1_ADM_s103
                                 + 0.99627 w_k1_AXP_s103
                                 + 1.02664 w_k1_DIS_s103
                                 + 0.928753 w_k1_FDX_s103
                                 + 1.43947 w_k1_IBM_s103
                                 + 1.09998 w_k1_INTC_s103
                                 + 1.13329 w_k1_KO_s103 + 1.07499 w_k1_MAT_s103
                                 + 1.023 w_k1_MSFT_s103 + 1.07085 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.10965 w_k1_ADM_s103
                                 + 0.948243 w_k1_AXP_s103
                                 + 0.996389 w_k1_DIS_s103
                                 + 1.08332 w_k1_FDX_s103
                                 + 0.88872 w_k1_IBM_s103
                                 + 1.0942 w_k1_INTC_s103
                                 + 0.935972 w_k1_KO_s103
                                 + 1.09066 w_k1_MAT_s103
                                 + 1.05806 w_k1_MSFT_s103
                                 + 0.980781 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.941274 w_k1_ADM_s103
                                 + 0.733971 w_k1_AXP_s103
                                 + 0.74442 w_k1_DIS_s103
                                 + 0.841348 w_k1_FDX_s103
                                 + 0.895542 w_k1_IBM_s103
                                 + 1.00163 w_k1_INTC_s103
                                 + 0.891476 w_k1_KO_s103
                                 + 0.679638 w_k1_MAT_s103
                                 + 0.890616 w_k1_MSFT_s103
                                 + 0.966595 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.19304 w_k1_ADM_s103
                                 + 0.916988 w_k1_AXP_s103
                                 + 1.37268 w_k1_DIS_s103
                                 + 1.30244 w_k1_FDX_s103
                                 + 0.653614 w_k1_IBM_s103
                                 + 1.43785 w_k1_INTC_s103
                                 + 1.00897 w_k1_KO_s103 + 0.49744 w_k1_MAT_s103
                                 + 1.38353 w_k1_MSFT_s103
                                 + 1.06355 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.823643 w_k1_ADM_s103
                                 + 0.771387 w_k1_AXP_s103
                                 + 0.766747 w_k1_DIS_s103
                                 + 0.793667 w_k1_FDX_s103
                                 + 0.99484 w_k1_IBM_s103
                                 + 0.874591 w_k1_INTC_s103
                                 + 0.736056 w_k1_KO_s103
                                 + 0.771966 w_k1_MAT_s103
                                 + 0.894899 w_k1_MSFT_s103
                                 + 0.875773 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.726337 w_k1_ADM_s104
                                 + 0.930669 w_k1_AXP_s104
                                 + 0.972531 w_k1_DIS_s104
                                 + 0.833783 w_k1_FDX_s104
                                 + 1.37023 w_k1_IBM_s104
                                 + 0.950231 w_k1_INTC_s104
                                 + 1.07807 w_k1_KO_s104 + 1.42536 w_k1_MAT_s104
                                 + 0.904599 w_k1_MSFT_s104
                                 + 0.959119 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.799366 w_k1_ADM_s104
                                 + 0.846056 w_k1_AXP_s104
                                 + 0.925742 w_k1_DIS_s104
                                 + 0.890728 w_k1_FDX_s104
                                 + 0.985575 w_k1_IBM_s104
                                 + 0.958276 w_k1_INTC_s104
                                 + 1.00324 w_k1_KO_s104 + 1.27676 w_k1_MAT_s104
                                 + 0.893892 w_k1_MSFT_s104
                                 + 0.935228 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.02477 w_k1_ADM_s104
                                 + 1.35829 w_k1_AXP_s104
                                 + 1.40774 w_k1_DIS_s104
                                 + 1.15026 w_k1_FDX_s104
                                 + 1.49716 w_k1_IBM_s104
                                 + 1.08629 w_k1_INTC_s104
                                 + 1.32473 w_k1_KO_s104 + 2.1633 w_k1_MAT_s104
                                 + 1.21577 w_k1_MSFT_s104
                                 + 1.10669 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.15015 w_k1_ADM_s104
                                 + 0.994114 w_k1_AXP_s104
                                 + 0.97029 w_k1_DIS_s104
                                 + 1.04789 w_k1_FDX_s104
                                 + 1.03551 w_k1_IBM_s104
                                 + 0.995871 w_k1_INTC_s104
                                 + 1.03432 w_k1_KO_s104 + 1.14607 w_k1_MAT_s104
                                 + 1.04819 w_k1_MSFT_s104
                                 + 1.04931 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.793228 w_k1_ADM_s104
                                 + 1.11966 w_k1_AXP_s104
                                 + 0.923528 w_k1_DIS_s104
                                 + 1.05248 w_k1_FDX_s104
                                 + 0.836652 w_k1_IBM_s104
                                 + 1.02399 w_k1_INTC_s104
                                 + 0.933611 w_k1_KO_s104
                                 + 1.29791 w_k1_MAT_s104
                                 + 0.938564 w_k1_MSFT_s104
                                 + 0.91733 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.805762 w_k1_ADM_s104
                                 + 0.962938 w_k1_AXP_s104
                                 + 1.36901 w_k1_DIS_s104
                                 + 1.04058 w_k1_FDX_s104
                                 + 1.21601 w_k1_IBM_s104
                                 + 1.23911 w_k1_INTC_s104
                                 + 1.34424 w_k1_KO_s104
                                 + 0.873988 w_k1_MAT_s104
                                 + 1.15563 w_k1_MSFT_s104
                                 + 1.07445 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.753818 w_k1_ADM_s104
                                 + 0.709106 w_k1_AXP_s104
                                 + 0.67484 w_k1_DIS_s104
                                 + 0.846021 w_k1_FDX_s104
                                 + 0.704934 w_k1_IBM_s104
                                 + 0.895705 w_k1_INTC_s104
                                 + 0.76444 w_k1_KO_s104
                                 + 0.809826 w_k1_MAT_s104
                                 + 0.817838 w_k1_MSFT_s104
                                 + 0.803782 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.34723 w_k1_ADM_s104
                                 + 1.59616 w_k1_AXP_s104
                                 + 1.52145 w_k1_DIS_s104
                                 + 1.41897 w_k1_FDX_s104
                                 + 1.02889 w_k1_IBM_s104
                                 + 1.07981 w_k1_INTC_s104
                                 + 1.30813 w_k1_KO_s104 + 1.72405 w_k1_MAT_s104
                                 + 1.15142 w_k1_MSFT_s104
                                 + 1.19668 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.11013 w_k1_ADM_s105
                                 + 1.05259 w_k1_AXP_s105
                                 + 1.18785 w_k1_DIS_s105
                                 + 1.03904 w_k1_FDX_s105
                                 + 1.58316 w_k1_IBM_s105
                                 + 1.0128 w_k1_INTC_s105 + 1.25121 w_k1_KO_s105
                                 + 2.03688 w_k1_MAT_s105
                                 + 0.995902 w_k1_MSFT_s105
                                 + 1.19811 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.869075 w_k1_ADM_s105
                                 + 0.631976 w_k1_AXP_s105
                                 + 0.68981 w_k1_DIS_s105
                                 + 0.799366 w_k1_FDX_s105
                                 + 0.820135 w_k1_IBM_s105
                                 + 0.684285 w_k1_INTC_s105
                                 + 0.803188 w_k1_KO_s105
                                 + 0.925119 w_k1_MAT_s105
                                 + 0.820643 w_k1_MSFT_s105
                                 + 0.853022 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.00856 w_k1_ADM_s105
                                 + 0.782942 w_k1_AXP_s105
                                 + 1.14413 w_k1_DIS_s105
                                 + 0.950794 w_k1_FDX_s105
                                 + 1.02644 w_k1_IBM_s105
                                 + 0.957357 w_k1_INTC_s105
                                 + 1.13982 w_k1_KO_s105
                                 + 0.874107 w_k1_MAT_s105
                                 + 1.00906 w_k1_MSFT_s105
                                 + 0.927236 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.918539 w_k1_ADM_s105
                                 + 1.02998 w_k1_AXP_s105
                                 + 1.08534 w_k1_DIS_s105
                                 + 1.07865 w_k1_FDX_s105
                                 + 0.81746 w_k1_IBM_s105
                                 + 1.12552 w_k1_INTC_s105
                                 + 0.984133 w_k1_KO_s105
                                 + 0.974366 w_k1_MAT_s105
                                 + 1.2432 w_k1_MSFT_s105 + 1.02293 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.940496 w_k1_ADM_s105
                                 + 1.02526 w_k1_AXP_s105
                                 + 0.854841 w_k1_DIS_s105
                                 + 0.947124 w_k1_FDX_s105
                                 + 0.954992 w_k1_IBM_s105
                                 + 0.978044 w_k1_INTC_s105
                                 + 1.03183 w_k1_KO_s105 + 1.09228 w_k1_MAT_s105
                                 + 0.913711 w_k1_MSFT_s105
                                 + 1.00374 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.997712 w_k1_ADM_s105
                                 + 1.04239 w_k1_AXP_s105
                                 + 1.21761 w_k1_DIS_s105
                                 + 1.04476 w_k1_FDX_s105
                                 + 1.30267 w_k1_IBM_s105
                                 + 0.96884 w_k1_INTC_s105
                                 + 1.06791 w_k1_KO_s105 + 1.27155 w_k1_MAT_s105
                                 + 0.973373 w_k1_MSFT_s105
                                 + 1.10962 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.29333 w_k1_ADM_s105
                                 + 1.20827 w_k1_AXP_s105
                                 + 1.52502 w_k1_DIS_s105
                                 + 1.26241 w_k1_FDX_s105
                                 + 1.03257 w_k1_IBM_s105
                                 + 1.04065 w_k1_INTC_s105 + 1.2415 w_k1_KO_s105
                                 + 1.22695 w_k1_MAT_s105
                                 + 1.23984 w_k1_MSFT_s105
                                 + 1.06828 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.26746 w_k1_ADM_s105
                                 + 1.59996 w_k1_AXP_s105
                                 + 1.70215 w_k1_DIS_s105
                                 + 1.41709 w_k1_FDX_s105
                                 + 1.33993 w_k1_IBM_s105
                                 + 1.11857 w_k1_INTC_s105
                                 + 1.43973 w_k1_KO_s105 + 2.50253 w_k1_MAT_s105
                                 + 1.29925 w_k1_MSFT_s105
                                 + 1.27307 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.844784 w_k1_ADM_s106
                                 + 1.20481 w_k1_AXP_s106
                                 + 0.904543 w_k1_DIS_s106
                                 + 0.914908 w_k1_FDX_s106
                                 + 1.16236 w_k1_IBM_s106
                                 + 0.863796 w_k1_INTC_s106
                                 + 0.96351 w_k1_KO_s106 + 1.02545 w_k1_MAT_s106
                                 + 0.871148 w_k1_MSFT_s106
                                 + 0.957499 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.1777 w_k1_ADM_s106
                                 + 1.37069 w_k1_AXP_s106
                                 + 1.43041 w_k1_DIS_s106
                                 + 1.18864 w_k1_FDX_s106
                                 + 1.56641 w_k1_IBM_s106
                                 + 1.09954 w_k1_INTC_s106
                                 + 1.39075 w_k1_KO_s106 + 1.53199 w_k1_MAT_s106
                                 + 1.23593 w_k1_MSFT_s106
                                 + 1.14068 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.28338 w_k1_ADM_s106
                                 + 1.57832 w_k1_AXP_s106
                                 + 1.71082 w_k1_DIS_s106
                                 + 1.38285 w_k1_FDX_s106
                                 + 1.13771 w_k1_IBM_s106
                                 + 1.0626 w_k1_INTC_s106 + 1.09739 w_k1_KO_s106
                                 + 0.948368 w_k1_MAT_s106
                                 + 1.14293 w_k1_MSFT_s106
                                 + 1.13499 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.796697 w_k1_ADM_s106
                                 + 0.783545 w_k1_AXP_s106
                                 + 0.960482 w_k1_DIS_s106
                                 + 0.925766 w_k1_FDX_s106
                                 + 0.723912 w_k1_IBM_s106
                                 + 0.938409 w_k1_INTC_s106
                                 + 0.777323 w_k1_KO_s106
                                 + 0.699285 w_k1_MAT_s106
                                 + 0.921206 w_k1_MSFT_s106
                                 + 0.838428 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.00241 w_k1_ADM_s106
                                 + 1.21517 w_k1_AXP_s106
                                 + 1.07757 w_k1_DIS_s106
                                 + 1.11881 w_k1_FDX_s106
                                 + 0.86955 w_k1_IBM_s106
                                 + 1.01663 w_k1_INTC_s106
                                 + 0.805063 w_k1_KO_s106
                                 + 0.699963 w_k1_MAT_s106
                                 + 0.936279 w_k1_MSFT_s106
                                 + 1.02185 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.586612 w_k1_ADM_s106
                                 + 0.681554 w_k1_AXP_s106
                                 + 0.713383 w_k1_DIS_s106
                                 + 0.69778 w_k1_FDX_s106
                                 + 1.04247 w_k1_IBM_s106
                                 + 0.653384 w_k1_INTC_s106
                                 + 0.819008 w_k1_KO_s106
                                 + 0.956757 w_k1_MAT_s106
                                 + 0.686398 w_k1_MSFT_s106
                                 + 0.785078 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.952757 w_k1_ADM_s106
                                 + 1.08098 w_k1_AXP_s106
                                 + 1.47729 w_k1_DIS_s106
                                 + 1.08737 w_k1_FDX_s106
                                 + 1.26153 w_k1_IBM_s106
                                 + 0.989038 w_k1_INTC_s106
                                 + 1.1827 w_k1_KO_s106 + 0.912207 w_k1_MAT_s106
                                 + 1.08202 w_k1_MSFT_s106
                                 + 1.04291 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.978618 w_k1_ADM_s106
                                 + 1.00558 w_k1_AXP_s106 + 1.1565 w_k1_DIS_s106
                                 + 1.01005 w_k1_FDX_s106 + 1.0467 w_k1_IBM_s106
                                 + 0.88427 w_k1_INTC_s106
                                 + 0.933348 w_k1_KO_s106
                                 + 1.07313 w_k1_MAT_s106
                                 + 0.914841 w_k1_MSFT_s106
                                 + 1.04332 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13533 w_k1_ADM_s107
                                 + 1.09869 w_k1_AXP_s107
                                 + 0.907242 w_k1_DIS_s107
                                 + 1.04799 w_k1_FDX_s107
                                 + 0.905987 w_k1_IBM_s107
                                 + 0.938256 w_k1_INTC_s107
                                 + 0.950988 w_k1_KO_s107
                                 + 1.01941 w_k1_MAT_s107
                                 + 0.930649 w_k1_MSFT_s107
                                 + 0.917692 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.98883 w_k1_ADM_s107
                                 + 0.791826 w_k1_AXP_s107
                                 + 0.945624 w_k1_DIS_s107
                                 + 1.06469 w_k1_FDX_s107
                                 + 0.663526 w_k1_IBM_s107
                                 + 0.953414 w_k1_INTC_s107
                                 + 0.844786 w_k1_KO_s107
                                 + 0.727167 w_k1_MAT_s107
                                 + 1.05329 w_k1_MSFT_s107
                                 + 0.923538 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.06968 w_k1_ADM_s107
                                 + 1.07583 w_k1_AXP_s107
                                 + 1.10844 w_k1_DIS_s107
                                 + 1.01716 w_k1_FDX_s107
                                 + 1.41055 w_k1_IBM_s107
                                 + 0.924826 w_k1_INTC_s107
                                 + 1.14816 w_k1_KO_s107 + 1.897 w_k1_MAT_s107
                                 + 0.860498 w_k1_MSFT_s107
                                 + 1.13172 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.50897 w_k1_ADM_s107
                                 + 1.47397 w_k1_AXP_s107
                                 + 1.38336 w_k1_DIS_s107
                                 + 1.29513 w_k1_FDX_s107
                                 + 1.09279 w_k1_IBM_s107
                                 + 1.0722 w_k1_INTC_s107 + 1.35291 w_k1_KO_s107
                                 + 1.52438 w_k1_MAT_s107
                                 + 1.13454 w_k1_MSFT_s107
                                 + 1.18834 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.06957 w_k1_ADM_s107
                                 + 0.919621 w_k1_AXP_s107
                                 + 0.747005 w_k1_DIS_s107
                                 + 0.829449 w_k1_FDX_s107
                                 + 1.13248 w_k1_IBM_s107
                                 + 0.741009 w_k1_INTC_s107
                                 + 1.10131 w_k1_KO_s107 + 1.23486 w_k1_MAT_s107
                                 + 0.816872 w_k1_MSFT_s107
                                 + 1.0519 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.852818 w_k1_ADM_s107
                                 + 0.60503 w_k1_AXP_s107
                                 + 0.61668 w_k1_DIS_s107
                                 + 0.694863 w_k1_FDX_s107
                                 + 0.732836 w_k1_IBM_s107
                                 + 0.712761 w_k1_INTC_s107
                                 + 0.690684 w_k1_KO_s107
                                 + 0.840003 w_k1_MAT_s107
                                 + 0.786777 w_k1_MSFT_s107
                                 + 0.809704 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.794404 w_k1_ADM_s107
                                 + 0.790167 w_k1_AXP_s107
                                 + 0.647647 w_k1_DIS_s107
                                 + 0.777104 w_k1_FDX_s107
                                 + 0.842676 w_k1_IBM_s107
                                 + 0.816631 w_k1_INTC_s107
                                 + 0.795167 w_k1_KO_s107
                                 + 0.967957 w_k1_MAT_s107
                                 + 0.747067 w_k1_MSFT_s107
                                 + 0.880605 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.75455 w_k1_ADM_s107
                                 + 1.84326 w_k1_AXP_s107
                                 + 1.87919 w_k1_DIS_s107
                                 + 1.65232 w_k1_FDX_s107
                                 + 1.45018 w_k1_IBM_s107
                                 + 1.22176 w_k1_INTC_s107
                                 + 1.53581 w_k1_KO_s107 + 1.94156 w_k1_MAT_s107
                                 + 1.27658 w_k1_MSFT_s107 + 1.3982 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09091 w_k1_ADM_s110
                                 + 0.983999 w_k1_AXP_s110
                                 + 1.20401 w_k1_DIS_s110
                                 + 0.971651 w_k1_FDX_s110
                                 + 1.34043 w_k1_IBM_s110
                                 + 0.999243 w_k1_INTC_s110
                                 + 1.32112 w_k1_KO_s110 + 1.11585 w_k1_MAT_s110
                                 + 1.1913 w_k1_MSFT_s110 + 1.05099 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.808767 w_k1_ADM_s110
                                 + 0.644296 w_k1_AXP_s110
                                 + 0.646688 w_k1_DIS_s110
                                 + 0.734082 w_k1_FDX_s110
                                 + 0.942249 w_k1_IBM_s110
                                 + 0.700261 w_k1_INTC_s110
                                 + 0.884257 w_k1_KO_s110
                                 + 1.14925 w_k1_MAT_s110
                                 + 0.785842 w_k1_MSFT_s110
                                 + 0.93168 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.912142 w_k1_ADM_s110
                                 + 0.749739 w_k1_AXP_s110
                                 + 0.937216 w_k1_DIS_s110
                                 + 0.934798 w_k1_FDX_s110
                                 + 0.828467 w_k1_IBM_s110
                                 + 0.88027 w_k1_INTC_s110
                                 + 0.972348 w_k1_KO_s110
                                 + 0.968836 w_k1_MAT_s110
                                 + 1.04009 w_k1_MSFT_s110
                                 + 1.01753 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.66172 w_k1_ADM_s110
                                 + 1.49722 w_k1_AXP_s110
                                 + 1.56084 w_k1_DIS_s110
                                 + 1.37165 w_k1_FDX_s110
                                 + 1.24459 w_k1_IBM_s110
                                 + 1.18018 w_k1_INTC_s110
                                 + 1.40483 w_k1_KO_s110 + 1.37732 w_k1_MAT_s110
                                 + 1.34422 w_k1_MSFT_s110
                                 + 1.39508 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.889495 w_k1_ADM_s110
                                 + 0.881353 w_k1_AXP_s110
                                 + 0.7154 w_k1_DIS_s110
                                 + 0.813731 w_k1_FDX_s110
                                 + 0.776683 w_k1_IBM_s110
                                 + 0.94123 w_k1_INTC_s110
                                 + 0.827026 w_k1_KO_s110
                                 + 1.09149 w_k1_MAT_s110
                                 + 0.849864 w_k1_MSFT_s110
                                 + 0.883672 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.03472 w_k1_ADM_s110
                                 + 1.0831 w_k1_AXP_s110
                                 + 0.900643 w_k1_DIS_s110
                                 + 1.01621 w_k1_FDX_s110
                                 + 0.874338 w_k1_IBM_s110
                                 + 0.984962 w_k1_INTC_s110
                                 + 0.880132 w_k1_KO_s110
                                 + 1.00789 w_k1_MAT_s110 + 1.008 w_k1_MSFT_s110
                                 + 0.925522 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.33592 w_k1_ADM_s110
                                 + 1.1185 w_k1_AXP_s110 + 1.20969 w_k1_DIS_s110
                                 + 1.14939 w_k1_FDX_s110
                                 + 1.00802 w_k1_IBM_s110
                                 + 1.01875 w_k1_INTC_s110
                                 + 1.16413 w_k1_KO_s110 + 1.05389 w_k1_MAT_s110
                                 + 1.20528 w_k1_MSFT_s110
                                 + 0.980839 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.41142 w_k1_ADM_s110
                                 + 0.836811 w_k1_AXP_s110
                                 + 1.13649 w_k1_DIS_s110
                                 + 1.19819 w_k1_FDX_s110
                                 + 0.738113 w_k1_IBM_s110
                                 + 0.924724 w_k1_INTC_s110
                                 + 1.0529 w_k1_KO_s110 + 0.741273 w_k1_MAT_s110
                                 + 1.21778 w_k1_MSFT_s110 + 1.0467 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.735233 w_k1_ADM_s111
                                 + 0.995973 w_k1_AXP_s111
                                 + 0.984653 w_k1_DIS_s111
                                 + 0.958888 w_k1_FDX_s111
                                 + 0.939397 w_k1_IBM_s111
                                 + 1.09939 w_k1_INTC_s111
                                 + 0.970295 w_k1_KO_s111
                                 + 1.16387 w_k1_MAT_s111
                                 + 1.0649 w_k1_MSFT_s111
                                 + 0.969066 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.16736 w_k1_ADM_s111
                                 + 1.40816 w_k1_AXP_s111
                                 + 1.29375 w_k1_DIS_s111
                                 + 1.24441 w_k1_FDX_s111
                                 + 1.04329 w_k1_IBM_s111
                                 + 1.35475 w_k1_INTC_s111 + 1.1645 w_k1_KO_s111
                                 + 1.26628 w_k1_MAT_s111
                                 + 1.30717 w_k1_MSFT_s111
                                 + 1.27854 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.645626 w_k1_ADM_s111
                                 + 0.967859 w_k1_AXP_s111
                                 + 1.03542 w_k1_DIS_s111
                                 + 0.934677 w_k1_FDX_s111
                                 + 1.16246 w_k1_IBM_s111
                                 + 1.21295 w_k1_INTC_s111
                                 + 1.02195 w_k1_KO_s111 + 1.38043 w_k1_MAT_s111
                                 + 1.18744 w_k1_MSFT_s111
                                 + 0.989408 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.623045 w_k1_ADM_s111
                                 + 0.800581 w_k1_AXP_s111
                                 + 0.67362 w_k1_DIS_s111
                                 + 0.766879 w_k1_FDX_s111
                                 + 0.75003 w_k1_IBM_s111
                                 + 0.713968 w_k1_INTC_s111
                                 + 0.803237 w_k1_KO_s111
                                 + 0.792134 w_k1_MAT_s111
                                 + 0.767751 w_k1_MSFT_s111
                                 + 0.908379 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.06592 w_k1_ADM_s111
                                 + 1.41485 w_k1_AXP_s111
                                 + 1.27154 w_k1_DIS_s111
                                 + 1.18112 w_k1_FDX_s111
                                 + 1.59347 w_k1_IBM_s111
                                 + 1.15479 w_k1_INTC_s111
                                 + 1.31069 w_k1_KO_s111 + 1.63949 w_k1_MAT_s111
                                 + 1.17447 w_k1_MSFT_s111
                                 + 1.18624 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.725205 w_k1_ADM_s111
                                 + 0.931859 w_k1_AXP_s111
                                 + 0.709079 w_k1_DIS_s111
                                 + 0.897839 w_k1_FDX_s111
                                 + 0.897774 w_k1_IBM_s111
                                 + 0.966616 w_k1_INTC_s111
                                 + 0.82756 w_k1_KO_s111 + 1.00917 w_k1_MAT_s111
                                 + 0.950036 w_k1_MSFT_s111
                                 + 0.950769 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.36151 w_k1_ADM_s111
                                 + 1.76521 w_k1_AXP_s111
                                 + 1.61259 w_k1_DIS_s111
                                 + 1.38134 w_k1_FDX_s111 + 1.0574 w_k1_IBM_s111
                                 + 1.23237 w_k1_INTC_s111
                                 + 1.15373 w_k1_KO_s111 + 1.44328 w_k1_MAT_s111
                                 + 1.43073 w_k1_MSFT_s111
                                 + 1.34633 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.24071 w_k1_ADM_s111
                                 + 1.05525 w_k1_AXP_s111
                                 + 1.16832 w_k1_DIS_s111 + 1.2337 w_k1_FDX_s111
                                 + 0.88244 w_k1_IBM_s111
                                 + 1.17969 w_k1_INTC_s111
                                 + 1.07539 w_k1_KO_s111
                                 + 0.697984 w_k1_MAT_s111
                                 + 1.25468 w_k1_MSFT_s111
                                 + 1.15321 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.34493 w_k1_ADM_s112
                                 + 1.14505 w_k1_AXP_s112
                                 + 1.19409 w_k1_DIS_s112
                                 + 1.18622 w_k1_FDX_s112
                                 + 1.02549 w_k1_IBM_s112
                                 + 0.947542 w_k1_INTC_s112
                                 + 1.19842 w_k1_KO_s112 + 1.01862 w_k1_MAT_s112
                                 + 1.04921 w_k1_MSFT_s112
                                 + 1.06603 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.810694 w_k1_ADM_s112
                                 + 0.725809 w_k1_AXP_s112
                                 + 0.809294 w_k1_DIS_s112
                                 + 0.879513 w_k1_FDX_s112
                                 + 0.93288 w_k1_IBM_s112
                                 + 0.667783 w_k1_INTC_s112
                                 + 0.916297 w_k1_KO_s112
                                 + 0.757403 w_k1_MAT_s112
                                 + 0.838203 w_k1_MSFT_s112
                                 + 0.878253 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.96894 w_k1_ADM_s112
                                 + 0.997788 w_k1_AXP_s112
                                 + 0.759416 w_k1_DIS_s112
                                 + 0.919826 w_k1_FDX_s112
                                 + 1.32858 w_k1_IBM_s112
                                 + 0.922981 w_k1_INTC_s112
                                 + 1.16106 w_k1_KO_s112 + 1.38755 w_k1_MAT_s112
                                 + 0.939581 w_k1_MSFT_s112
                                 + 1.00092 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.23305 w_k1_ADM_s112
                                 + 1.12091 w_k1_AXP_s112
                                 + 1.12195 w_k1_DIS_s112
                                 + 1.02466 w_k1_FDX_s112
                                 + 1.53157 w_k1_IBM_s112
                                 + 0.932944 w_k1_INTC_s112
                                 + 1.48396 w_k1_KO_s112 + 1.30938 w_k1_MAT_s112
                                 + 0.9691 w_k1_MSFT_s112 + 1.22151 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.5949 w_k1_ADM_s112
                                 + 1.33534 w_k1_AXP_s112
                                 + 1.68127 w_k1_DIS_s112
                                 + 1.47971 w_k1_FDX_s112
                                 + 1.41186 w_k1_IBM_s112
                                 + 1.08372 w_k1_INTC_s112
                                 + 1.44547 w_k1_KO_s112 + 1.27895 w_k1_MAT_s112
                                 + 1.25284 w_k1_MSFT_s112
                                 + 1.29842 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.14353 w_k1_ADM_s112
                                 + 1.10208 w_k1_AXP_s112
                                 + 1.50753 w_k1_DIS_s112
                                 + 1.48412 w_k1_FDX_s112
                                 + 0.919514 w_k1_IBM_s112
                                 + 0.992404 w_k1_INTC_s112
                                 + 1.26505 w_k1_KO_s112
                                 + 0.856633 w_k1_MAT_s112
                                 + 1.24325 w_k1_MSFT_s112
                                 + 1.17618 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.21212 w_k1_ADM_s112
                                 + 0.822622 w_k1_AXP_s112
                                 + 0.889289 w_k1_DIS_s112
                                 + 1.05098 w_k1_FDX_s112
                                 + 0.721383 w_k1_IBM_s112
                                 + 0.802521 w_k1_INTC_s112
                                 + 0.889867 w_k1_KO_s112
                                 + 0.692349 w_k1_MAT_s112
                                 + 0.910705 w_k1_MSFT_s112
                                 + 0.949455 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.07383 w_k1_ADM_s112
                                 + 1.15317 w_k1_AXP_s112
                                 + 0.968363 w_k1_DIS_s112
                                 + 1.12515 w_k1_FDX_s112
                                 + 1.23076 w_k1_IBM_s112
                                 + 0.889785 w_k1_INTC_s112
                                 + 1.3114 w_k1_KO_s112 + 1.07348 w_k1_MAT_s112
                                 + 1.03218 w_k1_MSFT_s112
                                 + 1.06034 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.835158 w_k1_ADM_s113
                                 + 1.57947 w_k1_AXP_s113
                                 + 1.46245 w_k1_DIS_s113 + 1.1369 w_k1_FDX_s113
                                 + 1.54266 w_k1_IBM_s113
                                 + 1.15744 w_k1_INTC_s113
                                 + 1.61773 w_k1_KO_s113 + 2.1363 w_k1_MAT_s113
                                 + 1.20737 w_k1_MSFT_s113
                                 + 1.26822 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.1516 w_k1_ADM_s113
                                 + 1.22109 w_k1_AXP_s113
                                 + 1.11273 w_k1_DIS_s113
                                 + 1.29718 w_k1_FDX_s113
                                 + 0.772914 w_k1_IBM_s113
                                 + 0.918451 w_k1_INTC_s113
                                 + 0.928726 w_k1_KO_s113
                                 + 0.746957 w_k1_MAT_s113
                                 + 1.05291 w_k1_MSFT_s113
                                 + 0.911654 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.642201 w_k1_ADM_s113
                                 + 0.770574 w_k1_AXP_s113
                                 + 0.551501 w_k1_DIS_s113
                                 + 0.743256 w_k1_FDX_s113
                                 + 0.751039 w_k1_IBM_s113
                                 + 0.737718 w_k1_INTC_s113
                                 + 0.756569 w_k1_KO_s113
                                 + 0.708256 w_k1_MAT_s113
                                 + 0.736451 w_k1_MSFT_s113
                                 + 0.763047 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.828381 w_k1_ADM_s113
                                 + 0.747258 w_k1_AXP_s113
                                 + 0.911844 w_k1_DIS_s113
                                 + 0.743451 w_k1_FDX_s113
                                 + 0.916177 w_k1_IBM_s113
                                 + 0.649119 w_k1_INTC_s113
                                 + 0.932277 w_k1_KO_s113
                                 + 0.810902 w_k1_MAT_s113
                                 + 0.810163 w_k1_MSFT_s113
                                 + 0.752089 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.834944 w_k1_ADM_s113
                                 + 0.871419 w_k1_AXP_s113
                                 + 0.874752 w_k1_DIS_s113
                                 + 0.988246 w_k1_FDX_s113
                                 + 0.855526 w_k1_IBM_s113
                                 + 0.705158 w_k1_INTC_s113
                                 + 0.985738 w_k1_KO_s113
                                 + 1.06748 w_k1_MAT_s113
                                 + 0.872771 w_k1_MSFT_s113
                                 + 0.943769 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.735658 w_k1_ADM_s113
                                 + 1.39386 w_k1_AXP_s113
                                 + 0.898179 w_k1_DIS_s113
                                 + 0.944676 w_k1_FDX_s113
                                 + 1.15248 w_k1_IBM_s113
                                 + 0.682617 w_k1_INTC_s113
                                 + 0.998408 w_k1_KO_s113
                                 + 1.63248 w_k1_MAT_s113
                                 + 0.833121 w_k1_MSFT_s113
                                 + 0.850087 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.05316 w_k1_ADM_s113
                                 + 0.967877 w_k1_AXP_s113
                                 + 1.18495 w_k1_DIS_s113
                                 + 1.10544 w_k1_FDX_s113
                                 + 1.09343 w_k1_IBM_s113
                                 + 1.07382 w_k1_INTC_s113
                                 + 1.09425 w_k1_KO_s113 + 1.19809 w_k1_MAT_s113
                                 + 1.05338 w_k1_MSFT_s113
                                 + 0.892257 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.787636 w_k1_ADM_s113
                                 + 0.667189 w_k1_AXP_s113
                                 + 0.600404 w_k1_DIS_s113
                                 + 0.823447 w_k1_FDX_s113
                                 + 0.626846 w_k1_IBM_s113
                                 + 0.878165 w_k1_INTC_s113
                                 + 0.758712 w_k1_KO_s113
                                 + 0.779737 w_k1_MAT_s113
                                 + 0.865408 w_k1_MSFT_s113
                                 + 0.861101 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.14827 w_k1_ADM_s114
                                 + 0.96578 w_k1_AXP_s114
                                 + 0.839802 w_k1_DIS_s114
                                 + 0.925367 w_k1_FDX_s114
                                 + 1.18598 w_k1_IBM_s114
                                 + 1.07064 w_k1_INTC_s114
                                 + 1.06221 w_k1_KO_s114 + 1.18567 w_k1_MAT_s114
                                 + 0.969024 w_k1_MSFT_s114
                                 + 1.07606 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.20613 w_k1_ADM_s114
                                 + 0.927705 w_k1_AXP_s114
                                 + 0.806669 w_k1_DIS_s114
                                 + 1.08254 w_k1_FDX_s114
                                 + 0.743489 w_k1_IBM_s114
                                 + 0.91186 w_k1_INTC_s114
                                 + 0.793184 w_k1_KO_s114
                                 + 0.99588 w_k1_MAT_s114
                                 + 0.938737 w_k1_MSFT_s114
                                 + 1.04308 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.0593 w_k1_ADM_s114
                                 + 0.971004 w_k1_AXP_s114
                                 + 1.04858 w_k1_DIS_s114
                                 + 1.08545 w_k1_FDX_s114
                                 + 0.944391 w_k1_IBM_s114
                                 + 1.01167 w_k1_INTC_s114
                                 + 1.07291 w_k1_KO_s114
                                 + 0.943827 w_k1_MAT_s114
                                 + 1.06963 w_k1_MSFT_s114
                                 + 1.18431 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.12061 w_k1_ADM_s114
                                 + 1.10855 w_k1_AXP_s114
                                 + 1.23241 w_k1_DIS_s114
                                 + 1.06295 w_k1_FDX_s114
                                 + 1.02534 w_k1_IBM_s114
                                 + 1.00038 w_k1_INTC_s114
                                 + 1.16722 w_k1_KO_s114 + 1.04276 w_k1_MAT_s114
                                 + 1.08939 w_k1_MSFT_s114
                                 + 1.24198 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.01081 w_k1_ADM_s114
                                 + 1.20596 w_k1_AXP_s114
                                 + 1.27872 w_k1_DIS_s114
                                 + 1.09825 w_k1_FDX_s114
                                 + 1.54251 w_k1_IBM_s114
                                 + 1.15563 w_k1_INTC_s114
                                 + 1.29727 w_k1_KO_s114 + 1.48226 w_k1_MAT_s114
                                 + 1.14535 w_k1_MSFT_s114
                                 + 1.23681 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.35629 w_k1_ADM_s114
                                 + 1.27019 w_k1_AXP_s114
                                 + 1.32355 w_k1_DIS_s114
                                 + 1.34023 w_k1_FDX_s114
                                 + 0.841466 w_k1_IBM_s114
                                 + 1.1778 w_k1_INTC_s114 + 1.22875 w_k1_KO_s114
                                 + 0.959975 w_k1_MAT_s114
                                 + 1.14654 w_k1_MSFT_s114 + 1.2429 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.54435 w_k1_ADM_s114
                                 + 1.58421 w_k1_AXP_s114
                                 + 1.74816 w_k1_DIS_s114
                                 + 1.74073 w_k1_FDX_s114
                                 + 1.12502 w_k1_IBM_s114
                                 + 1.4663 w_k1_INTC_s114 + 1.49979 w_k1_KO_s114
                                 + 1.31556 w_k1_MAT_s114
                                 + 1.34484 w_k1_MSFT_s114
                                 + 1.61958 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.915805 w_k1_ADM_s114
                                 + 0.639869 w_k1_AXP_s114
                                 + 0.596493 w_k1_DIS_s114
                                 + 0.728016 w_k1_FDX_s114
                                 + 0.795825 w_k1_IBM_s114
                                 + 0.892798 w_k1_INTC_s114
                                 + 0.906831 w_k1_KO_s114
                                 + 0.53008 w_k1_MAT_s114
                                 + 0.802823 w_k1_MSFT_s114
                                 + 0.84925 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.937397 w_k1_ADM_s115
                                 + 0.938838 w_k1_AXP_s115
                                 + 0.792191 w_k1_DIS_s115
                                 + 0.845571 w_k1_FDX_s115 + 1.109 w_k1_IBM_s115
                                 + 0.774406 w_k1_INTC_s115
                                 + 0.808479 w_k1_KO_s115
                                 + 0.681292 w_k1_MAT_s115
                                 + 0.765936 w_k1_MSFT_s115
                                 + 0.907787 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.05713 w_k1_ADM_s115
                                 + 1.14469 w_k1_AXP_s115
                                 + 1.24146 w_k1_DIS_s115
                                 + 1.24874 w_k1_FDX_s115 + 1.0692 w_k1_IBM_s115
                                 + 1.12259 w_k1_INTC_s115
                                 + 1.05008 w_k1_KO_s115 + 1.05567 w_k1_MAT_s115
                                 + 1.2101 w_k1_MSFT_s115 + 1.0954 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.27167 w_k1_ADM_s115
                                 + 1.71756 w_k1_AXP_s115
                                 + 1.84492 w_k1_DIS_s115
                                 + 1.44718 w_k1_FDX_s115
                                 + 1.16263 w_k1_IBM_s115
                                 + 1.09185 w_k1_INTC_s115
                                 + 1.17847 w_k1_KO_s115 + 1.20773 w_k1_MAT_s115
                                 + 1.19503 w_k1_MSFT_s115
                                 + 1.16419 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.933536 w_k1_ADM_s115
                                 + 0.915983 w_k1_AXP_s115
                                 + 0.867179 w_k1_DIS_s115
                                 + 0.98499 w_k1_FDX_s115
                                 + 0.686801 w_k1_IBM_s115
                                 + 0.948208 w_k1_INTC_s115
                                 + 0.7287 w_k1_KO_s115 + 0.724198 w_k1_MAT_s115
                                 + 0.899897 w_k1_MSFT_s115
                                 + 0.960189 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.956329 w_k1_ADM_s115
                                 + 1.13826 w_k1_AXP_s115
                                 + 0.855613 w_k1_DIS_s115
                                 + 0.961277 w_k1_FDX_s115
                                 + 1.39247 w_k1_IBM_s115
                                 + 0.972917 w_k1_INTC_s115
                                 + 1.01062 w_k1_KO_s115 + 1.08047 w_k1_MAT_s115
                                 + 0.761732 w_k1_MSFT_s115
                                 + 1.06584 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.958798 w_k1_ADM_s115
                                 + 1.01418 w_k1_AXP_s115
                                 + 0.758807 w_k1_DIS_s115
                                 + 0.897357 w_k1_FDX_s115
                                 + 0.878247 w_k1_IBM_s115
                                 + 0.872604 w_k1_INTC_s115
                                 + 0.853967 w_k1_KO_s115
                                 + 0.780655 w_k1_MAT_s115
                                 + 0.997343 w_k1_MSFT_s115
                                 + 0.868155 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.99074 w_k1_ADM_s115
                                 + 1.38305 w_k1_AXP_s115 + 1.3624 w_k1_DIS_s115
                                 + 1.15733 w_k1_FDX_s115
                                 + 1.72621 w_k1_IBM_s115
                                 + 1.04971 w_k1_INTC_s115
                                 + 1.20862 w_k1_KO_s115 + 2.04837 w_k1_MAT_s115
                                 + 0.991289 w_k1_MSFT_s115
                                 + 1.15372 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.49873 w_k1_ADM_s115
                                 + 1.43221 w_k1_AXP_s115
                                 + 1.67312 w_k1_DIS_s115
                                 + 1.27089 w_k1_FDX_s115
                                 + 1.44227 w_k1_IBM_s115
                                 + 1.15324 w_k1_INTC_s115
                                 + 1.28698 w_k1_KO_s115 + 1.13385 w_k1_MAT_s115
                                 + 1.27213 w_k1_MSFT_s115 + 1.2412 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.969293 w_k1_ADM_s116
                                 + 1.08702 w_k1_AXP_s116
                                 + 1.11598 w_k1_DIS_s116
                                 + 0.897358 w_k1_FDX_s116
                                 + 1.70569 w_k1_IBM_s116
                                 + 0.975428 w_k1_INTC_s116
                                 + 1.3123 w_k1_KO_s116 + 1.45895 w_k1_MAT_s116
                                 + 0.904689 w_k1_MSFT_s116
                                 + 1.08708 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.811255 w_k1_ADM_s116
                                 + 1.19924 w_k1_AXP_s116
                                 + 1.05261 w_k1_DIS_s116
                                 + 1.23692 w_k1_FDX_s116
                                 + 0.661717 w_k1_IBM_s116
                                 + 1.34797 w_k1_INTC_s116
                                 + 0.906969 w_k1_KO_s116
                                 + 0.842144 w_k1_MAT_s116
                                 + 1.23395 w_k1_MSFT_s116
                                 + 1.03397 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.27306 w_k1_ADM_s116
                                 + 1.58659 w_k1_AXP_s116
                                 + 1.27155 w_k1_DIS_s116
                                 + 1.23779 w_k1_FDX_s116
                                 + 1.15119 w_k1_IBM_s116
                                 + 1.20421 w_k1_INTC_s116
                                 + 1.03263 w_k1_KO_s116
                                 + 0.983286 w_k1_MAT_s116
                                 + 1.05954 w_k1_MSFT_s116
                                 + 1.09088 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.785012 w_k1_ADM_s116
                                 + 0.734271 w_k1_AXP_s116
                                 + 0.731667 w_k1_DIS_s116
                                 + 0.978609 w_k1_FDX_s116
                                 + 0.727193 w_k1_IBM_s116
                                 + 0.857986 w_k1_INTC_s116
                                 + 0.762669 w_k1_KO_s116
                                 + 0.706028 w_k1_MAT_s116
                                 + 0.833812 w_k1_MSFT_s116
                                 + 0.843393 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.984207 w_k1_ADM_s116
                                 + 0.840642 w_k1_AXP_s116
                                 + 0.898504 w_k1_DIS_s116
                                 + 0.994426 w_k1_FDX_s116
                                 + 0.873491 w_k1_IBM_s116
                                 + 0.910529 w_k1_INTC_s116
                                 + 0.873276 w_k1_KO_s116
                                 + 0.818039 w_k1_MAT_s116
                                 + 0.958831 w_k1_MSFT_s116
                                 + 0.962327 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.522491 w_k1_ADM_s116
                                 + 0.544693 w_k1_AXP_s116
                                 + 0.563979 w_k1_DIS_s116
                                 + 0.695472 w_k1_FDX_s116
                                 + 0.673199 w_k1_IBM_s116
                                 + 0.855454 w_k1_INTC_s116
                                 + 0.638191 w_k1_KO_s116
                                 + 0.516061 w_k1_MAT_s116
                                 + 0.618773 w_k1_MSFT_s116
                                 + 0.70086 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.13162 w_k1_ADM_s116
                                 + 0.993175 w_k1_AXP_s116 + 1.214 w_k1_DIS_s116
                                 + 1.22522 w_k1_FDX_s116
                                 + 0.988229 w_k1_IBM_s116
                                 + 1.23756 w_k1_INTC_s116
                                 + 0.907895 w_k1_KO_s116
                                 + 0.870651 w_k1_MAT_s116
                                 + 1.1347 w_k1_MSFT_s116 + 1.036 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.795597 w_k1_ADM_s116
                                 + 0.759938 w_k1_AXP_s116
                                 + 0.671903 w_k1_DIS_s116
                                 + 0.731273 w_k1_FDX_s116
                                 + 1.08732 w_k1_IBM_s116
                                 + 0.863812 w_k1_INTC_s116
                                 + 0.930471 w_k1_KO_s116
                                 + 0.741922 w_k1_MAT_s116
                                 + 0.782087 w_k1_MSFT_s116
                                 + 0.842544 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.789419 w_k1_ADM_s117
                                 + 0.848849 w_k1_AXP_s117 + 1.147 w_k1_DIS_s117
                                 + 0.870967 w_k1_FDX_s117
                                 + 1.06645 w_k1_IBM_s117
                                 + 0.947128 w_k1_INTC_s117
                                 + 1.15071 w_k1_KO_s117
                                 + 0.899191 w_k1_MAT_s117
                                 + 1.00858 w_k1_MSFT_s117
                                 + 1.04539 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.896744 w_k1_ADM_s117
                                 + 1.07471 w_k1_AXP_s117
                                 + 1.17411 w_k1_DIS_s117
                                 + 1.06736 w_k1_FDX_s117
                                 + 1.16047 w_k1_IBM_s117
                                 + 1.12406 w_k1_INTC_s117
                                 + 1.26719 w_k1_KO_s117 + 1.00574 w_k1_MAT_s117
                                 + 0.933838 w_k1_MSFT_s117
                                 + 1.09708 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.13207 w_k1_ADM_s117
                                 + 1.45211 w_k1_AXP_s117
                                 + 1.31346 w_k1_DIS_s117
                                 + 1.35214 w_k1_FDX_s117
                                 + 0.958308 w_k1_IBM_s117
                                 + 1.3394 w_k1_INTC_s117 + 1.093 w_k1_KO_s117
                                 + 0.882196 w_k1_MAT_s117
                                 + 1.21163 w_k1_MSFT_s117
                                 + 1.10542 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.10927 w_k1_ADM_s117
                                 + 1.19518 w_k1_AXP_s117
                                 + 1.84286 w_k1_DIS_s117
                                 + 1.41957 w_k1_FDX_s117
                                 + 1.23125 w_k1_IBM_s117
                                 + 1.21073 w_k1_INTC_s117
                                 + 1.61751 w_k1_KO_s117 + 1.36386 w_k1_MAT_s117
                                 + 1.38503 w_k1_MSFT_s117
                                 + 1.34182 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.668623 w_k1_ADM_s117
                                 + 0.85527 w_k1_AXP_s117
                                 + 0.889073 w_k1_DIS_s117
                                 + 0.890823 w_k1_FDX_s117
                                 + 0.821878 w_k1_IBM_s117
                                 + 0.654938 w_k1_INTC_s117
                                 + 0.85394 w_k1_KO_s117
                                 + 0.939641 w_k1_MAT_s117
                                 + 0.828438 w_k1_MSFT_s117
                                 + 0.976257 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.881295 w_k1_ADM_s117
                                 + 1.29681 w_k1_AXP_s117
                                 + 1.41248 w_k1_DIS_s117
                                 + 0.952606 w_k1_FDX_s117
                                 + 1.62893 w_k1_IBM_s117
                                 + 0.878492 w_k1_INTC_s117
                                 + 1.39842 w_k1_KO_s117 + 1.43104 w_k1_MAT_s117
                                 + 1.06158 w_k1_MSFT_s117
                                 + 1.16458 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.541216 w_k1_ADM_s117
                                 + 0.671371 w_k1_AXP_s117
                                 + 0.639532 w_k1_DIS_s117
                                 + 0.695032 w_k1_FDX_s117
                                 + 0.762319 w_k1_IBM_s117
                                 + 0.638232 w_k1_INTC_s117
                                 + 0.653791 w_k1_KO_s117
                                 + 0.583637 w_k1_MAT_s117
                                 + 0.58576 w_k1_MSFT_s117
                                 + 0.683061 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.868393 w_k1_ADM_s117
                                 + 1.05119 w_k1_AXP_s117
                                 + 0.77576 w_k1_DIS_s117
                                 + 0.907905 w_k1_FDX_s117
                                 + 1.23144 w_k1_IBM_s117
                                 + 0.800289 w_k1_INTC_s117
                                 + 0.989152 w_k1_KO_s117
                                 + 1.14336 w_k1_MAT_s117
                                 + 0.81951 w_k1_MSFT_s117
                                 + 0.966001 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0481711522848807 w_k1_ADM_s000
                                 + 0.0371704920784288 w_k1_ADM_s001
                                 + 0.0317014839301388 w_k1_ADM_s002
                                 + 0.0210863623502379 w_k1_ADM_s003
                                 + 0.0506097594079079 w_k1_ADM_s004
                                 + 0.0157961614278372 w_k1_ADM_s005
                                 + 0.0225333150445884 w_k1_ADM_s006
                                 + 0.0465247353318657 w_k1_ADM_s007
                                 + 0.0222497719004289 w_k1_ADM_s010
                                 + 0.0379106103838258 w_k1_ADM_s011
                                 + 0.021666466205516 w_k1_ADM_s012
                                 + 0.0390047027150562 w_k1_ADM_s013
                                 + 0.0299407850550467 w_k1_ADM_s014
                                 + 0.0139480428873577 w_k1_ADM_s015
                                 + 0.0271482771506027 w_k1_ADM_s016
                                 + 0.0311261550048243 w_k1_ADM_s017
                                 + 0.0197367937901167 w_k1_ADM_s100
                                 + 0.0319046191741371 w_k1_ADM_s101
                                 + 0.0619067627155845 w_k1_ADM_s102
                                 + 0.0977311992227982 w_k1_ADM_s103
                                 + 0.0862702218219693 w_k1_ADM_s104
                                 + 0.0233655488259845 w_k1_ADM_s105
                                 + 0.0316872877493154 w_k1_ADM_s106
                                 + 0.0484253407591141 w_k1_ADM_s107
                                 + 0.014127245940752 w_k1_ADM_s110
                                 + 0.0116739759105494 w_k1_ADM_s111
                                 + 0.0254952676630887 w_k1_ADM_s112
                                 + 0.00569241403807124 w_k1_ADM_s113
                                 + 0.0272132086066384 w_k1_ADM_s114
                                 + 0.0140877275293174 w_k1_ADM_s115
                                 + 0.0101436665504033 w_k1_ADM_s116
                                 + 0.0087706602674706 w_k1_ADM_s117
                                 + 0.0530963123130993 w_k1_AXP_s000
                                 + 0.0332071257071425 w_k1_AXP_s001
                                 + 0.0300333680237356 w_k1_AXP_s002
                                 + 0.0219463626802192 w_k1_AXP_s003
                                 + 0.0455760182259653 w_k1_AXP_s004
                                 + 0.015057013012435 w_k1_AXP_s005
                                 + 0.0209856651172964 w_k1_AXP_s006
                                 + 0.0415932912965418 w_k1_AXP_s007
                                 + 0.0258583761248796 w_k1_AXP_s010
                                 + 0.0447265632849493 w_k1_AXP_s011
                                 + 0.0215335782974706 w_k1_AXP_s012
                                 + 0.0377884775989736 w_k1_AXP_s013
                                 + 0.0301558796065768 w_k1_AXP_s014
                                 + 0.0128544696614063 w_k1_AXP_s015
                                 + 0.0302542164539525 w_k1_AXP_s016
                                 + 0.0297605605530582 w_k1_AXP_s017
                                 + 0.0217256287770161 w_k1_AXP_s100
                                 + 0.0351283800771877 w_k1_AXP_s101
                                 + 0.0552054703580041 w_k1_AXP_s102
                                 + 0.0889062986425355 w_k1_AXP_s103
                                 + 0.0957876407714363 w_k1_AXP_s104
                                 + 0.022163282675136 w_k1_AXP_s105
                                 + 0.0362971168225618 w_k1_AXP_s106
                                 + 0.0452214139286197 w_k1_AXP_s107
                                 + 0.0120694050060559 w_k1_AXP_s110
                                 + 0.0143868475570921 w_k1_AXP_s111
                                 + 0.022005059506235 w_k1_AXP_s112
                                 + 0.00695512716170016 w_k1_AXP_s113
                                 + 0.0246835270239351 w_k1_AXP_s114
                                 + 0.0153535837788769 w_k1_AXP_s115
                                 + 0.0104326014952406 w_k1_AXP_s116
                                 + 0.0109108156262069 w_k1_AXP_s117
                                 + 0.0546561619887921 w_k1_DIS_s000
                                 + 0.0345339000595 w_k1_DIS_s001
                                 + 0.0345983655569063 w_k1_DIS_s002
                                 + 0.0225903712035358 w_k1_DIS_s003
                                 + 0.0470940526772474 w_k1_DIS_s004
                                 + 0.0153615644027112 w_k1_DIS_s005
                                 + 0.0202903496233404 w_k1_DIS_s006
                                 + 0.0423089451908502 w_k1_DIS_s007
                                 + 0.0241700225005631 w_k1_DIS_s010
                                 + 0.0429342181826989 w_k1_DIS_s011
                                 + 0.0220880725690675 w_k1_DIS_s012
                                 + 0.0393654839670057 w_k1_DIS_s013
                                 + 0.031027713781216 w_k1_DIS_s014
                                 + 0.0133804240685066 w_k1_DIS_s015
                                 + 0.0307545660619172 w_k1_DIS_s016
                                 + 0.0310454538320469 w_k1_DIS_s017
                                 + 0.0202617608706224 w_k1_DIS_s100
                                 + 0.0358429307595439 w_k1_DIS_s101
                                 + 0.0613198887155445 w_k1_DIS_s102
                                 + 0.096474730655811 w_k1_DIS_s103
                                 + 0.097547731278018 w_k1_DIS_s104
                                 + 0.0246170149783318 w_k1_DIS_s105
                                 + 0.0383086216004325 w_k1_DIS_s106
                                 + 0.0426539032513165 w_k1_DIS_s107
                                 + 0.0133176095395666 w_k1_DIS_s110
                                 + 0.0135591460971099 w_k1_DIS_s111
                                 + 0.0235545667218404 w_k1_DIS_s112
                                 + 0.0065333337771333 w_k1_DIS_s113
                                 + 0.0246411192228223 w_k1_DIS_s114
                                 + 0.0142993000264719 w_k1_DIS_s115
                                 + 0.010370898478244 w_k1_DIS_s116
                                 + 0.011761409472274 w_k1_DIS_s117
                                 + 0.0502358215434357 w_k1_FDX_s000
                                 + 0.0344190336021556 w_k1_FDX_s001
                                 + 0.0315994878863213 w_k1_FDX_s002
                                 + 0.0210756482561051 w_k1_FDX_s003
                                 + 0.0475770564037823 w_k1_FDX_s004
                                 + 0.0154131749529853 w_k1_FDX_s005
                                 + 0.0198614759724288 w_k1_FDX_s006
                                 + 0.0398852186247049 w_k1_FDX_s007
                                 + 0.0243213099961597 w_k1_FDX_s010
                                 + 0.0416171400790875 w_k1_FDX_s011
                                 + 0.0218862377449549 w_k1_FDX_s012
                                 + 0.0387735400999541 w_k1_FDX_s013
                                 + 0.0287835098212895 w_k1_FDX_s014
                                 + 0.0134503294617706 w_k1_FDX_s015
                                 + 0.0293006818136578 w_k1_FDX_s016
                                 + 0.0292973347810132 w_k1_FDX_s017
                                 + 0.0205798887481798 w_k1_FDX_s100
                                 + 0.0364437235109419 w_k1_FDX_s101
                                 + 0.0614440226263908 w_k1_FDX_s102
                                 + 0.0957203428928191 w_k1_FDX_s103
                                 + 0.0958658880506968 w_k1_FDX_s104
                                 + 0.0232762530706308 w_k1_FDX_s105
                                 + 0.0347000544775501 w_k1_FDX_s106
                                 + 0.0440525692852717 w_k1_FDX_s107
                                 + 0.0126502891797208 w_k1_FDX_s110
                                 + 0.0137413875113827 w_k1_FDX_s111
                                 + 0.0245839189856083 w_k1_FDX_s112
                                 + 0.00652605623538853 w_k1_FDX_s113
                                 + 0.0253904934853309 w_k1_FDX_s114
                                 + 0.0145556726561837 w_k1_FDX_s115
                                 + 0.0106560684502202 w_k1_FDX_s116
                                 + 0.0106587293028455 w_k1_FDX_s117
                                 + 0.0490383159470903 w_k1_IBM_s000
                                 + 0.0347979811162976 w_k1_IBM_s001
                                 + 0.0324616270569738 w_k1_IBM_s002
                                 + 0.0213610423012048 w_k1_IBM_s003
                                 + 0.0480449341287308 w_k1_IBM_s004
                                 + 0.0151140307338254 w_k1_IBM_s005
                                 + 0.0192647289210961 w_k1_IBM_s006
                                 + 0.0400021723892756 w_k1_IBM_s007
                                 + 0.0247490889657484 w_k1_IBM_s010
                                 + 0.0433186923413025 w_k1_IBM_s011
                                 + 0.0213082027074395 w_k1_IBM_s012
                                 + 0.038428639231802 w_k1_IBM_s013
                                 + 0.0286739985951335 w_k1_IBM_s014
                                 + 0.014452185530457 w_k1_IBM_s015
                                 + 0.0300431516754674 w_k1_IBM_s016
                                 + 0.0302795897575902 w_k1_IBM_s017
                                 + 0.0214150284715828 w_k1_IBM_s100
                                 + 0.0349210387394193 w_k1_IBM_s101
                                 + 0.0613926061642619 w_k1_IBM_s102
                                 + 0.0971872076532847 w_k1_IBM_s103
                                 + 0.09774892689547 w_k1_IBM_s104
                                 + 0.0238517493545641 w_k1_IBM_s105
                                 + 0.0360304891878471 w_k1_IBM_s106
                                 + 0.0437593490259754 w_k1_IBM_s107
                                 + 0.0131534631501898 w_k1_IBM_s110
                                 + 0.013629939945247 w_k1_IBM_s111
                                 + 0.023963275500639 w_k1_IBM_s112
                                 + 0.00651342997209742 w_k1_IBM_s113
                                 + 0.0253391827548838 w_k1_IBM_s114
                                 + 0.0144951812968156 w_k1_IBM_s115
                                 + 0.0111210414383916 w_k1_IBM_s116
                                 + 0.0111272603591363 w_k1_IBM_s117
                                 + 0.0509546643896152 w_k1_INTC_s000
                                 + 0.0306658859743526 w_k1_INTC_s001
                                 + 0.0323257394025078 w_k1_INTC_s002
                                 + 0.0215684322185922 w_k1_INTC_s003
                                 + 0.0474744978293768 w_k1_INTC_s004
                                 + 0.0153606232406503 w_k1_INTC_s005
                                 + 0.0201228170499301 w_k1_INTC_s006
                                 + 0.0331986649037646 w_k1_INTC_s007
                                 + 0.0212518556513085 w_k1_INTC_s010
                                 + 0.0408920613146029 w_k1_INTC_s011
                                 + 0.0206227960557056 w_k1_INTC_s012
                                 + 0.0424960492905828 w_k1_INTC_s013
                                 + 0.0269202334845121 w_k1_INTC_s014
                                 + 0.012905076741281 w_k1_INTC_s015
                                 + 0.0260541871455862 w_k1_INTC_s016
                                 + 0.0277174690917244 w_k1_INTC_s017
                                 + 0.0189367728125989 w_k1_INTC_s100
                                 + 0.0337609658084472 w_k1_INTC_s101
                                 + 0.0601718868066846 w_k1_INTC_s102
                                 + 0.104177722450704 w_k1_INTC_s103
                                 + 0.0961946535980559 w_k1_INTC_s104
                                 + 0.0215691399847126 w_k1_INTC_s105
                                 + 0.0319505629461622 w_k1_INTC_s106
                                 + 0.0391941355955368 w_k1_INTC_s107
                                 + 0.0119084264765745 w_k1_INTC_s110
                                 + 0.0144813839915601 w_k1_INTC_s111
                                 + 0.0192906853778078 w_k1_INTC_s112
                                 + 0.00571407048870527 w_k1_INTC_s113
                                 + 0.0255381114014895 w_k1_INTC_s114
                                 + 0.0134908111519851 w_k1_INTC_s115
                                 + 0.0109439718258133 w_k1_INTC_s116
                                 + 0.00952913969743398 w_k1_INTC_s117
                                 + 0.0519295593687832 w_k1_KO_s000
                                 + 0.0314471074931949 w_k1_KO_s001
                                 + 0.033630726949401 w_k1_KO_s002
                                 + 0.0225789303992502 w_k1_KO_s003
                                 + 0.0480776153583034 w_k1_KO_s004
                                 + 0.0159814648730808 w_k1_KO_s005
                                 + 0.0186412800231196 w_k1_KO_s006
                                 + 0.0429180645253184 w_k1_KO_s007
                                 + 0.0235947586234309 w_k1_KO_s010
                                 + 0.0369156258863146 w_k1_KO_s011
                                 + 0.0207896045631638 w_k1_KO_s012
                                 + 0.039706434590166 w_k1_KO_s013
                                 + 0.0275686366725919 w_k1_KO_s014
                                 + 0.0140568827883194 w_k1_KO_s015
                                 + 0.027720083801005 w_k1_KO_s016
                                 + 0.0279951724635564 w_k1_KO_s017
                                 + 0.0195482446507397 w_k1_KO_s100
                                 + 0.0334525388189604 w_k1_KO_s101
                                 + 0.0626651406295797 w_k1_KO_s102
                                 + 0.0936511162690415 w_k1_KO_s103
                                 + 0.0997870872263384 w_k1_KO_s104
                                 + 0.0243622191791701 w_k1_KO_s105
                                 + 0.0328940187900497 w_k1_KO_s106
                                 + 0.0445706407089352 w_k1_KO_s107
                                 + 0.0140169019045396 w_k1_KO_s110
                                 + 0.0135806595038117 w_k1_KO_s111
                                 + 0.025493349751753 w_k1_KO_s112
                                 + 0.00676917988193461 w_k1_KO_s113
                                 + 0.0264607549825073 w_k1_KO_s114
                                 + 0.0130202410289038 w_k1_KO_s115
                                 + 0.0100116993399005 w_k1_KO_s116
                                 + 0.0114384183591541 w_k1_KO_s117
                                 + 0.057118711468464 w_k1_MAT_s000
                                 + 0.0320199339912387 w_k1_MAT_s001
                                 + 0.0318476952254054 w_k1_MAT_s002
                                 + 0.021972603981899 w_k1_MAT_s003
                                 + 0.0418637278640474 w_k1_MAT_s004
                                 + 0.0153295680198299 w_k1_MAT_s005
                                 + 0.019772313908394 w_k1_MAT_s006
                                 + 0.0366337761763311 w_k1_MAT_s007
                                 + 0.0287067580645326 w_k1_MAT_s010
                                 + 0.0395208521586176 w_k1_MAT_s011
                                 + 0.0180458550505585 w_k1_MAT_s012
                                 + 0.0388706729434346 w_k1_MAT_s013
                                 + 0.0278694940892525 w_k1_MAT_s014
                                 + 0.014124827279155 w_k1_MAT_s015
                                 + 0.0297535496037524 w_k1_MAT_s016
                                 + 0.034135885290052 w_k1_MAT_s017
                                 + 0.0235114518929422 w_k1_MAT_s100
                                 + 0.0389503849642284 w_k1_MAT_s101
                                 + 0.0664382438238821 w_k1_MAT_s102
                                 + 0.0917826484095868 w_k1_MAT_s103
                                 + 0.121590580942685 w_k1_MAT_s104
                                 + 0.0277350723776237 w_k1_MAT_s105
                                 + 0.0329421306960441 w_k1_MAT_s106
                                 + 0.0537425256249126 w_k1_MAT_s107
                                 + 0.0138598352600985 w_k1_MAT_s110
                                 + 0.0150628344025522 w_k1_MAT_s111
                                 + 0.0216157948235989 w_k1_MAT_s112
                                 + 0.00786124685170252 w_k1_MAT_s113
                                 + 0.0253014025106553 w_k1_MAT_s114
                                 + 0.0132668070308714 w_k1_MAT_s115
                                 + 0.00951510569124486 w_k1_MAT_s116
                                 + 0.0107839066635884 w_k1_MAT_s117
                                 + 0.0483945700237585 w_k1_MSFT_s000
                                 + 0.0319481074977474 w_k1_MSFT_s001
                                 + 0.0331043022357599 w_k1_MSFT_s002
                                 + 0.0242436782267928 w_k1_MSFT_s003
                                 + 0.0481738730582384 w_k1_MSFT_s004
                                 + 0.0162880209552951 w_k1_MSFT_s005
                                 + 0.0199237776829997 w_k1_MSFT_s006
                                 + 0.038106374947558 w_k1_MSFT_s007
                                 + 0.0249610587479352 w_k1_MSFT_s010
                                 + 0.0418889790395815 w_k1_MSFT_s011
                                 + 0.0212121222657265 w_k1_MSFT_s012
                                 + 0.0399054106425837 w_k1_MSFT_s013
                                 + 0.0284447251156732 w_k1_MSFT_s014
                                 + 0.0131269736840704 w_k1_MSFT_s015
                                 + 0.0303936494294092 w_k1_MSFT_s016
                                 + 0.0287707219281052 w_k1_MSFT_s017
                                 + 0.0193224491857244 w_k1_MSFT_s100
                                 + 0.0360336831604575 w_k1_MSFT_s101
                                 + 0.0656977104237462 w_k1_MSFT_s102
                                 + 0.0995415275667689 w_k1_MSFT_s103
                                 + 0.0939228416869058 w_k1_MSFT_s104
                                 + 0.0232849769735664 w_k1_MSFT_s105
                                 + 0.0328178253802935 w_k1_MSFT_s106
                                 + 0.0402011378926481 w_k1_MSFT_s107
                                 + 0.0137290961717753 w_k1_MSFT_s110
                                 + 0.01472420936969 w_k1_MSFT_s111
                                 + 0.0222395208925873 w_k1_MSFT_s112
                                 + 0.00617634554675417 w_k1_MSFT_s113
                                 + 0.0247131935565496 w_k1_MSFT_s114
                                 + 0.0138042017658567 w_k1_MSFT_s115
                                 + 0.0100329933821242 w_k1_MSFT_s116
                                 + 0.0101347412375063 w_k1_MSFT_s117
                                 + 0.0485027902449517 w_k1_PG_s000
                                 + 0.0342826788755656 w_k1_PG_s001
                                 + 0.0342488043178898 w_k1_PG_s002
                                 + 0.0218997292081709 w_k1_PG_s003
                                 + 0.0471733439640159 w_k1_PG_s004
                                 + 0.0153738602561725 w_k1_PG_s005
                                 + 0.018566643001961 w_k1_PG_s006
                                 + 0.0393547631863056 w_k1_PG_s007
                                 + 0.0245875838883296 w_k1_PG_s010
                                 + 0.041652809653529 w_k1_PG_s011
                                 + 0.0201995541986888 w_k1_PG_s012
                                 + 0.0391731109265377 w_k1_PG_s013
                                 + 0.0290187254134927 w_k1_PG_s014
                                 + 0.0138587267582141 w_k1_PG_s015
                                 + 0.0292745548624511 w_k1_PG_s016
                                 + 0.0280157203928314 w_k1_PG_s017
                                 + 0.0203265427316602 w_k1_PG_s100
                                 + 0.0357708290464176 w_k1_PG_s101
                                 + 0.0652694828758588 w_k1_PG_s102
                                 + 0.0969114726238522 w_k1_PG_s103
                                 + 0.0933217727349972 w_k1_PG_s104
                                 + 0.0233067873838574 w_k1_PG_s105
                                 + 0.0332645004369886 w_k1_PG_s106
                                 + 0.0437740065353511 w_k1_PG_s107
                                 + 0.013117326883694 w_k1_PG_s110
                                 + 0.0142382425865942 w_k1_PG_s111
                                 + 0.0234197358777678 w_k1_PG_s112
                                 + 0.00594391648843864 w_k1_PG_s113
                                 + 0.0274447298259815 w_k1_PG_s114
                                 + 0.0139508110554123 w_k1_PG_s115
                                 + 0.0101857790926313 w_k1_PG_s116
                                 + 0.01104708452158 w_k1_PG_s117 >= 1.16
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
