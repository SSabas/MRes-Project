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
 w_k1_ADM_s0_return_constraint:  1.14587 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.95092 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.04582 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.12496 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.06752 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.788698 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.846739 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.31986 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.773346 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.791839 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.81632 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.00785 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.2268 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.07956 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.736401 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.33562 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.04079 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.816879 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.720305 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.771057 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.902749 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.04871 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.1442 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.15907 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.669742 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.863436 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.32571 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.822551 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.748359 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.12589 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.904238 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.916126 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.899535 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.30289 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.45148 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.01011 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.15338 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.50169 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19172 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.826476 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.03744 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.30902 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.19439 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.85106 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.821872 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.94808 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.589048 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.814998 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.666429 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.10811 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.33733 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.071 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.636385 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.54432 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.49907 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.894748 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.715855 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.89371 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.973232 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.04449 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.91768 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.26564 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.74762 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.846126 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.38909 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.749383 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.768758 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.23364 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.72079 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.855041 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.908479 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.3767 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.34763 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.841496 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.03667 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.1935 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29765 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.865493 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.94656 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.12275 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.15221 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.721868 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.956486 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 2.35391 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.530634 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.729836 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.757403 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.922358 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.3777 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.20639 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.647408 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 2.15005 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.40142 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.04884 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.712786 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.943597 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.08194 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.53206 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.16674 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.35826 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.573938 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.84299 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.45593 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.840762 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.636928 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.984713 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.726889 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.720805 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.856321 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.27575 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.37713 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.16883 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.06152 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.50937 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16765 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.900016 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.02623 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.05176 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.08983 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.832727 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.94163 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.7791 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.659504 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.750833 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.847961 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.955575 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.20313 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.23849 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.741389 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.61894 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.18031 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.983551 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.689977 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.885073 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.05184 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.18871 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.12043 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.11363 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.64132 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.966971 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.34996 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.761501 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.78623 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.08151 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.86257 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.820096 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.916282 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.12684 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.32741 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.00203 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.10847 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.46242 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15258 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.926271 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.881112 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.45065 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17706 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.743729 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.907741 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.32284 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.699578 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.32303 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.691281 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.0416 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.3363 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.743097 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.73322 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.22956 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.38698 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.850068 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.25931 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.19386 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.11754 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.18636 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.709693 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.62736 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.257 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.85469 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.026 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.15846 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.762984 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.921631 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.739191 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.44966 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.992853 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.52439 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.07097 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.26026 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.829209 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.887851 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10142 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.866763 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.988064 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06804 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15348 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.976045 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.10113 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.71683 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.780892 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.888195 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.783823 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.01564 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.19388 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.296 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.799718 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.48328 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.11308 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.01758 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.626933 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.943267 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.20392 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.22183 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.934059 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.11976 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.817458 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.986272 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.14648 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.779608 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.72514 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.954934 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.880296 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.816753 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.896816 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.14368 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.0572 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.06458 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.930237 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.13159 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20388 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.903157 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.884265 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.14231 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.11915 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.842148 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.02168 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.58861 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.647781 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.0506 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.715365 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.997459 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.43127 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.998653 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.751256 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.54429 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.32355 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.929485 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.907774 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.0828 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.05529 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.39095 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.01258 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.22805 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.0163 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.855546 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.11276 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00212 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.743123 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.90146 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.639631 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.936335 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.846975 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.27993 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.13286 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.13473 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.865169 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.04612 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.18699 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.947139 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.930166 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.47777 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.20183 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.660547 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.862863 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.43375 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.624284 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.26609 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.595043 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.05514 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.41943 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.620583 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.655428 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.46085 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.64674 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.802694 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.1439 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.3279 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.832489 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.13975 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.595858 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.81274 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.09093 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.748453 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.07215 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.956032 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.666419 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.08375 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.703261 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.46359 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.14462 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.81218 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.28297 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.26005 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.942285 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.807918 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18991 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.907295 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.933106 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.05131 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.12034 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.840587 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.06043 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.79435 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.650839 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.81381 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.859035 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.954525 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.265 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.31073 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.681559 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.55613 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.11846 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.01532 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.661434 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.858916 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.985091 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.14374 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.12791 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.03579 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.716159 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.922213 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.19191 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.846944 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.711028 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.02875 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.882356 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.761484 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.944794 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.18198 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17806 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.14255 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.15732 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.28411 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14484 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.92511 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.921147 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.0447 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.13119 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.893604 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.985145 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.47187 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.803637 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.977096 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.932344 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.959088 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.26557 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.07357 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.761591 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.21126 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.25252 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.990613 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.880765 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.03835 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.965309 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.18798 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.939199 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.07343 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.842832 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.880993 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.0884 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.921963 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.778689 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.961901 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.794703 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.886298 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.950464 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.16318 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.13561 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.0163 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.978569 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.08217 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.583999554543665 z_k1_s0000
                                 - 0.371000461634522 z_k1_s0001
                                 - 0.884001075476289 z_k1_s0002
                                 - 1.46000172486348 z_k1_s0003
                                 - 0.647002996294261 z_k1_s0004
                                 - 0.511000603702217 z_k1_s0005
                                 - 0.740003751633806 z_k1_s0006
                                 - 0.48000073038961 z_k1_s0007
                                 - 0.0790000734155323 z_k1_s0010
                                 - 0.110000017125383 z_k1_s0011
                                 - 0.147000268598587 z_k1_s0012
                                 - 0.229999449589147 z_k1_s0013
                                 - 0.368999582236731 z_k1_s0014
                                 - 0.0890000375541849 z_k1_s0015
                                 - 0.109000103031558 z_k1_s0016
                                 - 0.23899977403411 z_k1_s0017
                                 - 0.239000478411023 z_k1_s0020
                                 - 0.766001398723038 z_k1_s0021
                                 - 0.71899987457777 z_k1_s0022
                                 - 0.777002867630682 z_k1_s0023
                                 - 0.543000498083768 z_k1_s0024
                                 - 0.303000397899923 z_k1_s0025
                                 - 0.481000626341455 z_k1_s0026
                                 - 0.193000184440464 z_k1_s0027
                                 - 0.342001141102956 z_k1_s0030
                                 - 0.353999594826696 z_k1_s0031
                                 - 0.223999814828155 z_k1_s0032
                                 - 0.211999991105131 z_k1_s0033
                                 - 0.659999894761297 z_k1_s0034
                                 - 0.548999812824222 z_k1_s0035
                                 - 0.135000003382979 z_k1_s0036
                                 - 0.263999957904519 z_k1_s0037
                                 - 0.175999989447347 z_k1_s0040
                                 - 0.29199977549215 z_k1_s0041
                                 - 0.216000125049043 z_k1_s0042
                                 - 0.517000538251689 z_k1_s0043
                                 - 0.554001105283342 z_k1_s0044
                                 - 0.339998944613996 z_k1_s0045
                                 - 0.391000994306515 z_k1_s0046
                                 - 0.27400015607147 z_k1_s0047
                                 - 0.549999250807335 z_k1_s0050
                                 - 0.224999786093842 z_k1_s0051
                                 - 0.131000006719431 z_k1_s0052
                                 - 0.481000154887281 z_k1_s0053
                                 - 0.372998913275212 z_k1_s0054
                                 - 0.247999824200281 z_k1_s0055
                                 - 0.407998749872127 z_k1_s0056
                                 - 0.129999938645286 z_k1_s0057
                                 - 0.508998705717609 z_k1_s0060
                                 - 1.61899739067034 z_k1_s0061
                                 - 0.336999057496896 z_k1_s0062
                                 - 0.432999222956583 z_k1_s0063
                                 - 0.668000399379302 z_k1_s0064
                                 - 0.559999202001641 z_k1_s0065
                                 - 0.748998808217439 z_k1_s0066
                                 - 1.34799871918269 z_k1_s0067
                                 - 0.599000950551728 z_k1_s0070
                                 - 0.571000921550953 z_k1_s0071
                                 - 0.911997643173624 z_k1_s0072
                                 - 0.577000267480799 z_k1_s0073
                                 - 0.437999397199602 z_k1_s0074
                                 - 0.666999700412968 z_k1_s0075
                                 - 0.617997338665492 z_k1_s0076
                                 - 0.239999720350382 z_k1_s0077
                                 - 0.472001353220537 z_k1_s0100
                                 - 0.162000486664732 z_k1_s0101
                                 - 0.373000691785617 z_k1_s0102
                                 - 0.426000066319016 z_k1_s0103
                                 - 0.427001290841629 z_k1_s0104
                                 - 0.125000260641633 z_k1_s0105
                                 - 0.150000470030356 z_k1_s0106
                                 - 0.486000149490894 z_k1_s0107
                                 - 0.163999773559986 z_k1_s0110
                                 - 0.0890000706150869 z_k1_s0111
                                 - 0.0850001061380476 z_k1_s0112
                                 - 0.0950001403307834 z_k1_s0113
                                 - 0.410000048900653 z_k1_s0114
                                 - 0.107000156762039 z_k1_s0115
                                 - 0.168999729156285 z_k1_s0116
                                 - 0.111000121239078 z_k1_s0117
                                 - 0.265001552214961 z_k1_s0120
                                 - 0.140000456921812 z_k1_s0121
                                 - 0.208001141896776 z_k1_s0122
                                 - 0.149000971041941 z_k1_s0123
                                 - 0.235000803151016 z_k1_s0124
                                 - 0.22600043373136 z_k1_s0125
                                 - 0.0660002009359496 z_k1_s0126
                                 - 0.157999893456868 z_k1_s0127
                                 - 0.447001377679943 z_k1_s0130
                                 - 0.3629992340376 z_k1_s0131
                                 - 0.424999743128195 z_k1_s0132
                                 - 0.413000427467369 z_k1_s0133
                                 - 0.514001312711859 z_k1_s0134
                                 - 0.366001663209017 z_k1_s0135
                                 - 0.61100127039998 z_k1_s0136
                                 - 0.328000825542558 z_k1_s0137
                                 - 0.0760000384574535 z_k1_s0140
                                 - 0.147000229934836 z_k1_s0141
                                 - 0.163000360431161 z_k1_s0142
                                 - 0.197999676891721 z_k1_s0143
                                 - 0.155000295182998 z_k1_s0144
                                 - 0.116999985254226 z_k1_s0145
                                 - 0.0469999957328945 z_k1_s0146
                                 - 0.0660000537972654 z_k1_s0147
                                 - 0.189000327431468 z_k1_s0150
                                 - 0.127000028353408 z_k1_s0151
                                 - 0.0699999871042388 z_k1_s0152
                                 - 0.136999739011138 z_k1_s0153
                                 - 0.094999666248589 z_k1_s0154
                                 - 0.0520000249202939 z_k1_s0155
                                 - 0.0550000186176181 z_k1_s0156
                                 - 0.0800000197619894 z_k1_s0157
                                 - 0.180000516810133 z_k1_s0160
                                 - 0.128000478809892 z_k1_s0161
                                 - 0.305001750209707 z_k1_s0162
                                 - 0.206000464259955 z_k1_s0163
                                 - 0.0630003239841435 z_k1_s0164
                                 - 0.20500019108766 z_k1_s0165
                                 - 0.148000361332511 z_k1_s0166
                                 - 0.196000594548944 z_k1_s0167
                                 - 0.0929997915972568 z_k1_s0170
                                 - 0.134000510893559 z_k1_s0171
                                 - 0.169999868640374 z_k1_s0172
                                 - 0.0360000209474176 z_k1_s0173
                                 - 0.0610000308886757 z_k1_s0174
                                 - 0.110999884971041 z_k1_s0175
                                 - 0.13299990698336 z_k1_s0176
                                 - 0.0910002417783663 z_k1_s0177
                                 - 0.62899562553921 z_k1_s1000
                                 - 0.61199574376788 z_k1_s1001
                                 - 0.517998434896859 z_k1_s1002
                                 - 0.372998641720236 z_k1_s1003
                                 - 1.15699696348101 z_k1_s1004
                                 - 0.443998415140884 z_k1_s1005
                                 - 1.45299590690827 z_k1_s1006
                                 - 0.491998381917869 z_k1_s1007
                                 - 0.35299932841131 z_k1_s1010
                                 - 1.18599690189406 z_k1_s1011
                                 - 1.19399924105056 z_k1_s1012
                                 - 1.76299785151387 z_k1_s1013
                                 - 0.626998703431343 z_k1_s1014
                                 - 0.916996297477167 z_k1_s1015
                                 - 0.39999925525811 z_k1_s1016
                                 - 0.736997493849313 z_k1_s1017
                                 - 0.123000072902911 z_k1_s1020
                                 - 0.310000809714893 z_k1_s1021
                                 - 0.330000696373855 z_k1_s1022
                                 - 0.114000038356345 z_k1_s1023
                                 - 0.222000624015198 z_k1_s1024
                                 - 0.280000124226123 z_k1_s1025
                                 - 0.14600002811075 z_k1_s1026
                                 - 0.185999459228543 z_k1_s1027
                                 - 0.537000225483188 z_k1_s1030
                                 - 1.07799884021692 z_k1_s1031
                                 - 0.876003347794291 z_k1_s1032
                                 - 1.72800306167317 z_k1_s1033
                                 - 0.484000142693917 z_k1_s1034
                                 - 0.474000475564184 z_k1_s1035
                                 - 0.938003500400659 z_k1_s1036
                                 - 1.27299789025177 z_k1_s1037
                                 - 0.625001721028356 z_k1_s1040
                                 - 1.21100406057663 z_k1_s1041
                                 - 1.42700582270914 z_k1_s1042
                                 - 0.673001431389555 z_k1_s1043
                                 - 0.531001010943045 z_k1_s1044
                                 - 0.540001339824158 z_k1_s1045
                                 - 0.759001167912297 z_k1_s1046
                                 - 0.365000735315934 z_k1_s1047
                                 - 0.136999861819068 z_k1_s1050
                                 - 0.573999875568654 z_k1_s1051
                                 - 0.432000261723918 z_k1_s1052
                                 - 0.376999768643618 z_k1_s1053
                                 - 0.392000635119599 z_k1_s1054
                                 - 0.461999310677583 z_k1_s1055
                                 - 0.11899989564721 z_k1_s1056
                                 - 0.190999760334643 z_k1_s1057
                                 - 0.977994893701971 z_k1_s1060
                                 - 1.29699520757736 z_k1_s1061
                                 - 0.65499752180678 z_k1_s1062
                                 - 0.516998153925785 z_k1_s1063
                                 - 0.928996475562621 z_k1_s1064
                                 - 0.503998317889581 z_k1_s1065
                                 - 0.970996909676698 z_k1_s1066
                                 - 0.413998730141106 z_k1_s1067
                                 - 0.532999140831799 z_k1_s1070
                                 - 0.183999988301422 z_k1_s1071
                                 - 0.214999938331039 z_k1_s1072
                                 - 0.440999289131 z_k1_s1073
                                 - 0.311999497072272 z_k1_s1074
                                 - 0.163000022152327 z_k1_s1075
                                 - 0.325999474505002 z_k1_s1076
                                 - 0.674998911935205 z_k1_s1077
                                 - 0.441998939209785 z_k1_s1100
                                 - 0.590998388264747 z_k1_s1101
                                 - 0.354999840194906 z_k1_s1102
                                 - 1.08299995302203 z_k1_s1103
                                 - 0.593998033037806 z_k1_s1104
                                 - 0.433999886481627 z_k1_s1105
                                 - 0.478998831075771 z_k1_s1106
                                 - 0.294999680636689 z_k1_s1107
                                 - 0.512000578160137 z_k1_s1110
                                 - 0.313001258252386 z_k1_s1111
                                 - 0.209000465224506 z_k1_s1112
                                 - 0.493000255102698 z_k1_s1113
                                 - 0.137000425283335 z_k1_s1114
                                 - 0.210000447395867 z_k1_s1115
                                 - 0.117000340655057 z_k1_s1116
                                 - 0.215000578853198 z_k1_s1117
                                 - 0.308999507554084 z_k1_s1120
                                 - 0.259000700442376 z_k1_s1121
                                 - 0.168000157097596 z_k1_s1122
                                 - 0.274000108575762 z_k1_s1123
                                 - 0.585999497756522 z_k1_s1124
                                 - 0.291999398335824 z_k1_s1125
                                 - 0.213000019498641 z_k1_s1126
                                 - 0.239000709597439 z_k1_s1127
                                 - 0.124999838539922 z_k1_s1130
                                 - 0.151999832574906 z_k1_s1131
                                 - 0.268999489726937 z_k1_s1132
                                 - 0.31899919690322 z_k1_s1133
                                 - 0.560999833793602 z_k1_s1134
                                 - 0.145999791633856 z_k1_s1135
                                 - 0.214999882056446 z_k1_s1136
                                 - 0.114999897104666 z_k1_s1137
                                 - 0.247000503226486 z_k1_s1140
                                 - 0.149999324258605 z_k1_s1141
                                 - 0.279999822483166 z_k1_s1142
                                 - 0.242000427217826 z_k1_s1143
                                 - 0.072000059924285 z_k1_s1144
                                 - 0.223000729585157 z_k1_s1145
                                 - 0.0659999687138933 z_k1_s1146
                                 - 0.198000349541858 z_k1_s1147
                                 - 0.291000327538859 z_k1_s1150
                                 - 0.144999965783973 z_k1_s1151
                                 - 0.137999951587506 z_k1_s1152
                                 - 0.125000012765493 z_k1_s1153
                                 - 0.241999308963615 z_k1_s1154
                                 - 0.20300071809754 z_k1_s1155
                                 - 0.149000017667667 z_k1_s1156
                                 - 0.238999653050833 z_k1_s1157
                                 - 0.452001652686333 z_k1_s1160
                                 - 0.227999876386839 z_k1_s1161
                                 - 0.204000670822752 z_k1_s1162
                                 - 0.193000699112945 z_k1_s1163
                                 - 0.17100054359253 z_k1_s1164
                                 - 0.329002277532381 z_k1_s1165
                                 - 0.339000709262749 z_k1_s1166
                                 - 0.205000726096589 z_k1_s1167
                                 - 0.393999814691568 z_k1_s1170
                                 - 0.113999942563044 z_k1_s1171
                                 - 0.0929999333409121 z_k1_s1172
                                 - 0.199999371163443 z_k1_s1173
                                 - 0.241000070001623 z_k1_s1174
                                 - 0.176000650623355 z_k1_s1175
                                 - 0.190999474711059 z_k1_s1176
                                 - 0.0959999991583341 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.26444 w_k1_ADM_s000
                                 + 1.21706 w_k1_AXP_s000
                                 + 1.18806 w_k1_DIS_s000
                                 + 1.21638 w_k1_FDX_s000
                                 + 0.884228 w_k1_IBM_s000
                                 + 1.1021 w_k1_INTC_s000
                                 + 0.965323 w_k1_KO_s000
                                 + 1.14039 w_k1_MAT_s000
                                 + 1.14881 w_k1_MSFT_s000
                                 + 0.91494 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.719191 w_k1_ADM_s000
                                 + 1.07398 w_k1_AXP_s000
                                 + 0.740062 w_k1_DIS_s000
                                 + 0.839066 w_k1_FDX_s000
                                 + 0.909921 w_k1_IBM_s000
                                 + 0.897061 w_k1_INTC_s000
                                 + 0.80301 w_k1_KO_s000
                                 + 0.935732 w_k1_MAT_s000
                                 + 0.865107 w_k1_MSFT_s000
                                 + 0.940637 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.00403 w_k1_ADM_s000
                                 + 1.22319 w_k1_AXP_s000
                                 + 1.55953 w_k1_DIS_s000
                                 + 1.26688 w_k1_FDX_s000
                                 + 1.11592 w_k1_IBM_s000
                                 + 1.01492 w_k1_INTC_s000
                                 + 1.13613 w_k1_KO_s000 + 1.26242 w_k1_MAT_s000
                                 + 1.22303 w_k1_MSFT_s000
                                 + 1.21473 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.924703 w_k1_ADM_s000
                                 + 0.990593 w_k1_AXP_s000
                                 + 1.04416 w_k1_DIS_s000
                                 + 0.965611 w_k1_FDX_s000
                                 + 1.17572 w_k1_IBM_s000
                                 + 0.771234 w_k1_INTC_s000
                                 + 0.9985 w_k1_KO_s000 + 0.968474 w_k1_MAT_s000
                                 + 0.864955 w_k1_MSFT_s000
                                 + 0.952315 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.857459 w_k1_ADM_s000
                                 + 0.714597 w_k1_AXP_s000
                                 + 0.698241 w_k1_DIS_s000
                                 + 0.752504 w_k1_FDX_s000
                                 + 0.965008 w_k1_IBM_s000
                                 + 0.819192 w_k1_INTC_s000
                                 + 0.79335 w_k1_KO_s000
                                 + 0.725117 w_k1_MAT_s000
                                 + 0.829758 w_k1_MSFT_s000
                                 + 0.888442 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.822311 w_k1_ADM_s000
                                 + 0.780725 w_k1_AXP_s000
                                 + 0.764698 w_k1_DIS_s000
                                 + 0.892152 w_k1_FDX_s000
                                 + 0.709003 w_k1_IBM_s000
                                 + 0.770229 w_k1_INTC_s000
                                 + 0.766822 w_k1_KO_s000
                                 + 0.751786 w_k1_MAT_s000
                                 + 0.842619 w_k1_MSFT_s000
                                 + 0.880519 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.18178 w_k1_ADM_s000
                                 + 0.904857 w_k1_AXP_s000
                                 + 0.941563 w_k1_DIS_s000
                                 + 1.01708 w_k1_FDX_s000
                                 + 0.719694 w_k1_IBM_s000
                                 + 0.91853 w_k1_INTC_s000
                                 + 0.818845 w_k1_KO_s000
                                 + 0.786212 w_k1_MAT_s000
                                 + 0.913712 w_k1_MSFT_s000
                                 + 1.04749 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.27395 w_k1_ADM_s000
                                 + 1.40457 w_k1_AXP_s000
                                 + 1.55777 w_k1_DIS_s000
                                 + 1.18209 w_k1_FDX_s000 + 1.6067 w_k1_IBM_s000
                                 + 1.0227 w_k1_INTC_s000 + 1.49203 w_k1_KO_s000
                                 + 1.71364 w_k1_MAT_s000
                                 + 1.18154 w_k1_MSFT_s000
                                 + 1.12777 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.16108 w_k1_ADM_s001
                                 + 1.45694 w_k1_AXP_s001
                                 + 1.31581 w_k1_DIS_s001 + 1.2436 w_k1_FDX_s001
                                 + 1.14303 w_k1_IBM_s001
                                 + 1.34912 w_k1_INTC_s001
                                 + 1.20448 w_k1_KO_s001 + 1.81637 w_k1_MAT_s001
                                 + 1.18671 w_k1_MSFT_s001 + 1.1431 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.10267 w_k1_ADM_s001
                                 + 0.950264 w_k1_AXP_s001
                                 + 0.956544 w_k1_DIS_s001
                                 + 1.18811 w_k1_FDX_s001
                                 + 0.665389 w_k1_IBM_s001
                                 + 1.0332 w_k1_INTC_s001
                                 + 0.909818 w_k1_KO_s001
                                 + 0.934191 w_k1_MAT_s001
                                 + 1.06478 w_k1_MSFT_s001
                                 + 0.968051 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.29071 w_k1_ADM_s001
                                 + 1.07847 w_k1_AXP_s001
                                 + 1.80956 w_k1_DIS_s001
                                 + 1.26115 w_k1_FDX_s001
                                 + 1.46579 w_k1_IBM_s001
                                 + 0.930303 w_k1_INTC_s001
                                 + 1.25159 w_k1_KO_s001 + 1.40292 w_k1_MAT_s001
                                 + 1.16992 w_k1_MSFT_s001
                                 + 1.07188 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.00508 w_k1_ADM_s001
                                 + 0.710994 w_k1_AXP_s001
                                 + 0.90675 w_k1_DIS_s001
                                 + 0.783704 w_k1_FDX_s001
                                 + 1.33273 w_k1_IBM_s001
                                 + 0.719642 w_k1_INTC_s001
                                 + 1.07812 w_k1_KO_s001 + 1.3855 w_k1_MAT_s001
                                 + 0.766385 w_k1_MSFT_s001
                                 + 0.896833 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.11072 w_k1_ADM_s001
                                 + 0.944507 w_k1_AXP_s001
                                 + 1.04357 w_k1_DIS_s001
                                 + 0.924378 w_k1_FDX_s001
                                 + 0.997658 w_k1_IBM_s001
                                 + 0.950524 w_k1_INTC_s001
                                 + 1.12996 w_k1_KO_s001
                                 + 0.956267 w_k1_MAT_s001
                                 + 1.02681 w_k1_MSFT_s001
                                 + 0.989646 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.27105 w_k1_ADM_s001
                                 + 0.815422 w_k1_AXP_s001
                                 + 1.35602 w_k1_DIS_s001 + 1.0359 w_k1_FDX_s001
                                 + 0.976284 w_k1_IBM_s001
                                 + 0.941212 w_k1_INTC_s001
                                 + 1.29947 w_k1_KO_s001 + 1.10701 w_k1_MAT_s001
                                 + 1.20553 w_k1_MSFT_s001
                                 + 1.11527 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.35933 w_k1_ADM_s001
                                 + 1.33318 w_k1_AXP_s001
                                 + 0.964858 w_k1_DIS_s001
                                 + 1.09804 w_k1_FDX_s001
                                 + 1.15747 w_k1_IBM_s001
                                 + 1.01725 w_k1_INTC_s001
                                 + 1.07388 w_k1_KO_s001 + 1.52048 w_k1_MAT_s001
                                 + 0.97249 w_k1_MSFT_s001
                                 + 1.09096 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.949804 w_k1_ADM_s001
                                 + 0.728444 w_k1_AXP_s001
                                 + 0.688508 w_k1_DIS_s001
                                 + 0.784543 w_k1_FDX_s001
                                 + 0.698954 w_k1_IBM_s001
                                 + 0.95675 w_k1_INTC_s001
                                 + 0.848044 w_k1_KO_s001
                                 + 0.743721 w_k1_MAT_s001
                                 + 0.754331 w_k1_MSFT_s001
                                 + 0.906294 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.885358 w_k1_ADM_s002
                                 + 0.876118 w_k1_AXP_s002
                                 + 0.793964 w_k1_DIS_s002
                                 + 0.880423 w_k1_FDX_s002
                                 + 0.719631 w_k1_IBM_s002
                                 + 0.899617 w_k1_INTC_s002
                                 + 0.861814 w_k1_KO_s002
                                 + 0.964239 w_k1_MAT_s002
                                 + 0.986864 w_k1_MSFT_s002
                                 + 0.939987 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.83272 w_k1_ADM_s002
                                 + 0.8751 w_k1_AXP_s002
                                 + 0.897705 w_k1_DIS_s002
                                 + 0.971359 w_k1_FDX_s002
                                 + 0.656607 w_k1_IBM_s002
                                 + 1.04543 w_k1_INTC_s002
                                 + 0.779702 w_k1_KO_s002
                                 + 0.630022 w_k1_MAT_s002
                                 + 0.936467 w_k1_MSFT_s002
                                 + 0.941324 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.755209 w_k1_ADM_s002
                                 + 0.822505 w_k1_AXP_s002
                                 + 0.997054 w_k1_DIS_s002
                                 + 0.930936 w_k1_FDX_s002
                                 + 1.26903 w_k1_IBM_s002
                                 + 0.958735 w_k1_INTC_s002 + 1.312 w_k1_KO_s002
                                 + 1.12241 w_k1_MAT_s002
                                 + 1.07555 w_k1_MSFT_s002
                                 + 0.990381 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.06066 w_k1_ADM_s002
                                 + 1.10656 w_k1_AXP_s002
                                 + 1.47282 w_k1_DIS_s002
                                 + 1.17714 w_k1_FDX_s002 + 1.3438 w_k1_IBM_s002
                                 + 1.08014 w_k1_INTC_s002
                                 + 1.41836 w_k1_KO_s002 + 1.0097 w_k1_MAT_s002
                                 + 1.48093 w_k1_MSFT_s002
                                 + 1.16412 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.09308 w_k1_ADM_s002
                                 + 1.31334 w_k1_AXP_s002
                                 + 0.888356 w_k1_DIS_s002
                                 + 1.06697 w_k1_FDX_s002 + 1.0408 w_k1_IBM_s002
                                 + 0.881986 w_k1_INTC_s002
                                 + 1.02205 w_k1_KO_s002 + 1.2977 w_k1_MAT_s002
                                 + 1.0535 w_k1_MSFT_s002 + 1.04723 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.32214 w_k1_ADM_s002
                                 + 1.48241 w_k1_AXP_s002
                                 + 1.47885 w_k1_DIS_s002
                                 + 1.46637 w_k1_FDX_s002
                                 + 0.862697 w_k1_IBM_s002
                                 + 1.01776 w_k1_INTC_s002
                                 + 1.17763 w_k1_KO_s002 + 1.00646 w_k1_MAT_s002
                                 + 1.45973 w_k1_MSFT_s002
                                 + 1.19567 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.788021 w_k1_ADM_s002
                                 + 0.756817 w_k1_AXP_s002
                                 + 0.737823 w_k1_DIS_s002
                                 + 0.764133 w_k1_FDX_s002
                                 + 0.958957 w_k1_IBM_s002
                                 + 0.946848 w_k1_INTC_s002
                                 + 0.941869 w_k1_KO_s002
                                 + 0.894967 w_k1_MAT_s002
                                 + 0.992574 w_k1_MSFT_s002
                                 + 0.816191 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.893293 w_k1_ADM_s002
                                 + 1.41258 w_k1_AXP_s002
                                 + 1.36909 w_k1_DIS_s002
                                 + 1.11753 w_k1_FDX_s002
                                 + 1.99276 w_k1_IBM_s002
                                 + 1.20284 w_k1_INTC_s002 + 1.3992 w_k1_KO_s002
                                 + 1.89528 w_k1_MAT_s002
                                 + 1.22687 w_k1_MSFT_s002
                                 + 1.15277 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.21564 w_k1_ADM_s003
                                 + 1.101 w_k1_AXP_s003 + 1.51528 w_k1_DIS_s003
                                 + 1.28502 w_k1_FDX_s003
                                 + 0.982546 w_k1_IBM_s003
                                 + 1.37433 w_k1_INTC_s003
                                 + 1.24195 w_k1_KO_s003
                                 + 0.909331 w_k1_MAT_s003
                                 + 1.24715 w_k1_MSFT_s003
                                 + 1.06607 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.660393 w_k1_ADM_s003
                                 + 0.657185 w_k1_AXP_s003
                                 + 0.67374 w_k1_DIS_s003
                                 + 0.741514 w_k1_FDX_s003
                                 + 0.835114 w_k1_IBM_s003
                                 + 0.878119 w_k1_INTC_s003
                                 + 0.636903 w_k1_KO_s003
                                 + 0.774277 w_k1_MAT_s003
                                 + 0.812201 w_k1_MSFT_s003
                                 + 0.80627 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.11367 w_k1_ADM_s003
                                 + 1.24319 w_k1_AXP_s003
                                 + 1.37085 w_k1_DIS_s003
                                 + 0.992242 w_k1_FDX_s003
                                 + 1.62293 w_k1_IBM_s003
                                 + 1.09563 w_k1_INTC_s003
                                 + 1.17621 w_k1_KO_s003 + 1.57042 w_k1_MAT_s003
                                 + 1.10465 w_k1_MSFT_s003
                                 + 1.05853 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.11399 w_k1_ADM_s003
                                 + 1.45426 w_k1_AXP_s003
                                 + 1.82184 w_k1_DIS_s003
                                 + 1.33418 w_k1_FDX_s003
                                 + 1.46824 w_k1_IBM_s003
                                 + 1.18201 w_k1_INTC_s003
                                 + 1.36522 w_k1_KO_s003 + 1.28865 w_k1_MAT_s003
                                 + 1.47095 w_k1_MSFT_s003
                                 + 1.11677 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01329 w_k1_ADM_s003
                                 + 0.939638 w_k1_AXP_s003
                                 + 0.906258 w_k1_DIS_s003
                                 + 0.971805 w_k1_FDX_s003
                                 + 0.79262 w_k1_IBM_s003
                                 + 1.02006 w_k1_INTC_s003
                                 + 0.908045 w_k1_KO_s003
                                 + 0.842072 w_k1_MAT_s003
                                 + 1.042 w_k1_MSFT_s003 + 0.995548 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.0697 w_k1_ADM_s003
                                 + 0.976473 w_k1_AXP_s003
                                 + 1.2985 w_k1_DIS_s003 + 1.00013 w_k1_FDX_s003
                                 + 1.15318 w_k1_IBM_s003
                                 + 0.987871 w_k1_INTC_s003
                                 + 1.13138 w_k1_KO_s003 + 1.16663 w_k1_MAT_s003
                                 + 1.27046 w_k1_MSFT_s003
                                 + 1.04171 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.743619 w_k1_ADM_s003
                                 + 0.645378 w_k1_AXP_s003
                                 + 0.629869 w_k1_DIS_s003
                                 + 0.768222 w_k1_FDX_s003
                                 + 0.576312 w_k1_IBM_s003
                                 + 1.01237 w_k1_INTC_s003
                                 + 0.699709 w_k1_KO_s003
                                 + 0.468742 w_k1_MAT_s003
                                 + 0.834107 w_k1_MSFT_s003
                                 + 0.774797 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.972916 w_k1_ADM_s003
                                 + 1.01681 w_k1_AXP_s003
                                 + 0.832946 w_k1_DIS_s003
                                 + 0.853752 w_k1_FDX_s003
                                 + 1.09858 w_k1_IBM_s003
                                 + 0.857684 w_k1_INTC_s003
                                 + 0.898064 w_k1_KO_s003
                                 + 0.856717 w_k1_MAT_s003
                                 + 0.800017 w_k1_MSFT_s003
                                 + 0.944035 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.47766 w_k1_ADM_s004
                                 + 1.85655 w_k1_AXP_s004
                                 + 1.45315 w_k1_DIS_s004
                                 + 1.25428 w_k1_FDX_s004
                                 + 1.47835 w_k1_IBM_s004
                                 + 1.48606 w_k1_INTC_s004
                                 + 1.38888 w_k1_KO_s004 + 1.62511 w_k1_MAT_s004
                                 + 1.30687 w_k1_MSFT_s004
                                 + 1.04185 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.00163 w_k1_ADM_s004
                                 + 0.929992 w_k1_AXP_s004
                                 + 1.24454 w_k1_DIS_s004
                                 + 1.00449 w_k1_FDX_s004
                                 + 1.24138 w_k1_IBM_s004
                                 + 1.38517 w_k1_INTC_s004
                                 + 1.29182 w_k1_KO_s004
                                 + 0.581779 w_k1_MAT_s004
                                 + 1.27433 w_k1_MSFT_s004
                                 + 0.975198 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.17238 w_k1_ADM_s004
                                 + 1.18629 w_k1_AXP_s004
                                 + 0.684362 w_k1_DIS_s004
                                 + 1.04315 w_k1_FDX_s004
                                 + 0.690384 w_k1_IBM_s004
                                 + 1.10769 w_k1_INTC_s004
                                 + 0.812311 w_k1_KO_s004
                                 + 0.750592 w_k1_MAT_s004
                                 + 0.963351 w_k1_MSFT_s004
                                 + 0.836994 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.01292 w_k1_ADM_s004
                                 + 1.26228 w_k1_AXP_s004
                                 + 1.03326 w_k1_DIS_s004
                                 + 1.11026 w_k1_FDX_s004
                                 + 1.02735 w_k1_IBM_s004
                                 + 1.24054 w_k1_INTC_s004
                                 + 0.911021 w_k1_KO_s004
                                 + 0.938507 w_k1_MAT_s004
                                 + 1.05741 w_k1_MSFT_s004
                                 + 0.943047 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.691446 w_k1_ADM_s004
                                 + 0.847091 w_k1_AXP_s004
                                 + 0.664581 w_k1_DIS_s004
                                 + 0.807196 w_k1_FDX_s004
                                 + 0.900697 w_k1_IBM_s004
                                 + 1.02621 w_k1_INTC_s004
                                 + 0.838967 w_k1_KO_s004
                                 + 0.812286 w_k1_MAT_s004
                                 + 0.861315 w_k1_MSFT_s004
                                 + 0.760185 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.994408 w_k1_ADM_s004
                                 + 1.10724 w_k1_AXP_s004
                                 + 0.994895 w_k1_DIS_s004
                                 + 0.91391 w_k1_FDX_s004
                                 + 1.49944 w_k1_IBM_s004
                                 + 1.15307 w_k1_INTC_s004
                                 + 1.21675 w_k1_KO_s004 + 1.34355 w_k1_MAT_s004
                                 + 0.88814 w_k1_MSFT_s004 + 1.0543 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.766208 w_k1_ADM_s004
                                 + 1.02896 w_k1_AXP_s004
                                 + 0.961748 w_k1_DIS_s004
                                 + 0.895644 w_k1_FDX_s004 + 0.877 w_k1_IBM_s004
                                 + 1.10637 w_k1_INTC_s004
                                 + 0.969864 w_k1_KO_s004
                                 + 0.765287 w_k1_MAT_s004
                                 + 0.893975 w_k1_MSFT_s004
                                 + 0.894949 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.03642 w_k1_ADM_s004
                                 + 1.17881 w_k1_AXP_s004
                                 + 1.38217 w_k1_DIS_s004
                                 + 1.33335 w_k1_FDX_s004
                                 + 0.738001 w_k1_IBM_s004
                                 + 1.48607 w_k1_INTC_s004
                                 + 0.938019 w_k1_KO_s004
                                 + 0.85017 w_k1_MAT_s004
                                 + 1.22545 w_k1_MSFT_s004
                                 + 0.893971 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.0854 w_k1_ADM_s005
                                 + 1.11545 w_k1_AXP_s005 + 1.2933 w_k1_DIS_s005
                                 + 1.03397 w_k1_FDX_s005
                                 + 1.44737 w_k1_IBM_s005
                                 + 0.987025 w_k1_INTC_s005
                                 + 1.3353 w_k1_KO_s005 + 1.41356 w_k1_MAT_s005
                                 + 1.1036 w_k1_MSFT_s005 + 1.11582 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.759681 w_k1_ADM_s005
                                 + 0.808021 w_k1_AXP_s005
                                 + 0.960834 w_k1_DIS_s005
                                 + 0.78683 w_k1_FDX_s005
                                 + 1.21785 w_k1_IBM_s005
                                 + 0.983068 w_k1_INTC_s005
                                 + 1.12594 w_k1_KO_s005 + 1.24114 w_k1_MAT_s005
                                 + 0.983583 w_k1_MSFT_s005
                                 + 1.03161 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.713764 w_k1_ADM_s005
                                 + 0.795417 w_k1_AXP_s005
                                 + 0.56636 w_k1_DIS_s005
                                 + 0.811258 w_k1_FDX_s005
                                 + 0.678309 w_k1_IBM_s005
                                 + 0.809273 w_k1_INTC_s005
                                 + 0.626053 w_k1_KO_s005
                                 + 0.783782 w_k1_MAT_s005
                                 + 0.695661 w_k1_MSFT_s005
                                 + 0.804819 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.935479 w_k1_ADM_s005
                                 + 0.962689 w_k1_AXP_s005
                                 + 0.850273 w_k1_DIS_s005
                                 + 0.929534 w_k1_FDX_s005
                                 + 0.93828 w_k1_IBM_s005
                                 + 0.901165 w_k1_INTC_s005
                                 + 0.823555 w_k1_KO_s005
                                 + 1.20487 w_k1_MAT_s005
                                 + 0.888832 w_k1_MSFT_s005
                                 + 1.10096 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.0951 w_k1_ADM_s005
                                 + 0.973623 w_k1_AXP_s005
                                 + 1.00063 w_k1_DIS_s005
                                 + 1.03834 w_k1_FDX_s005
                                 + 0.881203 w_k1_IBM_s005
                                 + 1.08333 w_k1_INTC_s005
                                 + 0.975053 w_k1_KO_s005
                                 + 0.607776 w_k1_MAT_s005
                                 + 0.954879 w_k1_MSFT_s005
                                 + 0.935411 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.806837 w_k1_ADM_s005
                                 + 0.769266 w_k1_AXP_s005
                                 + 0.77927 w_k1_DIS_s005
                                 + 0.821915 w_k1_FDX_s005
                                 + 0.888572 w_k1_IBM_s005
                                 + 0.871625 w_k1_INTC_s005
                                 + 0.808989 w_k1_KO_s005 + 0.8547 w_k1_MAT_s005
                                 + 0.802189 w_k1_MSFT_s005
                                 + 0.932262 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.10105 w_k1_ADM_s005
                                 + 1.16826 w_k1_AXP_s005
                                 + 1.31224 w_k1_DIS_s005
                                 + 1.27578 w_k1_FDX_s005
                                 + 1.08961 w_k1_IBM_s005
                                 + 1.10428 w_k1_INTC_s005
                                 + 1.13106 w_k1_KO_s005 + 1.06715 w_k1_MAT_s005
                                 + 1.14186 w_k1_MSFT_s005
                                 + 1.15952 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.13149 w_k1_ADM_s005
                                 + 1.24286 w_k1_AXP_s005 + 1.1439 w_k1_DIS_s005
                                 + 1.3427 w_k1_FDX_s005
                                 + 0.810672 w_k1_IBM_s005
                                 + 1.17746 w_k1_INTC_s005
                                 + 1.02395 w_k1_KO_s005 + 1.02438 w_k1_MAT_s005
                                 + 1.15379 w_k1_MSFT_s005
                                 + 1.18737 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.963696 w_k1_ADM_s006
                                 + 1.0035 w_k1_AXP_s006 + 1.02511 w_k1_DIS_s006
                                 + 1.15442 w_k1_FDX_s006
                                 + 0.591987 w_k1_IBM_s006
                                 + 1.16857 w_k1_INTC_s006
                                 + 0.731979 w_k1_KO_s006
                                 + 0.796993 w_k1_MAT_s006
                                 + 1.01815 w_k1_MSFT_s006
                                 + 0.925802 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.17486 w_k1_ADM_s006
                                 + 1.16056 w_k1_AXP_s006
                                 + 1.14408 w_k1_DIS_s006
                                 + 1.07788 w_k1_FDX_s006
                                 + 1.12797 w_k1_IBM_s006
                                 + 0.980102 w_k1_INTC_s006
                                 + 1.03788 w_k1_KO_s006 + 1.48579 w_k1_MAT_s006
                                 + 1.04656 w_k1_MSFT_s006
                                 + 1.10975 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.25644 w_k1_ADM_s006
                                 + 1.57029 w_k1_AXP_s006
                                 + 1.68029 w_k1_DIS_s006
                                 + 1.51038 w_k1_FDX_s006
                                 + 0.941131 w_k1_IBM_s006
                                 + 1.34398 w_k1_INTC_s006
                                 + 1.07651 w_k1_KO_s006 + 1.04913 w_k1_MAT_s006
                                 + 1.2805 w_k1_MSFT_s006 + 1.12513 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.26791 w_k1_ADM_s006
                                 + 1.40254 w_k1_AXP_s006
                                 + 1.70526 w_k1_DIS_s006
                                 + 1.27838 w_k1_FDX_s006
                                 + 1.56986 w_k1_IBM_s006
                                 + 1.33699 w_k1_INTC_s006
                                 + 1.34061 w_k1_KO_s006 + 1.12026 w_k1_MAT_s006
                                 + 1.19561 w_k1_MSFT_s006
                                 + 1.29065 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.774528 w_k1_ADM_s006
                                 + 0.723177 w_k1_AXP_s006
                                 + 1.10988 w_k1_DIS_s006
                                 + 0.888709 w_k1_FDX_s006
                                 + 0.834165 w_k1_IBM_s006
                                 + 0.92728 w_k1_INTC_s006
                                 + 0.931309 w_k1_KO_s006
                                 + 0.927997 w_k1_MAT_s006
                                 + 0.977551 w_k1_MSFT_s006
                                 + 1.01426 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.1124 w_k1_ADM_s006
                                 + 0.91022 w_k1_AXP_s006
                                 + 1.26767 w_k1_DIS_s006
                                 + 0.923847 w_k1_FDX_s006
                                 + 1.42701 w_k1_IBM_s006
                                 + 0.862241 w_k1_INTC_s006
                                 + 1.17868 w_k1_KO_s006 + 1.13213 w_k1_MAT_s006
                                 + 0.91183 w_k1_MSFT_s006
                                 + 0.925205 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.868704 w_k1_ADM_s006
                                 + 1.10357 w_k1_AXP_s006
                                 + 0.924353 w_k1_DIS_s006
                                 + 1.05172 w_k1_FDX_s006
                                 + 0.906097 w_k1_IBM_s006
                                 + 1.03044 w_k1_INTC_s006
                                 + 0.751063 w_k1_KO_s006
                                 + 0.752243 w_k1_MAT_s006
                                 + 1.01708 w_k1_MSFT_s006
                                 + 0.985545 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.833282 w_k1_ADM_s006
                                 + 0.720717 w_k1_AXP_s006
                                 + 0.672519 w_k1_DIS_s006
                                 + 0.758643 w_k1_FDX_s006
                                 + 0.981427 w_k1_IBM_s006
                                 + 0.695713 w_k1_INTC_s006
                                 + 0.700132 w_k1_KO_s006
                                 + 1.00582 w_k1_MAT_s006
                                 + 0.72803 w_k1_MSFT_s006
                                 + 0.885997 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.15716 w_k1_ADM_s007
                                 + 1.22074 w_k1_AXP_s007
                                 + 0.950532 w_k1_DIS_s007
                                 + 0.987125 w_k1_FDX_s007
                                 + 1.1461 w_k1_IBM_s007
                                 + 1.08731 w_k1_INTC_s007
                                 + 1.14153 w_k1_KO_s007 + 1.36895 w_k1_MAT_s007
                                 + 0.80858 w_k1_MSFT_s007
                                 + 1.05747 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.00018 w_k1_ADM_s007
                                 + 0.781683 w_k1_AXP_s007
                                 + 0.75912 w_k1_DIS_s007
                                 + 0.935748 w_k1_FDX_s007
                                 + 0.709544 w_k1_IBM_s007
                                 + 0.791587 w_k1_INTC_s007
                                 + 0.72961 w_k1_KO_s007
                                 + 0.461759 w_k1_MAT_s007
                                 + 0.733091 w_k1_MSFT_s007
                                 + 0.830418 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.20676 w_k1_ADM_s007
                                 + 1.123 w_k1_AXP_s007 + 1.17343 w_k1_DIS_s007
                                 + 1.09894 w_k1_FDX_s007
                                 + 0.83411 w_k1_IBM_s007
                                 + 1.14176 w_k1_INTC_s007
                                 + 0.924671 w_k1_KO_s007
                                 + 0.955214 w_k1_MAT_s007
                                 + 1.04399 w_k1_MSFT_s007
                                 + 0.974787 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.726695 w_k1_ADM_s007
                                 + 0.791251 w_k1_AXP_s007
                                 + 0.61454 w_k1_DIS_s007
                                 + 0.711571 w_k1_FDX_s007
                                 + 0.759763 w_k1_IBM_s007
                                 + 0.726355 w_k1_INTC_s007
                                 + 0.676672 w_k1_KO_s007
                                 + 1.24732 w_k1_MAT_s007
                                 + 0.681924 w_k1_MSFT_s007
                                 + 0.820575 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.05061 w_k1_ADM_s007
                                 + 0.977852 w_k1_AXP_s007
                                 + 1.24818 w_k1_DIS_s007
                                 + 1.06581 w_k1_FDX_s007
                                 + 1.21506 w_k1_IBM_s007
                                 + 0.971268 w_k1_INTC_s007
                                 + 1.07778 w_k1_KO_s007 + 1.55313 w_k1_MAT_s007
                                 + 1.07209 w_k1_MSFT_s007
                                 + 1.09002 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.40097 w_k1_ADM_s007
                                 + 1.63391 w_k1_AXP_s007
                                 + 1.66277 w_k1_DIS_s007 + 1.3445 w_k1_FDX_s007
                                 + 1.25489 w_k1_IBM_s007
                                 + 1.21725 w_k1_INTC_s007
                                 + 1.27076 w_k1_KO_s007 + 1.41889 w_k1_MAT_s007
                                 + 1.1456 w_k1_MSFT_s007 + 1.19383 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.01346 w_k1_ADM_s007
                                 + 0.85421 w_k1_AXP_s007
                                 + 0.75981 w_k1_DIS_s007
                                 + 0.813372 w_k1_FDX_s007
                                 + 1.18423 w_k1_IBM_s007
                                 + 0.797106 w_k1_INTC_s007
                                 + 0.892559 w_k1_KO_s007
                                 + 1.18905 w_k1_MAT_s007
                                 + 0.738598 w_k1_MSFT_s007
                                 + 0.945676 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.2115 w_k1_ADM_s007
                                 + 1.43454 w_k1_AXP_s007
                                 + 1.05473 w_k1_DIS_s007
                                 + 0.99695 w_k1_FDX_s007
                                 + 1.64657 w_k1_IBM_s007
                                 + 0.955799 w_k1_INTC_s007
                                 + 1.17227 w_k1_KO_s007 + 2.02983 w_k1_MAT_s007
                                 + 0.919825 w_k1_MSFT_s007
                                 + 1.19584 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.7889 w_k1_ADM_s010
                                 + 0.700329 w_k1_AXP_s010
                                 + 0.604402 w_k1_DIS_s010
                                 + 0.744094 w_k1_FDX_s010
                                 + 0.828832 w_k1_IBM_s010
                                 + 0.890916 w_k1_INTC_s010
                                 + 0.765903 w_k1_KO_s010
                                 + 0.819818 w_k1_MAT_s010
                                 + 0.878166 w_k1_MSFT_s010
                                 + 0.92445 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.815278 w_k1_ADM_s010
                                 + 0.923901 w_k1_AXP_s010
                                 + 0.922957 w_k1_DIS_s010
                                 + 1.00244 w_k1_FDX_s010
                                 + 0.575172 w_k1_IBM_s010
                                 + 0.78743 w_k1_INTC_s010
                                 + 0.70702 w_k1_KO_s010
                                 + 0.616337 w_k1_MAT_s010
                                 + 0.99627 w_k1_MSFT_s010
                                 + 0.929028 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.44208 w_k1_ADM_s010
                                 + 1.64246 w_k1_AXP_s010 + 1.4752 w_k1_DIS_s010
                                 + 1.35639 w_k1_FDX_s010
                                 + 1.07069 w_k1_IBM_s010
                                 + 1.47532 w_k1_INTC_s010
                                 + 1.26629 w_k1_KO_s010 + 1.07076 w_k1_MAT_s010
                                 + 1.61526 w_k1_MSFT_s010
                                 + 1.30594 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.32562 w_k1_ADM_s010
                                 + 1.05858 w_k1_AXP_s010
                                 + 1.08756 w_k1_DIS_s010 + 1.0593 w_k1_FDX_s010
                                 + 0.816945 w_k1_IBM_s010
                                 + 1.05039 w_k1_INTC_s010
                                 + 1.01966 w_k1_KO_s010
                                 + 0.711767 w_k1_MAT_s010
                                 + 1.23837 w_k1_MSFT_s010
                                 + 1.04566 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.07486 w_k1_ADM_s010
                                 + 1.18457 w_k1_AXP_s010
                                 + 1.29491 w_k1_DIS_s010
                                 + 1.13195 w_k1_FDX_s010
                                 + 1.35588 w_k1_IBM_s010
                                 + 1.12422 w_k1_INTC_s010
                                 + 1.29351 w_k1_KO_s010 + 1.30057 w_k1_MAT_s010
                                 + 1.1458 w_k1_MSFT_s010 + 1.18743 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.962062 w_k1_ADM_s010
                                 + 1.38264 w_k1_AXP_s010
                                 + 0.856614 w_k1_DIS_s010
                                 + 1.10859 w_k1_FDX_s010
                                 + 0.74932 w_k1_IBM_s010
                                 + 1.05946 w_k1_INTC_s010
                                 + 0.780723 w_k1_KO_s010
                                 + 1.11327 w_k1_MAT_s010
                                 + 1.03789 w_k1_MSFT_s010
                                 + 0.991165 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.824881 w_k1_ADM_s010
                                 + 1.01867 w_k1_AXP_s010
                                 + 1.00675 w_k1_DIS_s010
                                 + 0.895159 w_k1_FDX_s010
                                 + 1.71448 w_k1_IBM_s010
                                 + 0.854753 w_k1_INTC_s010
                                 + 1.08578 w_k1_KO_s010 + 1.54416 w_k1_MAT_s010
                                 + 0.93905 w_k1_MSFT_s010
                                 + 1.04543 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.861353 w_k1_ADM_s010
                                 + 0.784786 w_k1_AXP_s010
                                 + 0.810382 w_k1_DIS_s010
                                 + 0.845545 w_k1_FDX_s010
                                 + 1.10329 w_k1_IBM_s010
                                 + 0.927535 w_k1_INTC_s010
                                 + 0.992769 w_k1_KO_s010
                                 + 0.981081 w_k1_MAT_s010
                                 + 0.945185 w_k1_MSFT_s010
                                 + 1.09543 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.93013 w_k1_ADM_s011
                                 + 1.12405 w_k1_AXP_s011
                                 + 1.55467 w_k1_DIS_s011
                                 + 1.11343 w_k1_FDX_s011
                                 + 1.33473 w_k1_IBM_s011
                                 + 0.953091 w_k1_INTC_s011
                                 + 1.26473 w_k1_KO_s011 + 1.1702 w_k1_MAT_s011
                                 + 1.03061 w_k1_MSFT_s011
                                 + 1.25141 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.17978 w_k1_ADM_s011
                                 + 1.50337 w_k1_AXP_s011 + 1.6182 w_k1_DIS_s011
                                 + 1.4468 w_k1_FDX_s011 + 1.06843 w_k1_IBM_s011
                                 + 1.26268 w_k1_INTC_s011
                                 + 1.25074 w_k1_KO_s011 + 1.27826 w_k1_MAT_s011
                                 + 1.36125 w_k1_MSFT_s011 + 1.3346 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.915808 w_k1_ADM_s011
                                 + 1.26203 w_k1_AXP_s011
                                 + 1.23851 w_k1_DIS_s011
                                 + 1.04782 w_k1_FDX_s011
                                 + 1.57206 w_k1_IBM_s011
                                 + 0.894546 w_k1_INTC_s011
                                 + 1.36531 w_k1_KO_s011 + 1.77106 w_k1_MAT_s011
                                 + 0.962143 w_k1_MSFT_s011
                                 + 1.30521 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.17906 w_k1_ADM_s011
                                 + 1.55001 w_k1_AXP_s011
                                 + 1.10011 w_k1_DIS_s011
                                 + 1.15863 w_k1_FDX_s011
                                 + 1.23869 w_k1_IBM_s011
                                 + 0.872014 w_k1_INTC_s011
                                 + 1.0827 w_k1_KO_s011 + 1.50931 w_k1_MAT_s011
                                 + 0.831511 w_k1_MSFT_s011
                                 + 1.10833 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.0571 w_k1_ADM_s011
                                 + 0.770065 w_k1_AXP_s011
                                 + 0.777705 w_k1_DIS_s011
                                 + 0.868903 w_k1_FDX_s011
                                 + 0.692304 w_k1_IBM_s011
                                 + 0.740303 w_k1_INTC_s011
                                 + 0.863073 w_k1_KO_s011
                                 + 0.684518 w_k1_MAT_s011
                                 + 0.787056 w_k1_MSFT_s011
                                 + 1.08377 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.754812 w_k1_ADM_s011
                                 + 0.751772 w_k1_AXP_s011
                                 + 0.728124 w_k1_DIS_s011
                                 + 0.763024 w_k1_FDX_s011
                                 + 1.09954 w_k1_IBM_s011
                                 + 0.787975 w_k1_INTC_s011
                                 + 0.820404 w_k1_KO_s011
                                 + 0.850308 w_k1_MAT_s011
                                 + 0.696626 w_k1_MSFT_s011
                                 + 0.938453 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.27488 w_k1_ADM_s011
                                 + 1.08079 w_k1_AXP_s011
                                 + 1.06603 w_k1_DIS_s011
                                 + 1.16201 w_k1_FDX_s011
                                 + 0.874404 w_k1_IBM_s011
                                 + 1.08762 w_k1_INTC_s011
                                 + 0.985528 w_k1_KO_s011
                                 + 0.953949 w_k1_MAT_s011
                                 + 1.01095 w_k1_MSFT_s011
                                 + 1.24194 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.842201 w_k1_ADM_s011
                                 + 0.849987 w_k1_AXP_s011
                                 + 1.00985 w_k1_DIS_s011
                                 + 0.836374 w_k1_FDX_s011
                                 + 1.60117 w_k1_IBM_s011
                                 + 0.893737 w_k1_INTC_s011
                                 + 1.18275 w_k1_KO_s011 + 1.43083 w_k1_MAT_s011
                                 + 0.835817 w_k1_MSFT_s011
                                 + 1.09249 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.912322 w_k1_ADM_s012
                                 + 0.84386 w_k1_AXP_s012
                                 + 0.769607 w_k1_DIS_s012
                                 + 0.969708 w_k1_FDX_s012
                                 + 0.671502 w_k1_IBM_s012
                                 + 0.881868 w_k1_INTC_s012
                                 + 0.759588 w_k1_KO_s012
                                 + 0.898189 w_k1_MAT_s012
                                 + 0.834495 w_k1_MSFT_s012
                                 + 0.959253 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.1649 w_k1_ADM_s012
                                 + 1.20774 w_k1_AXP_s012
                                 + 1.00281 w_k1_DIS_s012
                                 + 1.11023 w_k1_FDX_s012 + 1.4047 w_k1_IBM_s012
                                 + 1.07468 w_k1_INTC_s012
                                 + 0.972368 w_k1_KO_s012 + 1.4251 w_k1_MAT_s012
                                 + 0.976198 w_k1_MSFT_s012
                                 + 1.04972 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.867653 w_k1_ADM_s012
                                 + 0.787922 w_k1_AXP_s012
                                 + 0.903541 w_k1_DIS_s012
                                 + 0.90831 w_k1_FDX_s012
                                 + 1.24026 w_k1_IBM_s012
                                 + 1.03999 w_k1_INTC_s012
                                 + 1.06452 w_k1_KO_s012 + 1.13803 w_k1_MAT_s012
                                 + 0.833819 w_k1_MSFT_s012
                                 + 0.954465 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.962621 w_k1_ADM_s012
                                 + 1.18993 w_k1_AXP_s012
                                 + 1.37205 w_k1_DIS_s012
                                 + 1.29275 w_k1_FDX_s012
                                 + 1.03338 w_k1_IBM_s012
                                 + 1.47877 w_k1_INTC_s012
                                 + 1.22269 w_k1_KO_s012 + 1.11036 w_k1_MAT_s012
                                 + 1.23513 w_k1_MSFT_s012
                                 + 1.17298 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.830049 w_k1_ADM_s012
                                 + 0.590072 w_k1_AXP_s012
                                 + 0.784468 w_k1_DIS_s012
                                 + 0.850191 w_k1_FDX_s012
                                 + 0.90999 w_k1_IBM_s012
                                 + 0.938622 w_k1_INTC_s012
                                 + 0.901283 w_k1_KO_s012
                                 + 0.769705 w_k1_MAT_s012
                                 + 0.884236 w_k1_MSFT_s012
                                 + 0.773832 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.33167 w_k1_ADM_s012
                                 + 1.23561 w_k1_AXP_s012 + 1.129 w_k1_DIS_s012
                                 + 1.22722 w_k1_FDX_s012
                                 + 0.803194 w_k1_IBM_s012
                                 + 1.18752 w_k1_INTC_s012
                                 + 0.905745 w_k1_KO_s012
                                 + 0.869279 w_k1_MAT_s012
                                 + 0.960818 w_k1_MSFT_s012
                                 + 0.934345 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.88886 w_k1_ADM_s012
                                 + 0.872748 w_k1_AXP_s012
                                 + 0.724045 w_k1_DIS_s012
                                 + 0.88721 w_k1_FDX_s012
                                 + 0.991413 w_k1_IBM_s012
                                 + 1.00983 w_k1_INTC_s012
                                 + 0.838283 w_k1_KO_s012
                                 + 1.08116 w_k1_MAT_s012
                                 + 0.841933 w_k1_MSFT_s012
                                 + 0.882173 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.39504 w_k1_ADM_s012
                                 + 1.04831 w_k1_AXP_s012
                                 + 1.30368 w_k1_DIS_s012
                                 + 1.06678 w_k1_FDX_s012
                                 + 1.74698 w_k1_IBM_s012
                                 + 1.23946 w_k1_INTC_s012
                                 + 1.32088 w_k1_KO_s012 + 1.48979 w_k1_MAT_s012
                                 + 1.17686 w_k1_MSFT_s012
                                 + 1.08567 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.19222 w_k1_ADM_s013
                                 + 1.12981 w_k1_AXP_s013
                                 + 1.22363 w_k1_DIS_s013 + 1.0366 w_k1_FDX_s013
                                 + 1.51402 w_k1_IBM_s013
                                 + 1.09742 w_k1_INTC_s013
                                 + 1.35934 w_k1_KO_s013 + 1.42172 w_k1_MAT_s013
                                 + 1.12502 w_k1_MSFT_s013
                                 + 1.10229 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.688364 w_k1_ADM_s013
                                 + 0.572467 w_k1_AXP_s013
                                 + 0.602407 w_k1_DIS_s013
                                 + 0.749251 w_k1_FDX_s013
                                 + 0.739654 w_k1_IBM_s013
                                 + 0.834863 w_k1_INTC_s013
                                 + 0.873653 w_k1_KO_s013
                                 + 0.689995 w_k1_MAT_s013
                                 + 0.682318 w_k1_MSFT_s013
                                 + 0.895905 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.921598 w_k1_ADM_s013
                                 + 0.793168 w_k1_AXP_s013
                                 + 0.860591 w_k1_DIS_s013
                                 + 0.892514 w_k1_FDX_s013
                                 + 0.942553 w_k1_IBM_s013
                                 + 0.868437 w_k1_INTC_s013
                                 + 0.9667 w_k1_KO_s013 + 1.25601 w_k1_MAT_s013
                                 + 0.954212 w_k1_MSFT_s013
                                 + 1.03274 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.20601 w_k1_ADM_s013
                                 + 1.11893 w_k1_AXP_s013
                                 + 1.03814 w_k1_DIS_s013
                                 + 1.06443 w_k1_FDX_s013
                                 + 1.10324 w_k1_IBM_s013
                                 + 0.864484 w_k1_INTC_s013
                                 + 1.02692 w_k1_KO_s013 + 1.28741 w_k1_MAT_s013
                                 + 0.949834 w_k1_MSFT_s013
                                 + 0.992062 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.43569 w_k1_ADM_s013
                                 + 1.45139 w_k1_AXP_s013
                                 + 1.50036 w_k1_DIS_s013 + 1.3897 w_k1_FDX_s013
                                 + 1.0071 w_k1_IBM_s013
                                 + 1.24635 w_k1_INTC_s013
                                 + 1.21825 w_k1_KO_s013 + 1.23285 w_k1_MAT_s013
                                 + 1.1905 w_k1_MSFT_s013 + 1.21512 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.88827 w_k1_ADM_s013
                                 + 0.912029 w_k1_AXP_s013
                                 + 0.632519 w_k1_DIS_s013
                                 + 0.794061 w_k1_FDX_s013
                                 + 0.945838 w_k1_IBM_s013
                                 + 0.968123 w_k1_INTC_s013
                                 + 0.943256 w_k1_KO_s013
                                 + 0.978717 w_k1_MAT_s013
                                 + 0.722143 w_k1_MSFT_s013
                                 + 0.908368 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.44847 w_k1_ADM_s013
                                 + 1.15523 w_k1_AXP_s013
                                 + 1.01357 w_k1_DIS_s013
                                 + 1.15285 w_k1_FDX_s013
                                 + 0.691709 w_k1_IBM_s013
                                 + 0.985844 w_k1_INTC_s013
                                 + 0.891285 w_k1_KO_s013
                                 + 0.867809 w_k1_MAT_s013
                                 + 1.0247 w_k1_MSFT_s013
                                 + 0.996725 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.898573 w_k1_ADM_s013
                                 + 1.18405 w_k1_AXP_s013
                                 + 0.690779 w_k1_DIS_s013
                                 + 0.84738 w_k1_FDX_s013
                                 + 1.21403 w_k1_IBM_s013
                                 + 0.941571 w_k1_INTC_s013
                                 + 1.09983 w_k1_KO_s013 + 1.86981 w_k1_MAT_s013
                                 + 0.834776 w_k1_MSFT_s013
                                 + 0.957311 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.41811 w_k1_ADM_s014
                                 + 1.21228 w_k1_AXP_s014
                                 + 1.28486 w_k1_DIS_s014 + 1.221 w_k1_FDX_s014
                                 + 0.790433 w_k1_IBM_s014
                                 + 1.46852 w_k1_INTC_s014
                                 + 1.06114 w_k1_KO_s014
                                 + 0.725838 w_k1_MAT_s014
                                 + 1.26328 w_k1_MSFT_s014 + 1.1959 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.97064 w_k1_ADM_s014
                                 + 0.95429 w_k1_AXP_s014
                                 + 0.82205 w_k1_DIS_s014
                                 + 0.947645 w_k1_FDX_s014
                                 + 0.765232 w_k1_IBM_s014
                                 + 1.16073 w_k1_INTC_s014
                                 + 0.92299 w_k1_KO_s014
                                 + 0.959351 w_k1_MAT_s014
                                 + 1.07146 w_k1_MSFT_s014
                                 + 0.974751 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.914115 w_k1_ADM_s014
                                 + 0.642652 w_k1_AXP_s014
                                 + 0.616865 w_k1_DIS_s014
                                 + 0.840204 w_k1_FDX_s014
                                 + 0.667744 w_k1_IBM_s014
                                 + 0.897216 w_k1_INTC_s014
                                 + 0.730405 w_k1_KO_s014
                                 + 0.64462 w_k1_MAT_s014
                                 + 0.822141 w_k1_MSFT_s014
                                 + 0.812442 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.32552 w_k1_ADM_s014
                                 + 1.18037 w_k1_AXP_s014
                                 + 1.06415 w_k1_DIS_s014
                                 + 1.15399 w_k1_FDX_s014
                                 + 1.07409 w_k1_IBM_s014
                                 + 1.18771 w_k1_INTC_s014
                                 + 1.12183 w_k1_KO_s014 + 1.10814 w_k1_MAT_s014
                                 + 1.11217 w_k1_MSFT_s014
                                 + 1.17478 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.898788 w_k1_ADM_s014
                                 + 0.907338 w_k1_AXP_s014
                                 + 0.981201 w_k1_DIS_s014
                                 + 0.81362 w_k1_FDX_s014 + 1.3475 w_k1_IBM_s014
                                 + 1.43986 w_k1_INTC_s014
                                 + 1.22445 w_k1_KO_s014 + 1.28958 w_k1_MAT_s014
                                 + 1.11389 w_k1_MSFT_s014
                                 + 0.955216 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.863487 w_k1_ADM_s014
                                 + 0.747545 w_k1_AXP_s014
                                 + 0.816399 w_k1_DIS_s014
                                 + 0.85144 w_k1_FDX_s014
                                 + 0.999396 w_k1_IBM_s014
                                 + 1.09813 w_k1_INTC_s014
                                 + 0.913253 w_k1_KO_s014
                                 + 0.969322 w_k1_MAT_s014
                                 + 1.0676 w_k1_MSFT_s014
                                 + 0.891718 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.17228 w_k1_ADM_s014
                                 + 1.51452 w_k1_AXP_s014
                                 + 1.36879 w_k1_DIS_s014
                                 + 1.27599 w_k1_FDX_s014
                                 + 2.00289 w_k1_IBM_s014
                                 + 1.69922 w_k1_INTC_s014
                                 + 1.60262 w_k1_KO_s014 + 1.96026 w_k1_MAT_s014
                                 + 1.49306 w_k1_MSFT_s014
                                 + 1.42806 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.19409 w_k1_ADM_s014
                                 + 1.40907 w_k1_AXP_s014
                                 + 1.30554 w_k1_DIS_s014
                                 + 1.29122 w_k1_FDX_s014
                                 + 1.41907 w_k1_IBM_s014
                                 + 1.59454 w_k1_INTC_s014
                                 + 1.16904 w_k1_KO_s014 + 1.62223 w_k1_MAT_s014
                                 + 1.35788 w_k1_MSFT_s014
                                 + 1.11393 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.769685 w_k1_ADM_s015
                                 + 0.84241 w_k1_AXP_s015
                                 + 0.93798 w_k1_DIS_s015
                                 + 0.930533 w_k1_FDX_s015
                                 + 0.898832 w_k1_IBM_s015
                                 + 1.02961 w_k1_INTC_s015
                                 + 0.940615 w_k1_KO_s015
                                 + 0.63688 w_k1_MAT_s015
                                 + 0.926826 w_k1_MSFT_s015
                                 + 0.856028 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.37911 w_k1_ADM_s015
                                 + 1.16226 w_k1_AXP_s015 + 1.0351 w_k1_DIS_s015
                                 + 1.07594 w_k1_FDX_s015
                                 + 0.987767 w_k1_IBM_s015
                                 + 1.10231 w_k1_INTC_s015
                                 + 1.14264 w_k1_KO_s015 + 1.16838 w_k1_MAT_s015
                                 + 1.13208 w_k1_MSFT_s015
                                 + 1.06215 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.2238 w_k1_ADM_s015
                                 + 1.06939 w_k1_AXP_s015
                                 + 1.06575 w_k1_DIS_s015
                                 + 0.977067 w_k1_FDX_s015
                                 + 1.44281 w_k1_IBM_s015
                                 + 1.17532 w_k1_INTC_s015
                                 + 1.32579 w_k1_KO_s015 + 1.0189 w_k1_MAT_s015
                                 + 1.16843 w_k1_MSFT_s015
                                 + 1.11533 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.974751 w_k1_ADM_s015
                                 + 0.741338 w_k1_AXP_s015
                                 + 0.711331 w_k1_DIS_s015
                                 + 0.855814 w_k1_FDX_s015
                                 + 0.701009 w_k1_IBM_s015
                                 + 0.833942 w_k1_INTC_s015
                                 + 0.883611 w_k1_KO_s015
                                 + 0.800459 w_k1_MAT_s015
                                 + 0.910284 w_k1_MSFT_s015
                                 + 0.924366 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.08995 w_k1_ADM_s015
                                 + 0.834726 w_k1_AXP_s015
                                 + 0.777916 w_k1_DIS_s015
                                 + 0.906097 w_k1_FDX_s015
                                 + 1.13048 w_k1_IBM_s015
                                 + 1.04674 w_k1_INTC_s015
                                 + 1.02008 w_k1_KO_s015 + 1.25428 w_k1_MAT_s015
                                 + 0.99444 w_k1_MSFT_s015
                                 + 0.960103 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.01083 w_k1_ADM_s015
                                 + 1.39833 w_k1_AXP_s015
                                 + 1.22183 w_k1_DIS_s015
                                 + 1.02995 w_k1_FDX_s015
                                 + 1.68542 w_k1_IBM_s015
                                 + 1.14356 w_k1_INTC_s015
                                 + 1.38791 w_k1_KO_s015 + 1.97532 w_k1_MAT_s015
                                 + 1.119 w_k1_MSFT_s015 + 1.32344 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.86841 w_k1_ADM_s015
                                 + 0.908374 w_k1_AXP_s015
                                 + 1.24839 w_k1_DIS_s015
                                 + 1.06872 w_k1_FDX_s015
                                 + 1.07795 w_k1_IBM_s015
                                 + 1.11441 w_k1_INTC_s015
                                 + 1.06542 w_k1_KO_s015
                                 + 0.923332 w_k1_MAT_s015
                                 + 1.10304 w_k1_MSFT_s015 + 1.0699 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.36537 w_k1_ADM_s015
                                 + 1.45673 w_k1_AXP_s015 + 1.4396 w_k1_DIS_s015
                                 + 1.25072 w_k1_FDX_s015
                                 + 1.16886 w_k1_IBM_s015
                                 + 0.944343 w_k1_INTC_s015
                                 + 1.26654 w_k1_KO_s015 + 1.39905 w_k1_MAT_s015
                                 + 1.28549 w_k1_MSFT_s015
                                 + 1.26834 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.06843 w_k1_ADM_s016
                                 + 0.938442 w_k1_AXP_s016
                                 + 0.980869 w_k1_DIS_s016
                                 + 0.846871 w_k1_FDX_s016
                                 + 1.31142 w_k1_IBM_s016
                                 + 0.837368 w_k1_INTC_s016
                                 + 1.30957 w_k1_KO_s016 + 1.31685 w_k1_MAT_s016
                                 + 0.87726 w_k1_MSFT_s016
                                 + 1.18075 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.03875 w_k1_ADM_s016
                                 + 0.797954 w_k1_AXP_s016
                                 + 0.746829 w_k1_DIS_s016
                                 + 0.851569 w_k1_FDX_s016
                                 + 0.835963 w_k1_IBM_s016
                                 + 0.701928 w_k1_INTC_s016
                                 + 0.859042 w_k1_KO_s016
                                 + 1.19921 w_k1_MAT_s016
                                 + 0.819055 w_k1_MSFT_s016
                                 + 1.04055 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.30018 w_k1_ADM_s016
                                 + 1.12627 w_k1_AXP_s016
                                 + 1.24163 w_k1_DIS_s016
                                 + 1.22427 w_k1_FDX_s016
                                 + 0.958112 w_k1_IBM_s016
                                 + 0.889206 w_k1_INTC_s016
                                 + 1.11639 w_k1_KO_s016 + 1.17153 w_k1_MAT_s016
                                 + 1.16994 w_k1_MSFT_s016
                                 + 1.21464 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.34039 w_k1_ADM_s016
                                 + 1.35255 w_k1_AXP_s016
                                 + 1.40227 w_k1_DIS_s016
                                 + 1.15469 w_k1_FDX_s016
                                 + 1.46004 w_k1_IBM_s016
                                 + 1.08584 w_k1_INTC_s016
                                 + 1.35085 w_k1_KO_s016 + 1.57307 w_k1_MAT_s016
                                 + 1.15626 w_k1_MSFT_s016
                                 + 1.21945 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.797091 w_k1_ADM_s016
                                 + 0.48659 w_k1_AXP_s016
                                 + 0.621823 w_k1_DIS_s016
                                 + 0.651426 w_k1_FDX_s016
                                 + 0.918598 w_k1_IBM_s016
                                 + 0.543308 w_k1_INTC_s016
                                 + 0.843354 w_k1_KO_s016
                                 + 0.940996 w_k1_MAT_s016
                                 + 0.79631 w_k1_MSFT_s016
                                 + 0.943979 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.0551 w_k1_ADM_s016
                                 + 0.769873 w_k1_AXP_s016
                                 + 0.894038 w_k1_DIS_s016
                                 + 0.859478 w_k1_FDX_s016
                                 + 1.05525 w_k1_IBM_s016
                                 + 0.78714 w_k1_INTC_s016 + 1.0575 w_k1_KO_s016
                                 + 1.09665 w_k1_MAT_s016
                                 + 0.932188 w_k1_MSFT_s016
                                 + 0.919081 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.11622 w_k1_ADM_s016
                                 + 0.767131 w_k1_AXP_s016
                                 + 0.784991 w_k1_DIS_s016
                                 + 0.908845 w_k1_FDX_s016
                                 + 0.679354 w_k1_IBM_s016
                                 + 0.840411 w_k1_INTC_s016
                                 + 0.827215 w_k1_KO_s016
                                 + 0.704521 w_k1_MAT_s016
                                 + 0.94014 w_k1_MSFT_s016
                                 + 1.05886 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.17053 w_k1_ADM_s016
                                 + 1.01664 w_k1_AXP_s016
                                 + 1.22198 w_k1_DIS_s016
                                 + 1.03684 w_k1_FDX_s016
                                 + 1.07885 w_k1_IBM_s016
                                 + 0.736654 w_k1_INTC_s016
                                 + 1.23294 w_k1_KO_s016 + 1.4116 w_k1_MAT_s016
                                 + 0.998485 w_k1_MSFT_s016
                                 + 1.16014 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.01288 w_k1_ADM_s017
                                 + 0.917985 w_k1_AXP_s017
                                 + 1.29923 w_k1_DIS_s017
                                 + 1.03434 w_k1_FDX_s017
                                 + 1.38993 w_k1_IBM_s017
                                 + 1.15413 w_k1_INTC_s017
                                 + 1.26321 w_k1_KO_s017 + 1.02483 w_k1_MAT_s017
                                 + 1.15688 w_k1_MSFT_s017
                                 + 1.10741 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.907353 w_k1_ADM_s017
                                 + 0.914357 w_k1_AXP_s017
                                 + 0.813463 w_k1_DIS_s017
                                 + 0.86715 w_k1_FDX_s017
                                 + 1.13516 w_k1_IBM_s017
                                 + 0.883714 w_k1_INTC_s017
                                 + 0.952454 w_k1_KO_s017
                                 + 1.21648 w_k1_MAT_s017
                                 + 0.878641 w_k1_MSFT_s017
                                 + 1.02436 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.04555 w_k1_ADM_s017
                                 + 0.780094 w_k1_AXP_s017
                                 + 0.938256 w_k1_DIS_s017
                                 + 0.9196 w_k1_FDX_s017
                                 + 0.907619 w_k1_IBM_s017
                                 + 0.949051 w_k1_INTC_s017
                                 + 0.993862 w_k1_KO_s017
                                 + 0.725804 w_k1_MAT_s017
                                 + 1.08999 w_k1_MSFT_s017
                                 + 0.93035 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.0346 w_k1_ADM_s017
                                 + 1.1003 w_k1_AXP_s017 + 1.14555 w_k1_DIS_s017
                                 + 1.13872 w_k1_FDX_s017
                                 + 1.06037 w_k1_IBM_s017
                                 + 1.13491 w_k1_INTC_s017
                                 + 0.953066 w_k1_KO_s017
                                 + 1.03492 w_k1_MAT_s017
                                 + 1.19704 w_k1_MSFT_s017
                                 + 1.11387 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.720786 w_k1_ADM_s017
                                 + 0.870107 w_k1_AXP_s017
                                 + 1.12981 w_k1_DIS_s017
                                 + 0.738068 w_k1_FDX_s017
                                 + 1.78685 w_k1_IBM_s017
                                 + 0.789536 w_k1_INTC_s017
                                 + 1.34806 w_k1_KO_s017 + 1.50016 w_k1_MAT_s017
                                 + 0.894214 w_k1_MSFT_s017
                                 + 0.949289 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.42519 w_k1_ADM_s017
                                 + 1.27633 w_k1_AXP_s017
                                 + 1.63833 w_k1_DIS_s017
                                 + 1.43624 w_k1_FDX_s017
                                 + 1.10907 w_k1_IBM_s017
                                 + 1.25909 w_k1_INTC_s017
                                 + 1.13561 w_k1_KO_s017 + 1.15976 w_k1_MAT_s017
                                 + 1.30834 w_k1_MSFT_s017
                                 + 1.22664 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.89777 w_k1_ADM_s017
                                 + 0.867329 w_k1_AXP_s017
                                 + 0.883735 w_k1_DIS_s017
                                 + 1.03077 w_k1_FDX_s017
                                 + 0.546012 w_k1_IBM_s017
                                 + 0.984418 w_k1_INTC_s017
                                 + 0.660783 w_k1_KO_s017
                                 + 0.577771 w_k1_MAT_s017
                                 + 0.990158 w_k1_MSFT_s017
                                 + 0.811104 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.80195 w_k1_ADM_s017
                                 + 0.891455 w_k1_AXP_s017
                                 + 0.883971 w_k1_DIS_s017
                                 + 1.02411 w_k1_FDX_s017
                                 + 0.783883 w_k1_IBM_s017
                                 + 1.0043 w_k1_INTC_s017
                                 + 0.764975 w_k1_KO_s017
                                 + 0.732648 w_k1_MAT_s017
                                 + 1.01092 w_k1_MSFT_s017
                                 + 0.888441 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.909368 w_k1_ADM_s100
                                 + 1.01343 w_k1_AXP_s100
                                 + 1.18358 w_k1_DIS_s100
                                 + 0.988443 w_k1_FDX_s100
                                 + 1.11725 w_k1_IBM_s100
                                 + 1.30111 w_k1_INTC_s100
                                 + 0.994056 w_k1_KO_s100
                                 + 1.24935 w_k1_MAT_s100
                                 + 1.1051 w_k1_MSFT_s100 + 1.01167 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.17788 w_k1_ADM_s100
                                 + 1.34731 w_k1_AXP_s100 + 1.6471 w_k1_DIS_s100
                                 + 1.31136 w_k1_FDX_s100
                                 + 1.27724 w_k1_IBM_s100
                                 + 1.77637 w_k1_INTC_s100 + 1.3348 w_k1_KO_s100
                                 + 0.983395 w_k1_MAT_s100
                                 + 1.40084 w_k1_MSFT_s100
                                 + 1.18912 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.647068 w_k1_ADM_s100
                                 + 0.806725 w_k1_AXP_s100
                                 + 0.919778 w_k1_DIS_s100
                                 + 0.866731 w_k1_FDX_s100
                                 + 1.04092 w_k1_IBM_s100
                                 + 1.13522 w_k1_INTC_s100
                                 + 1.00379 w_k1_KO_s100
                                 + 0.960807 w_k1_MAT_s100
                                 + 0.918075 w_k1_MSFT_s100
                                 + 0.985298 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.970198 w_k1_ADM_s100
                                 + 1.23918 w_k1_AXP_s100 + 1.1573 w_k1_DIS_s100
                                 + 0.939892 w_k1_FDX_s100
                                 + 1.64954 w_k1_IBM_s100
                                 + 1.01337 w_k1_INTC_s100
                                 + 1.19466 w_k1_KO_s100 + 1.81205 w_k1_MAT_s100
                                 + 0.944362 w_k1_MSFT_s100
                                 + 1.18403 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.771439 w_k1_ADM_s100
                                 + 0.753204 w_k1_AXP_s100
                                 + 0.747684 w_k1_DIS_s100
                                 + 0.912205 w_k1_FDX_s100
                                 + 0.687074 w_k1_IBM_s100
                                 + 1.0945 w_k1_INTC_s100
                                 + 0.732008 w_k1_KO_s100
                                 + 0.870122 w_k1_MAT_s100
                                 + 0.93124 w_k1_MSFT_s100
                                 + 0.902739 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.835646 w_k1_ADM_s100
                                 + 0.997479 w_k1_AXP_s100
                                 + 0.897438 w_k1_DIS_s100
                                 + 0.868218 w_k1_FDX_s100
                                 + 1.18922 w_k1_IBM_s100
                                 + 1.08324 w_k1_INTC_s100
                                 + 0.864334 w_k1_KO_s100
                                 + 1.20387 w_k1_MAT_s100
                                 + 0.902441 w_k1_MSFT_s100
                                 + 0.991107 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.08133 w_k1_ADM_s100
                                 + 1.21896 w_k1_AXP_s100
                                 + 1.16641 w_k1_DIS_s100
                                 + 1.16497 w_k1_FDX_s100
                                 + 0.808938 w_k1_IBM_s100
                                 + 1.29243 w_k1_INTC_s100
                                 + 0.954284 w_k1_KO_s100
                                 + 0.912202 w_k1_MAT_s100
                                 + 1.13212 w_k1_MSFT_s100
                                 + 1.05912 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.723793 w_k1_ADM_s100
                                 + 0.718241 w_k1_AXP_s100
                                 + 0.646115 w_k1_DIS_s100
                                 + 0.718824 w_k1_FDX_s100
                                 + 0.868776 w_k1_IBM_s100
                                 + 0.969774 w_k1_INTC_s100
                                 + 0.809656 w_k1_KO_s100
                                 + 0.97784 w_k1_MAT_s100
                                 + 0.721627 w_k1_MSFT_s100
                                 + 0.850935 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.18552 w_k1_ADM_s101
                                 + 1.45015 w_k1_AXP_s101
                                 + 1.50508 w_k1_DIS_s101
                                 + 1.35735 w_k1_FDX_s101
                                 + 0.78055 w_k1_IBM_s101
                                 + 1.12063 w_k1_INTC_s101
                                 + 1.29905 w_k1_KO_s101
                                 + 0.794754 w_k1_MAT_s101
                                 + 1.37902 w_k1_MSFT_s101
                                 + 1.14822 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.790812 w_k1_ADM_s101
                                 + 0.9075 w_k1_AXP_s101
                                 + 0.898347 w_k1_DIS_s101
                                 + 0.976247 w_k1_FDX_s101
                                 + 0.736201 w_k1_IBM_s101
                                 + 0.908352 w_k1_INTC_s101
                                 + 0.936038 w_k1_KO_s101
                                 + 0.873502 w_k1_MAT_s101
                                 + 1.04266 w_k1_MSFT_s101
                                 + 0.943339 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.580275 w_k1_ADM_s101
                                 + 0.707526 w_k1_AXP_s101
                                 + 0.589488 w_k1_DIS_s101
                                 + 0.697078 w_k1_FDX_s101
                                 + 0.768809 w_k1_IBM_s101
                                 + 0.785038 w_k1_INTC_s101
                                 + 0.868663 w_k1_KO_s101
                                 + 0.62557 w_k1_MAT_s101
                                 + 0.709564 w_k1_MSFT_s101
                                 + 0.854787 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.703968 w_k1_ADM_s101
                                 + 1.05326 w_k1_AXP_s101
                                 + 1.02542 w_k1_DIS_s101
                                 + 0.967099 w_k1_FDX_s101
                                 + 1.01187 w_k1_IBM_s101
                                 + 0.990518 w_k1_INTC_s101
                                 + 1.08128 w_k1_KO_s101
                                 + 0.981855 w_k1_MAT_s101
                                 + 1.00145 w_k1_MSFT_s101
                                 + 1.01319 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.04932 w_k1_ADM_s101
                                 + 1.10895 w_k1_AXP_s101
                                 + 1.40574 w_k1_DIS_s101
                                 + 1.18205 w_k1_FDX_s101
                                 + 0.913173 w_k1_IBM_s101
                                 + 1.07087 w_k1_INTC_s101 + 1.2186 w_k1_KO_s101
                                 + 0.837723 w_k1_MAT_s101
                                 + 1.1487 w_k1_MSFT_s101 + 1.11416 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.20708 w_k1_ADM_s101
                                 + 1.47956 w_k1_AXP_s101
                                 + 1.35137 w_k1_DIS_s101
                                 + 1.17052 w_k1_FDX_s101
                                 + 1.60497 w_k1_IBM_s101
                                 + 1.38593 w_k1_INTC_s101
                                 + 1.47053 w_k1_KO_s101 + 1.34687 w_k1_MAT_s101
                                 + 1.23983 w_k1_MSFT_s101
                                 + 1.32836 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.62294 w_k1_ADM_s101
                                 + 1.73704 w_k1_AXP_s101
                                 + 1.84444 w_k1_DIS_s101
                                 + 1.61345 w_k1_FDX_s101
                                 + 1.17693 w_k1_IBM_s101
                                 + 1.37384 w_k1_INTC_s101
                                 + 1.75149 w_k1_KO_s101 + 1.48808 w_k1_MAT_s101
                                 + 1.6381 w_k1_MSFT_s101 + 1.4019 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.75612 w_k1_ADM_s101
                                 + 0.801028 w_k1_AXP_s101
                                 + 0.750927 w_k1_DIS_s101
                                 + 0.719637 w_k1_FDX_s101
                                 + 1.40665 w_k1_IBM_s101
                                 + 0.678247 w_k1_INTC_s101
                                 + 1.13315 w_k1_KO_s101 + 1.1309 w_k1_MAT_s101
                                 + 0.801074 w_k1_MSFT_s101
                                 + 0.906327 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.64679 w_k1_ADM_s102
                                 + 1.47097 w_k1_AXP_s102
                                 + 1.91964 w_k1_DIS_s102
                                 + 1.35991 w_k1_FDX_s102 + 1.5218 w_k1_IBM_s102
                                 + 0.969623 w_k1_INTC_s102
                                 + 1.17424 w_k1_KO_s102 + 1.74616 w_k1_MAT_s102
                                 + 1.4186 w_k1_MSFT_s102 + 1.15723 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.23773 w_k1_ADM_s102
                                 + 1.22628 w_k1_AXP_s102
                                 + 1.25487 w_k1_DIS_s102
                                 + 1.21441 w_k1_FDX_s102
                                 + 0.913463 w_k1_IBM_s102
                                 + 0.928557 w_k1_INTC_s102
                                 + 0.862563 w_k1_KO_s102
                                 + 1.03098 w_k1_MAT_s102
                                 + 0.90557 w_k1_MSFT_s102
                                 + 1.02284 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.98098 w_k1_ADM_s102
                                 + 0.970548 w_k1_AXP_s102
                                 + 1.0154 w_k1_DIS_s102
                                 + 0.953474 w_k1_FDX_s102
                                 + 1.22653 w_k1_IBM_s102
                                 + 0.896332 w_k1_INTC_s102
                                 + 0.966187 w_k1_KO_s102
                                 + 1.15742 w_k1_MAT_s102
                                 + 0.8394 w_k1_MSFT_s102 + 1.01655 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.49026 w_k1_ADM_s102
                                 + 1.0724 w_k1_AXP_s102 + 1.32952 w_k1_DIS_s102
                                 + 1.17019 w_k1_FDX_s102
                                 + 1.00586 w_k1_IBM_s102
                                 + 1.12149 w_k1_INTC_s102
                                 + 1.10379 w_k1_KO_s102 + 0.9219 w_k1_MAT_s102
                                 + 1.2288 w_k1_MSFT_s102 + 1.08198 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.90769 w_k1_ADM_s102
                                 + 0.66338 w_k1_AXP_s102
                                 + 0.662452 w_k1_DIS_s102
                                 + 0.79913 w_k1_FDX_s102
                                 + 0.735272 w_k1_IBM_s102
                                 + 0.626804 w_k1_INTC_s102
                                 + 0.696807 w_k1_KO_s102
                                 + 0.752397 w_k1_MAT_s102
                                 + 0.763326 w_k1_MSFT_s102
                                 + 0.831487 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.769312 w_k1_ADM_s102
                                 + 0.826496 w_k1_AXP_s102
                                 + 0.872808 w_k1_DIS_s102
                                 + 0.843063 w_k1_FDX_s102
                                 + 0.969608 w_k1_IBM_s102
                                 + 0.928362 w_k1_INTC_s102
                                 + 0.845633 w_k1_KO_s102
                                 + 0.90357 w_k1_MAT_s102
                                 + 0.849586 w_k1_MSFT_s102
                                 + 0.989499 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.947708 w_k1_ADM_s102
                                 + 1.3433 w_k1_AXP_s102 + 1.24188 w_k1_DIS_s102
                                 + 1.07257 w_k1_FDX_s102 + 1.6499 w_k1_IBM_s102
                                 + 1.07753 w_k1_INTC_s102
                                 + 1.18075 w_k1_KO_s102 + 2.05817 w_k1_MAT_s102
                                 + 0.983581 w_k1_MSFT_s102
                                 + 1.13411 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.997595 w_k1_ADM_s102
                                 + 0.895078 w_k1_AXP_s102
                                 + 0.747839 w_k1_DIS_s102
                                 + 0.909574 w_k1_FDX_s102
                                 + 0.569814 w_k1_IBM_s102
                                 + 0.855988 w_k1_INTC_s102
                                 + 0.560388 w_k1_KO_s102
                                 + 0.515663 w_k1_MAT_s102
                                 + 0.793387 w_k1_MSFT_s102
                                 + 0.739413 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.867291 w_k1_ADM_s103
                                 + 0.909243 w_k1_AXP_s103
                                 + 0.813855 w_k1_DIS_s103
                                 + 0.920593 w_k1_FDX_s103
                                 + 0.786211 w_k1_IBM_s103
                                 + 0.847987 w_k1_INTC_s103
                                 + 0.808194 w_k1_KO_s103
                                 + 0.944116 w_k1_MAT_s103
                                 + 0.881813 w_k1_MSFT_s103
                                 + 0.920669 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.03382 w_k1_ADM_s103
                                 + 1.20565 w_k1_AXP_s103
                                 + 1.07367 w_k1_DIS_s103
                                 + 0.906717 w_k1_FDX_s103
                                 + 1.48979 w_k1_IBM_s103
                                 + 0.958074 w_k1_INTC_s103
                                 + 1.25367 w_k1_KO_s103 + 1.71408 w_k1_MAT_s103
                                 + 0.983178 w_k1_MSFT_s103
                                 + 1.07265 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.2739 w_k1_ADM_s103
                                 + 1.20069 w_k1_AXP_s103
                                 + 0.884667 w_k1_DIS_s103
                                 + 1.05103 w_k1_FDX_s103
                                 + 0.776587 w_k1_IBM_s103
                                 + 0.908377 w_k1_INTC_s103
                                 + 0.874781 w_k1_KO_s103
                                 + 0.648082 w_k1_MAT_s103
                                 + 0.89902 w_k1_MSFT_s103
                                 + 0.925776 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.03019 w_k1_ADM_s103
                                 + 1.23282 w_k1_AXP_s103 + 1.0133 w_k1_DIS_s103
                                 + 0.927481 w_k1_FDX_s103
                                 + 1.07945 w_k1_IBM_s103
                                 + 0.984244 w_k1_INTC_s103
                                 + 1.12364 w_k1_KO_s103 + 1.2608 w_k1_MAT_s103
                                 + 0.971986 w_k1_MSFT_s103
                                 + 1.04869 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.844089 w_k1_ADM_s103
                                 + 0.735307 w_k1_AXP_s103
                                 + 0.874887 w_k1_DIS_s103
                                 + 0.790588 w_k1_FDX_s103
                                 + 0.896957 w_k1_IBM_s103
                                 + 0.760018 w_k1_INTC_s103
                                 + 0.8843 w_k1_KO_s103 + 0.974864 w_k1_MAT_s103
                                 + 0.921586 w_k1_MSFT_s103
                                 + 0.827629 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.944446 w_k1_ADM_s103
                                 + 0.689291 w_k1_AXP_s103
                                 + 0.615003 w_k1_DIS_s103
                                 + 0.771827 w_k1_FDX_s103
                                 + 0.590266 w_k1_IBM_s103
                                 + 0.80183 w_k1_INTC_s103
                                 + 0.748678 w_k1_KO_s103
                                 + 0.609271 w_k1_MAT_s103
                                 + 0.81177 w_k1_MSFT_s103
                                 + 0.792208 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.25242 w_k1_ADM_s103
                                 + 0.940562 w_k1_AXP_s103
                                 + 1.12385 w_k1_DIS_s103 + 1.1498 w_k1_FDX_s103
                                 + 0.669565 w_k1_IBM_s103
                                 + 1.04608 w_k1_INTC_s103
                                 + 1.00306 w_k1_KO_s103
                                 + 0.835066 w_k1_MAT_s103
                                 + 1.18572 w_k1_MSFT_s103
                                 + 1.03787 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.10349 w_k1_ADM_s103
                                 + 1.53602 w_k1_AXP_s103 + 1.6158 w_k1_DIS_s103
                                 + 1.24048 w_k1_FDX_s103
                                 + 1.48966 w_k1_IBM_s103
                                 + 1.06213 w_k1_INTC_s103 + 1.4499 w_k1_KO_s103
                                 + 1.45861 w_k1_MAT_s103
                                 + 1.26614 w_k1_MSFT_s103
                                 + 1.23808 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.24227 w_k1_ADM_s104
                                 + 1.6841 w_k1_AXP_s104 + 1.78566 w_k1_DIS_s104
                                 + 1.53478 w_k1_FDX_s104
                                 + 1.33065 w_k1_IBM_s104
                                 + 1.51542 w_k1_INTC_s104
                                 + 1.53213 w_k1_KO_s104 + 1.28157 w_k1_MAT_s104
                                 + 1.47985 w_k1_MSFT_s104
                                 + 1.26365 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.16672 w_k1_ADM_s104
                                 + 1.02873 w_k1_AXP_s104
                                 + 1.15272 w_k1_DIS_s104
                                 + 1.03935 w_k1_FDX_s104
                                 + 0.963248 w_k1_IBM_s104
                                 + 1.01125 w_k1_INTC_s104
                                 + 1.02277 w_k1_KO_s104 + 0.83135 w_k1_MAT_s104
                                 + 1.10568 w_k1_MSFT_s104 + 1.1064 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.907946 w_k1_ADM_s104
                                 + 0.835751 w_k1_AXP_s104
                                 + 0.772537 w_k1_DIS_s104
                                 + 0.86414 w_k1_FDX_s104
                                 + 0.805816 w_k1_IBM_s104
                                 + 1.03966 w_k1_INTC_s104
                                 + 0.947264 w_k1_KO_s104
                                 + 0.764489 w_k1_MAT_s104
                                 + 1.02497 w_k1_MSFT_s104
                                 + 0.955397 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.19029 w_k1_ADM_s104
                                 + 1.46395 w_k1_AXP_s104
                                 + 1.10074 w_k1_DIS_s104
                                 + 1.23593 w_k1_FDX_s104
                                 + 1.05779 w_k1_IBM_s104
                                 + 1.46146 w_k1_INTC_s104
                                 + 1.24637 w_k1_KO_s104 + 1.17653 w_k1_MAT_s104
                                 + 1.17085 w_k1_MSFT_s104
                                 + 1.08384 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.866303 w_k1_ADM_s104
                                 + 0.858446 w_k1_AXP_s104
                                 + 0.71735 w_k1_DIS_s104
                                 + 0.781446 w_k1_FDX_s104
                                 + 1.18301 w_k1_IBM_s104
                                 + 0.804289 w_k1_INTC_s104
                                 + 1.05751 w_k1_KO_s104 + 1.05604 w_k1_MAT_s104
                                 + 0.773263 w_k1_MSFT_s104
                                 + 0.981682 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.01685 w_k1_ADM_s104
                                 + 1.04348 w_k1_AXP_s104
                                 + 1.30432 w_k1_DIS_s104
                                 + 1.35682 w_k1_FDX_s104
                                 + 0.661393 w_k1_IBM_s104
                                 + 1.06541 w_k1_INTC_s104
                                 + 0.883505 w_k1_KO_s104
                                 + 0.63698 w_k1_MAT_s104
                                 + 1.23819 w_k1_MSFT_s104
                                 + 1.06084 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.915585 w_k1_ADM_s104
                                 + 1.22536 w_k1_AXP_s104
                                 + 0.916779 w_k1_DIS_s104
                                 + 0.954966 w_k1_FDX_s104
                                 + 1.45462 w_k1_IBM_s104
                                 + 0.822774 w_k1_INTC_s104
                                 + 1.08501 w_k1_KO_s104 + 1.56929 w_k1_MAT_s104
                                 + 0.871476 w_k1_MSFT_s104
                                 + 1.0277 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.655045 w_k1_ADM_s104
                                 + 0.590924 w_k1_AXP_s104
                                 + 0.559569 w_k1_DIS_s104
                                 + 0.692929 w_k1_FDX_s104
                                 + 0.811428 w_k1_IBM_s104
                                 + 0.77918 w_k1_INTC_s104
                                 + 0.867822 w_k1_KO_s104
                                 + 0.804877 w_k1_MAT_s104
                                 + 0.752 w_k1_MSFT_s104 + 0.775766 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.820541 w_k1_ADM_s105
                                 + 0.882533 w_k1_AXP_s105
                                 + 1.02279 w_k1_DIS_s105
                                 + 1.05177 w_k1_FDX_s105
                                 + 0.685863 w_k1_IBM_s105
                                 + 1.12437 w_k1_INTC_s105
                                 + 0.886647 w_k1_KO_s105
                                 + 0.738459 w_k1_MAT_s105
                                 + 1.1771 w_k1_MSFT_s105
                                 + 0.844576 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.815731 w_k1_ADM_s105
                                 + 1.16855 w_k1_AXP_s105
                                 + 0.894723 w_k1_DIS_s105
                                 + 0.946599 w_k1_FDX_s105
                                 + 1.2138 w_k1_IBM_s105
                                 + 1.22287 w_k1_INTC_s105
                                 + 1.06751 w_k1_KO_s105 + 1.35157 w_k1_MAT_s105
                                 + 0.883335 w_k1_MSFT_s105
                                 + 0.888688 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.711614 w_k1_ADM_s105
                                 + 0.643504 w_k1_AXP_s105
                                 + 0.578519 w_k1_DIS_s105
                                 + 0.753492 w_k1_FDX_s105
                                 + 0.748341 w_k1_IBM_s105
                                 + 1.00246 w_k1_INTC_s105
                                 + 0.831218 w_k1_KO_s105
                                 + 0.781525 w_k1_MAT_s105
                                 + 0.832954 w_k1_MSFT_s105
                                 + 0.792652 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.2485 w_k1_ADM_s105
                                 + 0.994859 w_k1_AXP_s105
                                 + 1.2125 w_k1_DIS_s105 + 1.14656 w_k1_FDX_s105
                                 + 1.03563 w_k1_IBM_s105
                                 + 1.17403 w_k1_INTC_s105
                                 + 1.05943 w_k1_KO_s105 + 1.05345 w_k1_MAT_s105
                                 + 1.0934 w_k1_MSFT_s105 + 1.0797 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.17612 w_k1_ADM_s105
                                 + 1.56282 w_k1_AXP_s105
                                 + 1.50004 w_k1_DIS_s105
                                 + 1.20866 w_k1_FDX_s105
                                 + 1.50108 w_k1_IBM_s105
                                 + 1.34643 w_k1_INTC_s105
                                 + 1.20765 w_k1_KO_s105 + 2.09354 w_k1_MAT_s105
                                 + 1.0374 w_k1_MSFT_s105 + 1.17817 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.959115 w_k1_ADM_s105
                                 + 0.877018 w_k1_AXP_s105
                                 + 0.796268 w_k1_DIS_s105
                                 + 0.864669 w_k1_FDX_s105
                                 + 0.975967 w_k1_IBM_s105
                                 + 0.958229 w_k1_INTC_s105
                                 + 0.872669 w_k1_KO_s105
                                 + 0.795495 w_k1_MAT_s105
                                 + 0.794524 w_k1_MSFT_s105
                                 + 0.881158 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.29749 w_k1_ADM_s105
                                 + 1.51109 w_k1_AXP_s105
                                 + 1.27476 w_k1_DIS_s105
                                 + 1.44639 w_k1_FDX_s105
                                 + 0.958039 w_k1_IBM_s105
                                 + 1.66631 w_k1_INTC_s105
                                 + 1.08217 w_k1_KO_s105 + 1.0214 w_k1_MAT_s105
                                 + 1.29777 w_k1_MSFT_s105
                                 + 1.00866 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.806372 w_k1_ADM_s105
                                 + 0.947421 w_k1_AXP_s105
                                 + 0.80596 w_k1_DIS_s105
                                 + 0.890606 w_k1_FDX_s105
                                 + 0.745803 w_k1_IBM_s105
                                 + 1.23003 w_k1_INTC_s105
                                 + 0.77817 w_k1_KO_s105
                                 + 0.822802 w_k1_MAT_s105
                                 + 0.98312 w_k1_MSFT_s105
                                 + 0.829475 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.08137 w_k1_ADM_s106
                                 + 0.885832 w_k1_AXP_s106
                                 + 1.09379 w_k1_DIS_s106
                                 + 0.959373 w_k1_FDX_s106
                                 + 1.13219 w_k1_IBM_s106
                                 + 1.2362 w_k1_INTC_s106 + 1.1232 w_k1_KO_s106
                                 + 1.18702 w_k1_MAT_s106
                                 + 1.1302 w_k1_MSFT_s106 + 1.09851 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.00824 w_k1_ADM_s106
                                 + 0.771754 w_k1_AXP_s106
                                 + 0.906252 w_k1_DIS_s106
                                 + 0.936823 w_k1_FDX_s106
                                 + 0.785315 w_k1_IBM_s106
                                 + 1.04404 w_k1_INTC_s106
                                 + 0.959421 w_k1_KO_s106
                                 + 0.720766 w_k1_MAT_s106
                                 + 1.11414 w_k1_MSFT_s106
                                 + 0.905979 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.838224 w_k1_ADM_s106
                                 + 0.819208 w_k1_AXP_s106
                                 + 0.717911 w_k1_DIS_s106
                                 + 0.868562 w_k1_FDX_s106
                                 + 0.623046 w_k1_IBM_s106
                                 + 1.14232 w_k1_INTC_s106
                                 + 0.701724 w_k1_KO_s106
                                 + 0.541631 w_k1_MAT_s106
                                 + 0.899254 w_k1_MSFT_s106
                                 + 0.8167 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.22546 w_k1_ADM_s106
                                 + 1.28945 w_k1_AXP_s106
                                 + 0.939176 w_k1_DIS_s106
                                 + 0.943382 w_k1_FDX_s106
                                 + 0.988278 w_k1_IBM_s106
                                 + 1.02747 w_k1_INTC_s106
                                 + 0.978691 w_k1_KO_s106
                                 + 1.15763 w_k1_MAT_s106
                                 + 0.968025 w_k1_MSFT_s106
                                 + 1.0259 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.15188 w_k1_ADM_s106
                                 + 1.0697 w_k1_AXP_s106 + 1.36675 w_k1_DIS_s106
                                 + 1.27307 w_k1_FDX_s106
                                 + 0.821666 w_k1_IBM_s106
                                 + 1.36076 w_k1_INTC_s106
                                 + 1.10492 w_k1_KO_s106
                                 + 0.913047 w_k1_MAT_s106
                                 + 1.27888 w_k1_MSFT_s106
                                 + 1.10573 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.893928 w_k1_ADM_s106
                                 + 1.02426 w_k1_AXP_s106
                                 + 0.920309 w_k1_DIS_s106
                                 + 0.831101 w_k1_FDX_s106
                                 + 1.37541 w_k1_IBM_s106
                                 + 1.05874 w_k1_INTC_s106
                                 + 1.05349 w_k1_KO_s106 + 1.1619 w_k1_MAT_s106
                                 + 0.999087 w_k1_MSFT_s106
                                 + 0.944883 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.20239 w_k1_ADM_s106
                                 + 1.21337 w_k1_AXP_s106
                                 + 1.55736 w_k1_DIS_s106
                                 + 1.13173 w_k1_FDX_s106
                                 + 1.55801 w_k1_IBM_s106
                                 + 1.46073 w_k1_INTC_s106 + 1.4353 w_k1_KO_s106
                                 + 1.63816 w_k1_MAT_s106
                                 + 1.33139 w_k1_MSFT_s106
                                 + 1.17244 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.765441 w_k1_ADM_s106
                                 + 0.542813 w_k1_AXP_s106
                                 + 0.502994 w_k1_DIS_s106
                                 + 0.640087 w_k1_FDX_s106
                                 + 0.737259 w_k1_IBM_s106
                                 + 0.797306 w_k1_INTC_s106
                                 + 0.676066 w_k1_KO_s106
                                 + 0.616133 w_k1_MAT_s106
                                 + 0.746982 w_k1_MSFT_s106
                                 + 0.736818 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.940439 w_k1_ADM_s107
                                 + 1.05326 w_k1_AXP_s107
                                 + 1.07785 w_k1_DIS_s107
                                 + 1.17178 w_k1_FDX_s107
                                 + 0.726779 w_k1_IBM_s107
                                 + 1.0322 w_k1_INTC_s107
                                 + 0.838826 w_k1_KO_s107
                                 + 0.720847 w_k1_MAT_s107
                                 + 0.988694 w_k1_MSFT_s107
                                 + 1.02194 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.29286 w_k1_ADM_s107
                                 + 1.62125 w_k1_AXP_s107
                                 + 1.64305 w_k1_DIS_s107
                                 + 1.40981 w_k1_FDX_s107
                                 + 1.18716 w_k1_IBM_s107 + 1.751 w_k1_INTC_s107
                                 + 1.36901 w_k1_KO_s107 + 1.21598 w_k1_MAT_s107
                                 + 1.44861 w_k1_MSFT_s107
                                 + 1.28586 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.787492 w_k1_ADM_s107
                                 + 0.733 w_k1_AXP_s107 + 0.762598 w_k1_DIS_s107
                                 + 0.687498 w_k1_FDX_s107
                                 + 1.03252 w_k1_IBM_s107
                                 + 0.920006 w_k1_INTC_s107
                                 + 1.06901 w_k1_KO_s107
                                 + 0.941255 w_k1_MAT_s107
                                 + 0.839373 w_k1_MSFT_s107
                                 + 0.83118 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.826276 w_k1_ADM_s107
                                 + 0.937874 w_k1_AXP_s107
                                 + 0.814988 w_k1_DIS_s107
                                 + 0.846784 w_k1_FDX_s107
                                 + 0.933586 w_k1_IBM_s107
                                 + 1.03413 w_k1_INTC_s107
                                 + 0.949148 w_k1_KO_s107
                                 + 0.828921 w_k1_MAT_s107
                                 + 0.748305 w_k1_MSFT_s107
                                 + 0.925604 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.813269 w_k1_ADM_s107
                                 + 1.18844 w_k1_AXP_s107
                                 + 1.16824 w_k1_DIS_s107
                                 + 0.943912 w_k1_FDX_s107
                                 + 1.59826 w_k1_IBM_s107
                                 + 0.905643 w_k1_INTC_s107
                                 + 1.24064 w_k1_KO_s107 + 1.68689 w_k1_MAT_s107
                                 + 0.873213 w_k1_MSFT_s107
                                 + 1.02561 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.788698 w_k1_ADM_s107
                                 + 0.666683 w_k1_AXP_s107
                                 + 0.709615 w_k1_DIS_s107
                                 + 0.770576 w_k1_FDX_s107
                                 + 0.709748 w_k1_IBM_s107
                                 + 0.745881 w_k1_INTC_s107
                                 + 0.826836 w_k1_KO_s107
                                 + 0.561809 w_k1_MAT_s107
                                 + 0.774629 w_k1_MSFT_s107
                                 + 0.869718 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.839689 w_k1_ADM_s107
                                 + 0.90339 w_k1_AXP_s107
                                 + 0.780162 w_k1_DIS_s107
                                 + 0.92601 w_k1_FDX_s107
                                 + 0.646682 w_k1_IBM_s107
                                 + 0.916494 w_k1_INTC_s107
                                 + 0.797969 w_k1_KO_s107
                                 + 0.512328 w_k1_MAT_s107
                                 + 0.745556 w_k1_MSFT_s107
                                 + 0.97386 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.783312 w_k1_ADM_s107
                                 + 1.14809 w_k1_AXP_s107
                                 + 1.20907 w_k1_DIS_s107
                                 + 1.05625 w_k1_FDX_s107
                                 + 1.06292 w_k1_IBM_s107
                                 + 1.14954 w_k1_INTC_s107 + 1.1547 w_k1_KO_s107
                                 + 1.17852 w_k1_MAT_s107
                                 + 0.8913 w_k1_MSFT_s107 + 1.05596 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.25018 w_k1_ADM_s110
                                 + 0.902956 w_k1_AXP_s110
                                 + 1.06448 w_k1_DIS_s110 + 1.0468 w_k1_FDX_s110
                                 + 0.665638 w_k1_IBM_s110
                                 + 0.978928 w_k1_INTC_s110
                                 + 1.03786 w_k1_KO_s110
                                 + 0.506769 w_k1_MAT_s110
                                 + 1.04693 w_k1_MSFT_s110
                                 + 1.01252 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.29174 w_k1_ADM_s110
                                 + 1.39999 w_k1_AXP_s110
                                 + 1.68849 w_k1_DIS_s110
                                 + 1.28268 w_k1_FDX_s110
                                 + 1.31489 w_k1_IBM_s110
                                 + 1.21557 w_k1_INTC_s110
                                 + 1.46612 w_k1_KO_s110 + 1.37151 w_k1_MAT_s110
                                 + 1.22657 w_k1_MSFT_s110
                                 + 1.34682 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.814424 w_k1_ADM_s110
                                 + 0.973787 w_k1_AXP_s110
                                 + 0.657733 w_k1_DIS_s110
                                 + 0.853902 w_k1_FDX_s110
                                 + 0.690712 w_k1_IBM_s110
                                 + 0.77194 w_k1_INTC_s110
                                 + 0.811791 w_k1_KO_s110
                                 + 0.87565 w_k1_MAT_s110
                                 + 0.80492 w_k1_MSFT_s110
                                 + 0.877466 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.849513 w_k1_ADM_s110
                                 + 0.821171 w_k1_AXP_s110
                                 + 0.902875 w_k1_DIS_s110
                                 + 0.862675 w_k1_FDX_s110
                                 + 0.989039 w_k1_IBM_s110
                                 + 0.912559 w_k1_INTC_s110
                                 + 1.01867 w_k1_KO_s110
                                 + 0.732448 w_k1_MAT_s110
                                 + 0.849796 w_k1_MSFT_s110
                                 + 1.05138 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.845191 w_k1_ADM_s110
                                 + 1.04403 w_k1_AXP_s110
                                 + 0.907788 w_k1_DIS_s110
                                 + 0.861416 w_k1_FDX_s110
                                 + 1.42469 w_k1_IBM_s110
                                 + 0.702822 w_k1_INTC_s110
                                 + 1.16544 w_k1_KO_s110 + 1.4991 w_k1_MAT_s110
                                 + 0.853826 w_k1_MSFT_s110
                                 + 1.05374 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.729673 w_k1_ADM_s110
                                 + 0.565953 w_k1_AXP_s110
                                 + 0.571549 w_k1_DIS_s110
                                 + 0.615671 w_k1_FDX_s110
                                 + 0.84765 w_k1_IBM_s110
                                 + 0.595851 w_k1_INTC_s110
                                 + 0.795138 w_k1_KO_s110
                                 + 0.852458 w_k1_MAT_s110
                                 + 0.73601 w_k1_MSFT_s110
                                 + 0.791543 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.28965 w_k1_ADM_s110
                                 + 1.40091 w_k1_AXP_s110
                                 + 1.17453 w_k1_DIS_s110
                                 + 1.20599 w_k1_FDX_s110
                                 + 0.818459 w_k1_IBM_s110
                                 + 1.0111 w_k1_INTC_s110 + 1.04952 w_k1_KO_s110
                                 + 0.804355 w_k1_MAT_s110
                                 + 1.05632 w_k1_MSFT_s110
                                 + 1.12071 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.37336 w_k1_ADM_s110
                                 + 1.37729 w_k1_AXP_s110
                                 + 1.55744 w_k1_DIS_s110
                                 + 1.40243 w_k1_FDX_s110
                                 + 0.851712 w_k1_IBM_s110
                                 + 1.52371 w_k1_INTC_s110
                                 + 1.54005 w_k1_KO_s110 + 1.08695 w_k1_MAT_s110
                                 + 1.34013 w_k1_MSFT_s110
                                 + 1.14555 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.08651 w_k1_ADM_s111
                                 + 1.1119 w_k1_AXP_s111 + 1.19594 w_k1_DIS_s111
                                 + 1.05012 w_k1_FDX_s111
                                 + 1.03575 w_k1_IBM_s111
                                 + 1.12044 w_k1_INTC_s111
                                 + 1.14159 w_k1_KO_s111
                                 + 0.861347 w_k1_MAT_s111
                                 + 1.07864 w_k1_MSFT_s111
                                 + 1.02826 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.04723 w_k1_ADM_s111
                                 + 1.20248 w_k1_AXP_s111
                                 + 1.16003 w_k1_DIS_s111
                                 + 1.10772 w_k1_FDX_s111
                                 + 0.757056 w_k1_IBM_s111
                                 + 1.04482 w_k1_INTC_s111
                                 + 0.937699 w_k1_KO_s111
                                 + 0.826302 w_k1_MAT_s111
                                 + 1.19855 w_k1_MSFT_s111
                                 + 0.987171 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.984929 w_k1_ADM_s111
                                 + 1.26063 w_k1_AXP_s111
                                 + 1.18323 w_k1_DIS_s111
                                 + 0.992968 w_k1_FDX_s111
                                 + 1.80324 w_k1_IBM_s111
                                 + 1.13698 w_k1_INTC_s111
                                 + 1.34611 w_k1_KO_s111 + 1.70195 w_k1_MAT_s111
                                 + 1.1451 w_k1_MSFT_s111 + 1.26502 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.875156 w_k1_ADM_s111
                                 + 0.736907 w_k1_AXP_s111
                                 + 0.819702 w_k1_DIS_s111
                                 + 0.753551 w_k1_FDX_s111
                                 + 0.802285 w_k1_IBM_s111
                                 + 0.757163 w_k1_INTC_s111
                                 + 0.840156 w_k1_KO_s111
                                 + 0.76833 w_k1_MAT_s111
                                 + 0.935139 w_k1_MSFT_s111
                                 + 0.877776 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.974086 w_k1_ADM_s111
                                 + 1.22818 w_k1_AXP_s111
                                 + 1.32534 w_k1_DIS_s111
                                 + 1.14913 w_k1_FDX_s111
                                 + 1.23733 w_k1_IBM_s111
                                 + 1.17746 w_k1_INTC_s111
                                 + 1.18209 w_k1_KO_s111 + 1.09989 w_k1_MAT_s111
                                 + 1.13163 w_k1_MSFT_s111
                                 + 1.06819 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.00123 w_k1_ADM_s111
                                 + 1.39787 w_k1_AXP_s111
                                 + 1.64287 w_k1_DIS_s111
                                 + 1.19608 w_k1_FDX_s111 + 1.3489 w_k1_IBM_s111
                                 + 1.28137 w_k1_INTC_s111
                                 + 1.32454 w_k1_KO_s111 + 1.39637 w_k1_MAT_s111
                                 + 1.33423 w_k1_MSFT_s111
                                 + 1.07635 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.01047 w_k1_ADM_s111
                                 + 1.42799 w_k1_AXP_s111
                                 + 1.46981 w_k1_DIS_s111
                                 + 1.34751 w_k1_FDX_s111
                                 + 0.971673 w_k1_IBM_s111
                                 + 1.55134 w_k1_INTC_s111
                                 + 1.11867 w_k1_KO_s111 + 1.05723 w_k1_MAT_s111
                                 + 1.45933 w_k1_MSFT_s111 + 1.1471 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.605944 w_k1_ADM_s111
                                 + 0.807249 w_k1_AXP_s111
                                 + 0.822426 w_k1_DIS_s111
                                 + 0.739006 w_k1_FDX_s111
                                 + 1.19235 w_k1_IBM_s111
                                 + 0.932631 w_k1_INTC_s111
                                 + 1.03346 w_k1_KO_s111 + 1.0825 w_k1_MAT_s111
                                 + 0.986781 w_k1_MSFT_s111
                                 + 0.871371 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.14219 w_k1_ADM_s112
                                 + 1.00863 w_k1_AXP_s112
                                 + 1.18991 w_k1_DIS_s112
                                 + 1.18867 w_k1_FDX_s112
                                 + 0.948905 w_k1_IBM_s112
                                 + 1.10214 w_k1_INTC_s112
                                 + 1.16214 w_k1_KO_s112
                                 + 0.728144 w_k1_MAT_s112
                                 + 1.13733 w_k1_MSFT_s112
                                 + 1.14364 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.929799 w_k1_ADM_s112
                                 + 1.31456 w_k1_AXP_s112
                                 + 1.07521 w_k1_DIS_s112 + 1.0659 w_k1_FDX_s112
                                 + 1.24303 w_k1_IBM_s112
                                 + 1.09021 w_k1_INTC_s112
                                 + 0.994263 w_k1_KO_s112
                                 + 1.01926 w_k1_MAT_s112
                                 + 0.982016 w_k1_MSFT_s112
                                 + 1.0217 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.933888 w_k1_ADM_s112
                                 + 0.95314 w_k1_AXP_s112
                                 + 0.705428 w_k1_DIS_s112
                                 + 0.824162 w_k1_FDX_s112
                                 + 0.790289 w_k1_IBM_s112
                                 + 0.815723 w_k1_INTC_s112
                                 + 0.792299 w_k1_KO_s112
                                 + 0.791139 w_k1_MAT_s112
                                 + 0.847099 w_k1_MSFT_s112
                                 + 0.837185 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.29423 w_k1_ADM_s112
                                 + 1.11137 w_k1_AXP_s112
                                 + 0.806247 w_k1_DIS_s112
                                 + 1.03784 w_k1_FDX_s112
                                 + 0.904902 w_k1_IBM_s112
                                 + 1.04147 w_k1_INTC_s112
                                 + 0.88809 w_k1_KO_s112
                                 + 0.910106 w_k1_MAT_s112
                                 + 1.00826 w_k1_MSFT_s112
                                 + 1.00888 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.99012 w_k1_ADM_s112
                                 + 0.826898 w_k1_AXP_s112
                                 + 0.854308 w_k1_DIS_s112
                                 + 0.837657 w_k1_FDX_s112
                                 + 1.16275 w_k1_IBM_s112
                                 + 0.906119 w_k1_INTC_s112
                                 + 1.08359 w_k1_KO_s112 + 1.27694 w_k1_MAT_s112
                                 + 0.904798 w_k1_MSFT_s112
                                 + 0.984448 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.42113 w_k1_ADM_s112
                                 + 1.59821 w_k1_AXP_s112
                                 + 1.33424 w_k1_DIS_s112
                                 + 1.29282 w_k1_FDX_s112
                                 + 1.42173 w_k1_IBM_s112
                                 + 1.32397 w_k1_INTC_s112
                                 + 1.22867 w_k1_KO_s112 + 1.30117 w_k1_MAT_s112
                                 + 1.21379 w_k1_MSFT_s112 + 1.1285 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.864285 w_k1_ADM_s112
                                 + 0.68205 w_k1_AXP_s112
                                 + 0.642983 w_k1_DIS_s112
                                 + 0.752893 w_k1_FDX_s112
                                 + 0.761372 w_k1_IBM_s112
                                 + 0.826303 w_k1_INTC_s112
                                 + 0.792849 w_k1_KO_s112
                                 + 0.641194 w_k1_MAT_s112
                                 + 0.742194 w_k1_MSFT_s112
                                 + 0.874784 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.962716 w_k1_ADM_s112
                                 + 0.873275 w_k1_AXP_s112
                                 + 0.927262 w_k1_DIS_s112
                                 + 0.936976 w_k1_FDX_s112
                                 + 0.742841 w_k1_IBM_s112
                                 + 0.852669 w_k1_INTC_s112
                                 + 0.760487 w_k1_KO_s112
                                 + 0.650272 w_k1_MAT_s112
                                 + 0.930027 w_k1_MSFT_s112
                                 + 0.969051 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.00026 w_k1_ADM_s113
                                 + 1.55516 w_k1_AXP_s113
                                 + 1.41259 w_k1_DIS_s113
                                 + 1.35669 w_k1_FDX_s113
                                 + 1.06512 w_k1_IBM_s113
                                 + 1.0636 w_k1_INTC_s113 + 1.35593 w_k1_KO_s113
                                 + 1.63653 w_k1_MAT_s113
                                 + 1.2479 w_k1_MSFT_s113 + 1.24434 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.11055 w_k1_ADM_s113
                                 + 0.935681 w_k1_AXP_s113
                                 + 1.50063 w_k1_DIS_s113
                                 + 1.22697 w_k1_FDX_s113
                                 + 1.11251 w_k1_IBM_s113
                                 + 1.09604 w_k1_INTC_s113
                                 + 1.08938 w_k1_KO_s113 + 1.10891 w_k1_MAT_s113
                                 + 1.20344 w_k1_MSFT_s113
                                 + 1.10562 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.964171 w_k1_ADM_s113
                                 + 1.06199 w_k1_AXP_s113
                                 + 1.06142 w_k1_DIS_s113
                                 + 1.07675 w_k1_FDX_s113
                                 + 1.39067 w_k1_IBM_s113
                                 + 1.03015 w_k1_INTC_s113
                                 + 1.00513 w_k1_KO_s113 + 1.12972 w_k1_MAT_s113
                                 + 1.02765 w_k1_MSFT_s113
                                 + 1.08974 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.784182 w_k1_ADM_s113
                                 + 0.895619 w_k1_AXP_s113
                                 + 0.881671 w_k1_DIS_s113
                                 + 0.999641 w_k1_FDX_s113
                                 + 1.10854 w_k1_IBM_s113
                                 + 0.911932 w_k1_INTC_s113
                                 + 0.905803 w_k1_KO_s113
                                 + 0.91498 w_k1_MAT_s113
                                 + 0.88399 w_k1_MSFT_s113
                                 + 0.921932 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.696029 w_k1_ADM_s113
                                 + 0.694914 w_k1_AXP_s113
                                 + 0.707224 w_k1_DIS_s113
                                 + 0.809923 w_k1_FDX_s113
                                 + 0.819298 w_k1_IBM_s113
                                 + 0.676578 w_k1_INTC_s113
                                 + 0.717304 w_k1_KO_s113
                                 + 0.924627 w_k1_MAT_s113
                                 + 0.779569 w_k1_MSFT_s113
                                 + 0.848576 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.776259 w_k1_ADM_s113
                                 + 1.1708 w_k1_AXP_s113 + 1.17213 w_k1_DIS_s113
                                 + 0.962875 w_k1_FDX_s113
                                 + 1.84902 w_k1_IBM_s113
                                 + 0.996089 w_k1_INTC_s113
                                 + 1.20504 w_k1_KO_s113 + 1.37264 w_k1_MAT_s113
                                 + 1.02134 w_k1_MSFT_s113
                                 + 0.983626 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.04425 w_k1_ADM_s113
                                 + 1.11974 w_k1_AXP_s113
                                 + 0.888518 w_k1_DIS_s113
                                 + 1.11913 w_k1_FDX_s113
                                 + 0.783216 w_k1_IBM_s113
                                 + 0.989271 w_k1_INTC_s113
                                 + 0.760304 w_k1_KO_s113
                                 + 0.591333 w_k1_MAT_s113
                                 + 0.900199 w_k1_MSFT_s113
                                 + 0.917517 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.827432 w_k1_ADM_s113
                                 + 1.07332 w_k1_AXP_s113
                                 + 1.27644 w_k1_DIS_s113
                                 + 1.21533 w_k1_FDX_s113
                                 + 0.781948 w_k1_IBM_s113
                                 + 1.00733 w_k1_INTC_s113
                                 + 1.00758 w_k1_KO_s113
                                 + 0.899651 w_k1_MAT_s113
                                 + 1.19073 w_k1_MSFT_s113
                                 + 1.11159 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.874708 w_k1_ADM_s114
                                 + 1.26182 w_k1_AXP_s114
                                 + 1.43574 w_k1_DIS_s114
                                 + 1.17731 w_k1_FDX_s114
                                 + 1.44723 w_k1_IBM_s114
                                 + 1.22747 w_k1_INTC_s114
                                 + 1.17792 w_k1_KO_s114 + 1.54498 w_k1_MAT_s114
                                 + 1.14453 w_k1_MSFT_s114
                                 + 1.03335 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.918235 w_k1_ADM_s114
                                 + 1.02531 w_k1_AXP_s114
                                 + 0.892354 w_k1_DIS_s114
                                 + 1.03325 w_k1_FDX_s114
                                 + 0.821328 w_k1_IBM_s114
                                 + 0.878187 w_k1_INTC_s114
                                 + 0.824013 w_k1_KO_s114
                                 + 0.845832 w_k1_MAT_s114
                                 + 0.913268 w_k1_MSFT_s114
                                 + 0.87287 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.2032 w_k1_ADM_s114
                                 + 1.17047 w_k1_AXP_s114
                                 + 1.09651 w_k1_DIS_s114
                                 + 1.20692 w_k1_FDX_s114
                                 + 0.990565 w_k1_IBM_s114
                                 + 1.20619 w_k1_INTC_s114
                                 + 1.03103 w_k1_KO_s114
                                 + 0.819498 w_k1_MAT_s114
                                 + 1.15177 w_k1_MSFT_s114
                                 + 1.08287 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.703427 w_k1_ADM_s114
                                 + 0.758584 w_k1_AXP_s114
                                 + 1.02431 w_k1_DIS_s114
                                 + 1.01151 w_k1_FDX_s114
                                 + 0.822754 w_k1_IBM_s114
                                 + 1.03705 w_k1_INTC_s114
                                 + 0.889881 w_k1_KO_s114
                                 + 0.687307 w_k1_MAT_s114
                                 + 1.2693 w_k1_MSFT_s114
                                 + 0.905798 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.973701 w_k1_ADM_s114
                                 + 2.00667 w_k1_AXP_s114
                                 + 1.94884 w_k1_DIS_s114
                                 + 1.53771 w_k1_FDX_s114
                                 + 1.48014 w_k1_IBM_s114
                                 + 1.5565 w_k1_INTC_s114 + 1.34917 w_k1_KO_s114
                                 + 2.15829 w_k1_MAT_s114
                                 + 1.68993 w_k1_MSFT_s114
                                 + 1.12049 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.865237 w_k1_ADM_s114
                                 + 0.848507 w_k1_AXP_s114
                                 + 0.941361 w_k1_DIS_s114
                                 + 0.85083 w_k1_FDX_s114
                                 + 1.25719 w_k1_IBM_s114
                                 + 1.15247 w_k1_INTC_s114
                                 + 1.02082 w_k1_KO_s114
                                 + 0.780147 w_k1_MAT_s114
                                 + 1.06352 w_k1_MSFT_s114 + 1.0101 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.818679 w_k1_ADM_s114
                                 + 0.864169 w_k1_AXP_s114
                                 + 0.86924 w_k1_DIS_s114
                                 + 0.810054 w_k1_FDX_s114
                                 + 1.01306 w_k1_IBM_s114
                                 + 0.785786 w_k1_INTC_s114
                                 + 1.08048 w_k1_KO_s114 + 1.41141 w_k1_MAT_s114
                                 + 0.945565 w_k1_MSFT_s114
                                 + 0.896187 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.724115 w_k1_ADM_s114
                                 + 0.653858 w_k1_AXP_s114
                                 + 0.70142 w_k1_DIS_s114
                                 + 0.796997 w_k1_FDX_s114
                                 + 0.694151 w_k1_IBM_s114
                                 + 0.799906 w_k1_INTC_s114
                                 + 0.732637 w_k1_KO_s114
                                 + 0.672873 w_k1_MAT_s114
                                 + 0.924955 w_k1_MSFT_s114
                                 + 0.79152 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.25152 w_k1_ADM_s115
                                 + 1.39083 w_k1_AXP_s115
                                 + 1.29385 w_k1_DIS_s115
                                 + 1.15611 w_k1_FDX_s115 + 1.4391 w_k1_IBM_s115
                                 + 1.05863 w_k1_INTC_s115
                                 + 1.26753 w_k1_KO_s115 + 1.38929 w_k1_MAT_s115
                                 + 1.25917 w_k1_MSFT_s115
                                 + 1.07833 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.15988 w_k1_ADM_s115
                                 + 0.959538 w_k1_AXP_s115
                                 + 1.04283 w_k1_DIS_s115
                                 + 1.00529 w_k1_FDX_s115
                                 + 0.75804 w_k1_IBM_s115
                                 + 0.782316 w_k1_INTC_s115
                                 + 1.03484 w_k1_KO_s115
                                 + 0.780988 w_k1_MAT_s115
                                 + 1.00086 w_k1_MSFT_s115
                                 + 1.01179 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.39883 w_k1_ADM_s115
                                 + 1.19995 w_k1_AXP_s115
                                 + 0.959821 w_k1_DIS_s115
                                 + 1.14976 w_k1_FDX_s115
                                 + 0.991339 w_k1_IBM_s115
                                 + 0.957996 w_k1_INTC_s115
                                 + 0.940024 w_k1_KO_s115
                                 + 1.02221 w_k1_MAT_s115
                                 + 0.949194 w_k1_MSFT_s115
                                 + 1.10318 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.883954 w_k1_ADM_s115
                                 + 0.8017 w_k1_AXP_s115
                                 + 0.944619 w_k1_DIS_s115
                                 + 0.835278 w_k1_FDX_s115
                                 + 1.24026 w_k1_IBM_s115
                                 + 0.814581 w_k1_INTC_s115
                                 + 1.09826 w_k1_KO_s115 + 1.25372 w_k1_MAT_s115
                                 + 0.945137 w_k1_MSFT_s115
                                 + 0.998362 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.0581 w_k1_ADM_s115
                                 + 0.806219 w_k1_AXP_s115
                                 + 0.760087 w_k1_DIS_s115
                                 + 0.85267 w_k1_FDX_s115
                                 + 0.978138 w_k1_IBM_s115
                                 + 0.926782 w_k1_INTC_s115
                                 + 0.945797 w_k1_KO_s115
                                 + 0.975703 w_k1_MAT_s115
                                 + 0.856786 w_k1_MSFT_s115
                                 + 1.01401 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.28495 w_k1_ADM_s115
                                 + 1.23897 w_k1_AXP_s115
                                 + 1.43826 w_k1_DIS_s115
                                 + 1.30601 w_k1_FDX_s115
                                 + 0.985561 w_k1_IBM_s115
                                 + 1.24656 w_k1_INTC_s115 + 1.1304 w_k1_KO_s115
                                 + 0.990241 w_k1_MAT_s115
                                 + 1.31249 w_k1_MSFT_s115 + 1.1399 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.21913 w_k1_ADM_s115
                                 + 1.1388 w_k1_AXP_s115 + 1.10218 w_k1_DIS_s115
                                 + 1.02103 w_k1_FDX_s115 + 1.0809 w_k1_IBM_s115
                                 + 1.00279 w_k1_INTC_s115
                                 + 1.06883 w_k1_KO_s115
                                 + 0.828346 w_k1_MAT_s115
                                 + 0.995216 w_k1_MSFT_s115
                                 + 1.05698 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.15135 w_k1_ADM_s115
                                 + 0.697199 w_k1_AXP_s115
                                 + 0.665381 w_k1_DIS_s115
                                 + 0.835168 w_k1_FDX_s115
                                 + 0.695632 w_k1_IBM_s115
                                 + 0.805397 w_k1_INTC_s115
                                 + 0.706295 w_k1_KO_s115
                                 + 0.565251 w_k1_MAT_s115
                                 + 0.749949 w_k1_MSFT_s115
                                 + 0.851318 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.947993 w_k1_ADM_s116
                                 + 1.10703 w_k1_AXP_s116
                                 + 0.930372 w_k1_DIS_s116
                                 + 0.906434 w_k1_FDX_s116
                                 + 1.19455 w_k1_IBM_s116
                                 + 1.00058 w_k1_INTC_s116
                                 + 1.18854 w_k1_KO_s116 + 1.30273 w_k1_MAT_s116
                                 + 0.86422 w_k1_MSFT_s116
                                 + 0.929196 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.32554 w_k1_ADM_s116
                                 + 1.54392 w_k1_AXP_s116
                                 + 1.44358 w_k1_DIS_s116 + 1.3805 w_k1_FDX_s116
                                 + 0.951171 w_k1_IBM_s116
                                 + 1.14316 w_k1_INTC_s116
                                 + 1.14271 w_k1_KO_s116
                                 + 0.894491 w_k1_MAT_s116
                                 + 1.09654 w_k1_MSFT_s116
                                 + 1.07442 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.45059 w_k1_ADM_s116
                                 + 1.24584 w_k1_AXP_s116
                                 + 1.35166 w_k1_DIS_s116
                                 + 1.20873 w_k1_FDX_s116
                                 + 1.16282 w_k1_IBM_s116
                                 + 1.22357 w_k1_INTC_s116
                                 + 1.30313 w_k1_KO_s116 + 1.34651 w_k1_MAT_s116
                                 + 0.964487 w_k1_MSFT_s116
                                 + 1.21159 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00428 w_k1_ADM_s116
                                 + 0.982771 w_k1_AXP_s116
                                 + 0.874871 w_k1_DIS_s116
                                 + 1.03499 w_k1_FDX_s116
                                 + 0.739325 w_k1_IBM_s116
                                 + 1.103 w_k1_INTC_s116 + 0.873306 w_k1_KO_s116
                                 + 0.705883 w_k1_MAT_s116
                                 + 0.870562 w_k1_MSFT_s116
                                 + 0.937848 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.05019 w_k1_ADM_s116
                                 + 1.44955 w_k1_AXP_s116
                                 + 1.31453 w_k1_DIS_s116
                                 + 1.12011 w_k1_FDX_s116
                                 + 1.73614 w_k1_IBM_s116
                                 + 1.2915 w_k1_INTC_s116 + 1.41461 w_k1_KO_s116
                                 + 1.22503 w_k1_MAT_s116
                                 + 1.03425 w_k1_MSFT_s116
                                 + 1.03716 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.981911 w_k1_ADM_s116
                                 + 0.980253 w_k1_AXP_s116
                                 + 1.06184 w_k1_DIS_s116
                                 + 1.04581 w_k1_FDX_s116
                                 + 1.10404 w_k1_IBM_s116
                                 + 1.14061 w_k1_INTC_s116
                                 + 0.972366 w_k1_KO_s116
                                 + 0.701894 w_k1_MAT_s116
                                 + 0.915488 w_k1_MSFT_s116
                                 + 0.956109 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.848394 w_k1_ADM_s116
                                 + 0.79274 w_k1_AXP_s116
                                 + 0.670384 w_k1_DIS_s116
                                 + 0.802779 w_k1_FDX_s116
                                 + 0.740088 w_k1_IBM_s116
                                 + 0.840744 w_k1_INTC_s116
                                 + 0.729978 w_k1_KO_s116
                                 + 0.758918 w_k1_MAT_s116
                                 + 0.758166 w_k1_MSFT_s116
                                 + 0.843689 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.02039 w_k1_ADM_s116
                                 + 1.23457 w_k1_AXP_s116
                                 + 1.05997 w_k1_DIS_s116
                                 + 1.10775 w_k1_FDX_s116
                                 + 0.841623 w_k1_IBM_s116
                                 + 1.08572 w_k1_INTC_s116
                                 + 0.956619 w_k1_KO_s116
                                 + 0.763619 w_k1_MAT_s116
                                 + 1.01829 w_k1_MSFT_s116
                                 + 1.01796 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10107 w_k1_ADM_s117
                                 + 0.937627 w_k1_AXP_s117
                                 + 1.10301 w_k1_DIS_s117
                                 + 1.04124 w_k1_FDX_s117
                                 + 0.862803 w_k1_IBM_s117
                                 + 1.24719 w_k1_INTC_s117
                                 + 0.985901 w_k1_KO_s117
                                 + 0.890383 w_k1_MAT_s117
                                 + 1.22024 w_k1_MSFT_s117
                                 + 1.10039 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.959735 w_k1_ADM_s117
                                 + 0.830851 w_k1_AXP_s117
                                 + 1.08509 w_k1_DIS_s117
                                 + 0.855267 w_k1_FDX_s117
                                 + 1.39969 w_k1_IBM_s117
                                 + 0.943371 w_k1_INTC_s117
                                 + 1.26376 w_k1_KO_s117 + 1.12592 w_k1_MAT_s117
                                 + 0.902476 w_k1_MSFT_s117
                                 + 1.01246 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.812509 w_k1_ADM_s117
                                 + 0.903837 w_k1_AXP_s117
                                 + 0.775048 w_k1_DIS_s117
                                 + 0.773137 w_k1_FDX_s117
                                 + 0.969998 w_k1_IBM_s117
                                 + 0.869625 w_k1_INTC_s117
                                 + 0.938526 w_k1_KO_s117
                                 + 1.01875 w_k1_MAT_s117
                                 + 0.813911 w_k1_MSFT_s117
                                 + 0.994853 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15222 w_k1_ADM_s117
                                 + 1.34137 w_k1_AXP_s117
                                 + 0.977983 w_k1_DIS_s117
                                 + 1.00429 w_k1_FDX_s117
                                 + 1.21709 w_k1_IBM_s117
                                 + 1.00197 w_k1_INTC_s117 + 1.055 w_k1_KO_s117
                                 + 1.28626 w_k1_MAT_s117
                                 + 0.929159 w_k1_MSFT_s117
                                 + 1.12727 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.25059 w_k1_ADM_s117
                                 + 1.57022 w_k1_AXP_s117 + 1.5926 w_k1_DIS_s117
                                 + 1.27882 w_k1_FDX_s117
                                 + 1.41022 w_k1_IBM_s117
                                 + 1.1935 w_k1_INTC_s117 + 1.29975 w_k1_KO_s117
                                 + 1.32105 w_k1_MAT_s117
                                 + 1.28802 w_k1_MSFT_s117
                                 + 1.27493 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.949646 w_k1_ADM_s117
                                 + 0.702896 w_k1_AXP_s117
                                 + 0.890862 w_k1_DIS_s117
                                 + 0.95908 w_k1_FDX_s117
                                 + 0.598119 w_k1_IBM_s117
                                 + 1.00237 w_k1_INTC_s117
                                 + 0.761342 w_k1_KO_s117
                                 + 0.674359 w_k1_MAT_s117
                                 + 1.13514 w_k1_MSFT_s117
                                 + 0.986511 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.770741 w_k1_ADM_s117
                                 + 0.714888 w_k1_AXP_s117
                                 + 0.605565 w_k1_DIS_s117
                                 + 0.70892 w_k1_FDX_s117
                                 + 0.674735 w_k1_IBM_s117
                                 + 0.699521 w_k1_INTC_s117
                                 + 0.779229 w_k1_KO_s117 + 1.0106 w_k1_MAT_s117
                                 + 0.819473 w_k1_MSFT_s117
                                 + 0.893612 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.649065 w_k1_ADM_s117
                                 + 0.676057 w_k1_AXP_s117
                                 + 0.829742 w_k1_DIS_s117
                                 + 0.73971 w_k1_FDX_s117
                                 + 0.958275 w_k1_IBM_s117
                                 + 1.04238 w_k1_INTC_s117
                                 + 0.910842 w_k1_KO_s117
                                 + 0.819126 w_k1_MAT_s117
                                 + 0.895449 w_k1_MSFT_s117
                                 + 0.987783 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0570388268826358 w_k1_ADM_s000
                                 + 0.0153207045274671 w_k1_ADM_s001
                                 + 0.0376219294702525 w_k1_ADM_s002
                                 + 0.0274843160916869 w_k1_ADM_s003
                                 + 0.0263418514170453 w_k1_ADM_s004
                                 + 0.0251625308022615 w_k1_ADM_s005
                                 + 0.0627928788673727 w_k1_ADM_s006
                                 + 0.0509579748793828 w_k1_ADM_s007
                                 + 0.0272862317855773 w_k1_ADM_s010
                                 + 0.0127051109718529 w_k1_ADM_s011
                                 + 0.0150385884611734 w_k1_ADM_s012
                                 + 0.039153656840065 w_k1_ADM_s013
                                 + 0.0103616185472058 w_k1_ADM_s014
                                 + 0.00852924607508555 w_k1_ADM_s015
                                 + 0.0165909481954167 w_k1_ADM_s016
                                 + 0.0082531724351559 w_k1_ADM_s017
                                 + 0.0518075991713643 w_k1_ADM_s100
                                 + 0.0626157383355521 w_k1_ADM_s101
                                 + 0.0182069909970483 w_k1_ADM_s102
                                 + 0.0791202619788962 w_k1_ADM_s103
                                 + 0.062291647701832 w_k1_ADM_s104
                                 + 0.0257131343069359 w_k1_ADM_s105
                                 + 0.0655289123152873 w_k1_ADM_s106
                                 + 0.0245760749485167 w_k1_ADM_s107
                                 + 0.0436673881542693 w_k1_ADM_s110
                                 + 0.0211359339550723 w_k1_ADM_s111
                                 + 0.0251462868770626 w_k1_ADM_s112
                                 + 0.0162682471506746 w_k1_ADM_s113
                                 + 0.0132137619072277 w_k1_ADM_s114
                                 + 0.018096359074113 w_k1_ADM_s115
                                 + 0.0221988630892031 w_k1_ADM_s116
                                 + 0.015272896261162 w_k1_ADM_s117
                                 + 0.0584186788483244 w_k1_AXP_s000
                                 + 0.0128220101737066 w_k1_AXP_s001
                                 + 0.0412987447577897 w_k1_AXP_s002
                                 + 0.0270777222166312 w_k1_AXP_s003
                                 + 0.0307821675217695 w_k1_AXP_s004
                                 + 0.025547129038987 w_k1_AXP_s005
                                 + 0.0631711199921189 w_k1_AXP_s006
                                 + 0.0504859629821073 w_k1_AXP_s007
                                 + 0.0275667288535381 w_k1_AXP_s010
                                 + 0.0124583611992668 w_k1_AXP_s011
                                 + 0.0137504796440834 w_k1_AXP_s012
                                 + 0.0368608334483151 w_k1_AXP_s013
                                 + 0.00963161103439321 w_k1_AXP_s014
                                 + 0.00801754119469474 w_k1_AXP_s015
                                 + 0.0139447593760038 w_k1_AXP_s016
                                 + 0.00771349753324144 w_k1_AXP_s017
                                 + 0.057809369030952 w_k1_AXP_s100
                                 + 0.0762710637759575 w_k1_AXP_s101
                                 + 0.017449079855329 w_k1_AXP_s102
                                 + 0.0849029181433592 w_k1_AXP_s103
                                 + 0.0664127725198715 w_k1_AXP_s104
                                 + 0.0282329484317892 w_k1_AXP_s105
                                 + 0.0598339815151977 w_k1_AXP_s106
                                 + 0.029798195703432 w_k1_AXP_s107
                                 + 0.0440462947831967 w_k1_AXP_s110
                                 + 0.0237488761779895 w_k1_AXP_s111
                                 + 0.0242200948356458 w_k1_AXP_s112
                                 + 0.0183295346611367 w_k1_AXP_s113
                                 + 0.0149697230625995 w_k1_AXP_s114
                                 + 0.0159259829466279 w_k1_AXP_s115
                                 + 0.0238842712073079 w_k1_AXP_s116
                                 + 0.0152004975492355 w_k1_AXP_s117
                                 + 0.0615850750164429 w_k1_DIS_s000
                                 + 0.014592123527328 w_k1_DIS_s001
                                 + 0.0428826596248761 w_k1_DIS_s002
                                 + 0.0306596785370758 w_k1_DIS_s003
                                 + 0.0276237911753082 w_k1_DIS_s004
                                 + 0.0266126801898658 w_k1_DIS_s005
                                 + 0.0672886311685946 w_k1_DIS_s006
                                 + 0.0480604757229333 w_k1_DIS_s007
                                 + 0.0265321325884867 w_k1_DIS_s010
                                 + 0.0129779025194041 w_k1_DIS_s011
                                 + 0.0142998876618741 w_k1_DIS_s012
                                 + 0.0340869944851725 w_k1_DIS_s013
                                 + 0.00927845471921197 w_k1_DIS_s014
                                 + 0.00802057356588864 w_k1_DIS_s015
                                 + 0.0151785976011295 w_k1_DIS_s016
                                 + 0.00879326931543213 w_k1_DIS_s017
                                 + 0.0593680538219145 w_k1_DIS_s100
                                 + 0.0752019485685424 w_k1_DIS_s101
                                 + 0.0182363788443368 w_k1_DIS_s102
                                 + 0.0794645446167144 w_k1_DIS_s103
                                 + 0.0634053111766228 w_k1_DIS_s104
                                 + 0.0262225193186495 w_k1_DIS_s105
                                 + 0.0665487846516546 w_k1_DIS_s106
                                 + 0.0295079284600748 w_k1_DIS_s107
                                 + 0.044890218790951 w_k1_DIS_s110
                                 + 0.0250218765205153 w_k1_DIS_s111
                                 + 0.0223437853120126 w_k1_DIS_s112
                                 + 0.0187714782608676 w_k1_DIS_s113
                                 + 0.0158987906702274 w_k1_DIS_s114
                                 + 0.0157741273587861 w_k1_DIS_s115
                                 + 0.022129443985882 w_k1_DIS_s116
                                 + 0.0156188816192847 w_k1_DIS_s117
                                 + 0.0581418589195973 w_k1_FDX_s000
                                 + 0.0133505979661434 w_k1_FDX_s001
                                 + 0.0414537577634387 w_k1_FDX_s002
                                 + 0.0272664497948397 w_k1_FDX_s003
                                 + 0.0278684138358739 w_k1_FDX_s004
                                 + 0.0258530213832231 w_k1_FDX_s005
                                 + 0.0631661574501462 w_k1_FDX_s006
                                 + 0.0464394381825653 w_k1_FDX_s007
                                 + 0.0263793283862438 w_k1_FDX_s010
                                 + 0.0123761334462706 w_k1_FDX_s011
                                 + 0.0149821233972417 w_k1_FDX_s012
                                 + 0.035415368580463 w_k1_FDX_s013
                                 + 0.00968465203456099 w_k1_FDX_s014
                                 + 0.00796630007103649 w_k1_FDX_s015
                                 + 0.0142767347435145 w_k1_FDX_s016
                                 + 0.0084444898783462 w_k1_FDX_s017
                                 + 0.0571107952010625 w_k1_FDX_s100
                                 + 0.0716452940544815 w_k1_FDX_s101
                                 + 0.0173102652700464 w_k1_FDX_s102
                                 + 0.0740133024678472 w_k1_FDX_s103
                                 + 0.0640818582022927 w_k1_FDX_s104
                                 + 0.0266069941004745 w_k1_FDX_s105
                                 + 0.0617540213339524 w_k1_FDX_s106
                                 + 0.028401562748698 w_k1_FDX_s107
                                 + 0.0422842311638133 w_k1_FDX_s110
                                 + 0.0218856589075397 w_k1_FDX_s111
                                 + 0.023188683054089 w_k1_FDX_s112
                                 + 0.0193993700298728 w_k1_FDX_s113
                                 + 0.0154022634826975 w_k1_FDX_s114
                                 + 0.0156847685816886 w_k1_FDX_s115
                                 + 0.0220564451528591 w_k1_FDX_s116
                                 + 0.0146391773436354 w_k1_FDX_s117
                                 + 0.0584745503928792 w_k1_IBM_s000
                                 + 0.0143372984066612 w_k1_IBM_s001
                                 + 0.0430393731590104 w_k1_IBM_s002
                                 + 0.0283051667368224 w_k1_IBM_s003
                                 + 0.0285685121122978 w_k1_IBM_s004
                                 + 0.0270924000329466 w_k1_IBM_s005
                                 + 0.0648238445018511 w_k1_IBM_s006
                                 + 0.047869892051956 w_k1_IBM_s007
                                 + 0.0269777432069514 w_k1_IBM_s010
                                 + 0.0129228648341051 w_k1_IBM_s011
                                 + 0.0152338451816935 w_k1_IBM_s012
                                 + 0.0348615315891058 w_k1_IBM_s013
                                 + 0.0100766093679767 w_k1_IBM_s014
                                 + 0.00840193908620496 w_k1_IBM_s015
                                 + 0.0152225299891578 w_k1_IBM_s016
                                 + 0.00849901308624242 w_k1_IBM_s017
                                 + 0.0556465623285204 w_k1_IBM_s100
                                 + 0.0740232621227145 w_k1_IBM_s101
                                 + 0.0177137135880511 w_k1_IBM_s102
                                 + 0.0781207512485867 w_k1_IBM_s103
                                 + 0.0624551679520102 w_k1_IBM_s104
                                 + 0.0280017509334339 w_k1_IBM_s105
                                 + 0.0631943364720478 w_k1_IBM_s106
                                 + 0.0278214559891404 w_k1_IBM_s107
                                 + 0.0424507973454348 w_k1_IBM_s110
                                 + 0.0236249513988244 w_k1_IBM_s111
                                 + 0.024320957383656 w_k1_IBM_s112
                                 + 0.0201785191559739 w_k1_IBM_s113
                                 + 0.0154835855179597 w_k1_IBM_s114
                                 + 0.0158461983181141 w_k1_IBM_s115
                                 + 0.0222024798781302 w_k1_IBM_s116
                                 + 0.0149913666734203 w_k1_IBM_s117
                                 + 0.0509384885145955 w_k1_INTC_s000
                                 + 0.0129655890238804 w_k1_INTC_s001
                                 + 0.040192953030217 w_k1_INTC_s002
                                 + 0.0285556220513289 w_k1_INTC_s003
                                 + 0.0334697613889782 w_k1_INTC_s004
                                 + 0.0252738672001374 w_k1_INTC_s005
                                 + 0.0602531506965447 w_k1_INTC_s006
                                 + 0.0452300715158051 w_k1_INTC_s007
                                 + 0.0273731085883348 w_k1_INTC_s010
                                 + 0.0109841339016877 w_k1_INTC_s011
                                 + 0.0157224984317743 w_k1_INTC_s012
                                 + 0.0342586881089404 w_k1_INTC_s013
                                 + 0.0120041085835403 w_k1_INTC_s014
                                 + 0.00826857378131395 w_k1_INTC_s015
                                 + 0.0119982490733993 w_k1_INTC_s016
                                 + 0.00838186916873051 w_k1_INTC_s017
                                 + 0.0697386395306457 w_k1_INTC_s100
                                 + 0.0714822827346918 w_k1_INTC_s101
                                 + 0.0154638227861153 w_k1_INTC_s102
                                 + 0.070659223821854 w_k1_INTC_s103
                                 + 0.0655022230931904 w_k1_INTC_s104
                                 + 0.0313536648715425 w_k1_INTC_s105
                                 + 0.0738872143701283 w_k1_INTC_s106
                                 + 0.0300505111247861 w_k1_INTC_s107
                                 + 0.0402330881257493 w_k1_INTC_s110
                                 + 0.023240311109083 w_k1_INTC_s111
                                 + 0.0234270512959731 w_k1_INTC_s112
                                 + 0.017210880514085 w_k1_INTC_s113
                                 + 0.0160292595630892 w_k1_INTC_s114
                                 + 0.01474762016991 w_k1_INTC_s115
                                 + 0.0227908971472739 w_k1_INTC_s116
                                 + 0.0157793328404133 w_k1_INTC_s117
                                 + 0.0555108734065694 w_k1_KO_s000
                                 + 0.0147952807853415 w_k1_KO_s001
                                 + 0.0448350585831326 w_k1_KO_s002
                                 + 0.0275509575942655 w_k1_KO_s003
                                 + 0.0278282914215263 w_k1_KO_s004
                                 + 0.0262480063089436 w_k1_KO_s005
                                 + 0.0578466429363107 w_k1_KO_s006
                                 + 0.0448673032355152 w_k1_KO_s007
                                 + 0.0267802152110639 w_k1_KO_s010
                                 + 0.012771222160676 w_k1_KO_s011
                                 + 0.014215540736221 w_k1_KO_s012
                                 + 0.0363646465249495 w_k1_KO_s013
                                 + 0.0100662521384726 w_k1_KO_s014
                                 + 0.00865751573673907 w_k1_KO_s015
                                 + 0.015984620644304 w_k1_KO_s016
                                 + 0.0081415566325946 w_k1_KO_s017
                                 + 0.0542333114793932 w_k1_KO_s100
                                 + 0.0816043381631937 w_k1_KO_s101
                                 + 0.0152459165121814 w_k1_KO_s102
                                 + 0.080628880974708 w_k1_KO_s103
                                 + 0.0656563483570247 w_k1_KO_s104
                                 + 0.0264669282611836 w_k1_KO_s105
                                 + 0.0653944352855394 w_k1_KO_s106
                                 + 0.0290881409929423 w_k1_KO_s107
                                 + 0.0471100479418358 w_k1_KO_s110
                                 + 0.0236671054007238 w_k1_KO_s111
                                 + 0.0233744649565253 w_k1_KO_s112
                                 + 0.0175208637568536 w_k1_KO_s113
                                 + 0.0145974597899515 w_k1_KO_s114
                                 + 0.0157232337126073 w_k1_KO_s115
                                 + 0.0223752820264908 w_k1_KO_s116
                                 + 0.0151430525997835 w_k1_KO_s117
                                 + 0.0580076285688361 w_k1_MAT_s000
                                 + 0.0156601592625772 w_k1_MAT_s001
                                 + 0.0411048236415531 w_k1_MAT_s002
                                 + 0.0269575406820237 w_k1_MAT_s003
                                 + 0.0254222288471007 w_k1_MAT_s004
                                 + 0.0274616171970357 w_k1_MAT_s005
                                 + 0.068229535146543 w_k1_MAT_s006
                                 + 0.0552318163274343 w_k1_MAT_s007
                                 + 0.0259234345674325 w_k1_MAT_s010
                                 + 0.0129127764393465 w_k1_MAT_s011
                                 + 0.0152377455080053 w_k1_MAT_s012
                                 + 0.0408691345611714 w_k1_MAT_s013
                                 + 0.0103316860706961 w_k1_MAT_s014
                                 + 0.00834320658057823 w_k1_MAT_s015
                                 + 0.0173694529974732 w_k1_MAT_s016
                                 + 0.00782719311580064 w_k1_MAT_s017
                                 + 0.0590902430411891 w_k1_MAT_s100
                                 + 0.0698348405034899 w_k1_MAT_s101
                                 + 0.0183786671621278 w_k1_MAT_s102
                                 + 0.0850188636134442 w_k1_MAT_s103
                                 + 0.0608009031051799 w_k1_MAT_s104
                                 + 0.0307862783311545 w_k1_MAT_s105
                                 + 0.0632852874990259 w_k1_MAT_s106
                                 + 0.0275627656367518 w_k1_MAT_s107
                                 + 0.0410501245376647 w_k1_MAT_s110
                                 + 0.0223449707473926 w_k1_MAT_s111
                                 + 0.0229148267019444 w_k1_MAT_s112
                                 + 0.0191860915381877 w_k1_MAT_s113
                                 + 0.0146002541262369 w_k1_MAT_s114
                                 + 0.015109644722057 w_k1_MAT_s115
                                 + 0.0205792651643158 w_k1_MAT_s116
                                 + 0.0153988135526178 w_k1_MAT_s117
                                 + 0.0554658035267508 w_k1_MSFT_s000
                                 + 0.0133159380282323 w_k1_MSFT_s001
                                 + 0.0460576781937943 w_k1_MSFT_s002
                                 + 0.02982339425097 w_k1_MSFT_s003
                                 + 0.0282133338554035 w_k1_MSFT_s004
                                 + 0.0251792670516299 w_k1_MSFT_s005
                                 + 0.0606864418165339 w_k1_MSFT_s006
                                 + 0.04159421546728 w_k1_MSFT_s007
                                 + 0.029251431569851 w_k1_MSFT_s010
                                 + 0.0111180522577631 w_k1_MSFT_s011
                                 + 0.0138173446656024 w_k1_MSFT_s012
                                 + 0.0332451242498063 w_k1_MSFT_s013
                                 + 0.010650892046604 w_k1_MSFT_s014
                                 + 0.00841609306858418 w_k1_MSFT_s015
                                 + 0.0143388174946684 w_k1_MSFT_s016
                                 + 0.00877177121130795 w_k1_MSFT_s017
                                 + 0.0585834760840077 w_k1_MSFT_s100
                                 + 0.0743893898002885 w_k1_MSFT_s101
                                 + 0.0157081660881803 w_k1_MSFT_s102
                                 + 0.075553680180102 w_k1_MSFT_s103
                                 + 0.0652967319905367 w_k1_MSFT_s104
                                 + 0.0255628578861572 w_k1_MSFT_s105
                                 + 0.0693347870396665 w_k1_MSFT_s106
                                 + 0.025473683473075 w_k1_MSFT_s107
                                 + 0.0412163153022827 w_k1_MSFT_s110
                                 + 0.0244588544151196 w_k1_MSFT_s111
                                 + 0.0228935492901562 w_k1_MSFT_s112
                                 + 0.0181426941564102 w_k1_MSFT_s113
                                 + 0.0165374507755754 w_k1_MSFT_s114
                                 + 0.0156197719610845 w_k1_MSFT_s115
                                 + 0.0194923887273873 w_k1_MSFT_s116
                                 + 0.0159786173499837 w_k1_MSFT_s117
                                 + 0.056887528128067 w_k1_PG_s000
                                 + 0.0136058520083546 w_k1_PG_s001
                                 + 0.0410832933859911 w_k1_PG_s002
                                 + 0.0270666513200153 w_k1_PG_s003
                                 + 0.025109481612314 w_k1_PG_s004
                                 + 0.02688354254476 w_k1_PG_s005
                                 + 0.0633406756912432 w_k1_PG_s006
                                 + 0.0461520929163994 w_k1_PG_s007
                                 + 0.0283953655315113 w_k1_PG_s010
                                 + 0.0141615936213056 w_k1_PG_s011
                                 + 0.0139724263886838 w_k1_PG_s012
                                 + 0.035466107556143 w_k1_PG_s013
                                 + 0.00992239475015905 w_k1_PG_s014
                                 + 0.00821733761302512 w_k1_PG_s015
                                 + 0.0159938365628554 w_k1_PG_s016
                                 + 0.0082130104325672 w_k1_PG_s017
                                 + 0.0575816867407442 w_k1_PG_s100
                                 + 0.0747638260342565 w_k1_PG_s101
                                 + 0.0168298955778324 w_k1_PG_s102
                                 + 0.0759951576637484 w_k1_PG_s103
                                 + 0.0637973108725578 w_k1_PG_s104
                                 + 0.0252268385619955 w_k1_PG_s105
                                 + 0.0626162852267112 w_k1_PG_s106
                                 + 0.0286018927844483 w_k1_PG_s107
                                 + 0.0453785071920306 w_k1_PG_s110
                                 + 0.0222652186604979 w_k1_PG_s111
                                 + 0.0235942589380339 w_k1_PG_s112
                                 + 0.0185559039392889 w_k1_PG_s113
                                 + 0.014303727256498 w_k1_PG_s114
                                 + 0.0157528293106431 w_k1_PG_s115
                                 + 0.0207974691817699 w_k1_PG_s116
                                 + 0.0161333977860964 w_k1_PG_s117 >= 1.12
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
