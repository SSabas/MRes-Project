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
 w_k1_ADM_s0_return_constraint:  1.01068 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.899527 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.938464 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.35366 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.944255 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.01728 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.5323 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.06495 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.864665 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.27058 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.786274 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.04979 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.34631 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.981169 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.23475 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.880857 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.21565 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.855497 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.873672 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.975757 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.1129 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.0425 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.934184 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.715675 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.801251 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.21539 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.27109 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.10905 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.887678 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.24176 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.30016 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.725286 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.965261 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.11414 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.50792 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.879305 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.797858 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06864 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.11703 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.817027 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.88753 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.46172 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.887436 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.08354 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.92027 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.32577 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.856981 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.965787 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.605922 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.999689 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.33072 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.78197 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.44519 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.823749 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.15519 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.887506 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.860416 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.03443 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.968826 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.31473 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.0759 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.697269 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.652651 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.24069 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.48383 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.947535 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.80297 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.65102 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.19669 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.673872 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.893211 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.35621 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.55245 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.978004 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.719945 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.04781 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.18869 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.804889 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.895739 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.51971 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.820544 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.07813 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.84075 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.30717 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.723687 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.10001 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.523428 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.850761 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.17655 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.589831 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.525 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.755202 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.17341 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.819892 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.742466 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.12547 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.987989 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.39748 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.91611 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.661502 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.607748 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.47595 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.18163 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.27057 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.894925 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.78959 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.40012 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.605816 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.821443 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.31735 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.81483 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.773232 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.707903 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.10898 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.0962 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.887164 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.90582 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.30532 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.988798 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.01674 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.56931 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.05981 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.773007 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.1396 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.665556 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.00727 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.15011 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.88116 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.40203 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.831841 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.16684 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.7878 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.90779 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.950673 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.88645 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.14692 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.934026 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.665728 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.707258 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.09622 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.22102 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.1103 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.732257 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.53172 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.36815 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.697818 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.952766 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.08647 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.44129 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.879334 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.745546 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.06986 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15887 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.801256 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.95051 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.2111 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.728565 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.18599 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.36456 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.76004 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.0135 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.935225 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.692783 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.655719 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.18958 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.565216 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.968524 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.02011 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.956899 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.32836 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.666685 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.12043 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.61762 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.89193 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.08154 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.953769 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.657434 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.4447 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.905316 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.88868 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.33569 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.21343 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.847112 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.842765 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.712788 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.90232 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.24192 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.10013 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.39685 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.16417 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17648 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.913271 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.918834 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06093 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.992507 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.10114 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.50837 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.0492 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.877094 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.993451 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.702629 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.883123 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.08317 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.738901 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.16432 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.768995 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.25208 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.00767 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.976934 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.14393 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.973267 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.10958 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.897197 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.867855 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.689495 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.999862 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.03493 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.971004 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.762512 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.33461 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.18728 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.649208 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.867437 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.09403 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.29587 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.748692 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.828258 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.933297 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13523 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.856799 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.899917 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.28822 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.824684 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.07277 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.54982 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.37658 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.907776 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.03323 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.584351 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.876597 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.09432 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.670546 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.19143 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.829002 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.0312 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.00116 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.801899 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.19579 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.24742 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.49899 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.93288 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.802701 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.630199 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.30653 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.916525 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.991177 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.04393 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.26207 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.11344 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.727685 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.901453 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.28714 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.40064 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.858058 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.934336 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.02342 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25446 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.811968 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.848593 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.32074 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.583576 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.11785 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.38906 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.57476 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.995731 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.960803 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.556369 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.863256 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.29839 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.624664 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.935854 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.982131 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.943056 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.13975 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.582137 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.972106 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.58133 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.01972 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.47984 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.02013 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.808692 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.73935 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.07056 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.05625 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.52552 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.75892 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.898935 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.75936 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.641195 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 2.07556 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.51627 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.38421 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.19983 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.26191 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09399 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.9199 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.90031 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.2332 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.928291 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.01313 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.5948 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.09149 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.862334 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.1772 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.703408 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.01328 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.11938 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.84162 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.3074 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.821458 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.10222 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.948028 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.954612 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.16546 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.940869 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.24147 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.943731 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.784547 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.728879 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.08906 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.13588 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.185 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.799592 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.40185 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.29159 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.711178 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.0179 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.04105 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.40563 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.829253 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.779576 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.02818 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.0643 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.893641 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.978539 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.23647 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.958496 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11189 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.26187 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.20871 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.815618 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02624 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.737179 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.879253 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04592 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.806356 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.20489 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.835436 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00263 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.918353 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.867469 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.08091 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.10451 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.09578 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.963035 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.799514 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.770025 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.06952 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.01349 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.990776 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.890546 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.31333 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.01034 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.779028 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.891515 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.11624 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.22985 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.88317 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.856286 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.959866 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.248999326450379 z_k1_s0000
                                 - 0.358999482906801 z_k1_s0001
                                 - 0.228000640509975 z_k1_s0002
                                 - 0.245999203601455 z_k1_s0003
                                 - 0.145000067892425 z_k1_s0004
                                 - 0.248999326450379 z_k1_s0005
                                 - 0.290001005385675 z_k1_s0006
                                 - 0.407999315437163 z_k1_s0007
                                 - 0.407998803164354 z_k1_s0010
                                 - 0.766999277912318 z_k1_s0011
                                 - 0.547997158215755 z_k1_s0012
                                 - 0.541998007823993 z_k1_s0013
                                 - 0.247999372452533 z_k1_s0014
                                 - 1.02899445925248 z_k1_s0015
                                 - 1.02199629995946 z_k1_s0016
                                 - 0.532996733744254 z_k1_s0017
                                 - 1.16599938189416 z_k1_s0020
                                 - 0.749000461128782 z_k1_s0021
                                 - 0.814998179926583 z_k1_s0022
                                 - 0.951998360885199 z_k1_s0023
                                 - 1.10000088499571 z_k1_s0024
                                 - 1.13000444595308 z_k1_s0025
                                 - 1.29300096953581 z_k1_s0026
                                 - 0.576000675625205 z_k1_s0027
                                 - 1.01299991184325 z_k1_s0030
                                 - 1.4620044467782 z_k1_s0031
                                 - 1.01000065229671 z_k1_s0032
                                 - 1.08300186061854 z_k1_s0033
                                 - 0.539000586052686 z_k1_s0034
                                 - 1.51100196571635 z_k1_s0035
                                 - 2.37000143545851 z_k1_s0036
                                 - 0.62500050834512 z_k1_s0037
                                 - 0.363000008285535 z_k1_s0040
                                 - 0.320000120788921 z_k1_s0041
                                 - 0.775001285113966 z_k1_s0042
                                 - 0.402001670085097 z_k1_s0043
                                 - 1.0020011746892 z_k1_s0044
                                 - 0.385000067121032 z_k1_s0045
                                 - 0.298999907127737 z_k1_s0046
                                 - 0.303999902999438 z_k1_s0047
                                 - 0.45600142277011 z_k1_s0050
                                 - 0.325001113200048 z_k1_s0051
                                 - 0.5970003752406 z_k1_s0052
                                 - 1.19900112353458 z_k1_s0053
                                 - 0.572003767989529 z_k1_s0054
                                 - 1.06600487115583 z_k1_s0055
                                 - 0.59799939824811 z_k1_s0056
                                 - 0.445001127854257 z_k1_s0057
                                 - 0.741999749036849 z_k1_s0060
                                 - 1.62299997058571 z_k1_s0061
                                 - 0.981002179240076 z_k1_s0062
                                 - 0.728999976396729 z_k1_s0063
                                 - 1.22199629636166 z_k1_s0064
                                 - 0.446999595544708 z_k1_s0065
                                 - 0.530999794426115 z_k1_s0066
                                 - 1.75600128935572 z_k1_s0067
                                 - 0.127999930086166 z_k1_s0070
                                 - 0.449000708668036 z_k1_s0071
                                 - 0.298998939723003 z_k1_s0072
                                 - 0.202000088992263 z_k1_s0073
                                 - 0.117999964023189 z_k1_s0074
                                 - 0.436000160506068 z_k1_s0075
                                 - 0.123999989220982 z_k1_s0076
                                 - 0.521999230807563 z_k1_s0077
                                 - 0.411999991225959 z_k1_s0100
                                 - 0.144999933987788 z_k1_s0101
                                 - 0.19500002635597 z_k1_s0102
                                 - 0.202999004143063 z_k1_s0103
                                 - 0.16600000027842 z_k1_s0104
                                 - 0.201999328319641 z_k1_s0105
                                 - 0.416000658519297 z_k1_s0106
                                 - 0.224999728256942 z_k1_s0107
                                 - 0.595999792010135 z_k1_s0110
                                 - 0.56899963483165 z_k1_s0111
                                 - 0.144000228584619 z_k1_s0112
                                 - 0.124000044411596 z_k1_s0113
                                 - 0.159000170739182 z_k1_s0114
                                 - 0.143000232440982 z_k1_s0115
                                 - 0.484001269123825 z_k1_s0116
                                 - 0.39400161619645 z_k1_s0117
                                 - 0.205999377991616 z_k1_s0120
                                 - 0.152999460342526 z_k1_s0121
                                 - 0.360998333637348 z_k1_s0122
                                 - 0.184999386272231 z_k1_s0123
                                 - 0.2979988746778 z_k1_s0124
                                 - 0.635997979391861 z_k1_s0125
                                 - 0.449998716418375 z_k1_s0126
                                 - 0.291998065874451 z_k1_s0127
                                 - 0.0859998792455938 z_k1_s0130
                                 - 0.174999099582484 z_k1_s0131
                                 - 0.0999997851322375 z_k1_s0132
                                 - 0.217999812003706 z_k1_s0133
                                 - 0.0969997816422906 z_k1_s0134
                                 - 0.202999794553972 z_k1_s0135
                                 - 0.133999935084744 z_k1_s0136
                                 - 0.0909997746623968 z_k1_s0137
                                 - 0.366000253616144 z_k1_s0140
                                 - 0.672000838557546 z_k1_s0141
                                 - 1.03300070770076 z_k1_s0142
                                 - 0.685001244787119 z_k1_s0143
                                 - 0.771998638015294 z_k1_s0144
                                 - 0.277000135198552 z_k1_s0145
                                 - 0.429000054074638 z_k1_s0146
                                 - 0.711002057246264 z_k1_s0147
                                 - 0.212998914212166 z_k1_s0150
                                 - 0.15700012743177 z_k1_s0151
                                 - 0.117999833140257 z_k1_s0152
                                 - 0.234000209751775 z_k1_s0153
                                 - 0.388999324657507 z_k1_s0154
                                 - 0.219000395792982 z_k1_s0155
                                 - 0.114999870348499 z_k1_s0156
                                 - 0.110999868092911 z_k1_s0157
                                 - 0.13000009689654 z_k1_s0160
                                 - 0.116999668007244 z_k1_s0161
                                 - 0.109000290843843 z_k1_s0162
                                 - 0.233999336014488 z_k1_s0163
                                 - 0.101999866426695 z_k1_s0164
                                 - 0.0669999451390754 z_k1_s0165
                                 - 0.121999671734034 z_k1_s0166
                                 - 0.166999705275144 z_k1_s0167
                                 - 0.23700033702277 z_k1_s0170
                                 - 0.0879998391922633 z_k1_s0171
                                 - 0.34999893361735 z_k1_s0172
                                 - 0.19299919267777 z_k1_s0173
                                 - 0.157999789215582 z_k1_s0174
                                 - 0.207999916817802 z_k1_s0175
                                 - 0.305999422270842 z_k1_s0176
                                 - 0.0929999172724249 z_k1_s0177
                                 - 0.152000435814414 z_k1_s1000
                                 - 0.30399991482683 z_k1_s1001
                                 - 0.0990003027370591 z_k1_s1002
                                 - 0.33000100912353 z_k1_s1003
                                 - 0.297000908211177 z_k1_s1004
                                 - 0.460001696597042 z_k1_s1005
                                 - 0.256999408678176 z_k1_s1006
                                 - 0.493001797509395 z_k1_s1007
                                 - 0.102000352678556 z_k1_s1010
                                 - 0.160000190553367 z_k1_s1011
                                 - 0.198000351410184 z_k1_s1012
                                 - 0.0690002555767277 z_k1_s1013
                                 - 0.16900083707415 z_k1_s1014
                                 - 0.186001030721034 z_k1_s1015
                                 - 0.217000431838592 z_k1_s1016
                                 - 0.055000238903308 z_k1_s1017
                                 - 0.344000878824827 z_k1_s1020
                                 - 0.105000204304577 z_k1_s1021
                                 - 0.101000215699669 z_k1_s1022
                                 - 0.329000340680564 z_k1_s1023
                                 - 0.195000799866263 z_k1_s1024
                                 - 0.128000254957969 z_k1_s1025
                                 - 0.197000561818374 z_k1_s1026
                                 - 0.150000269735077 z_k1_s1027
                                 - 0.14000048868667 z_k1_s1030
                                 - 0.0540001015932785 z_k1_s1031
                                 - 0.175000393607959 z_k1_s1032
                                 - 0.085000166352394 z_k1_s1033
                                 - 0.107000156245862 z_k1_s1034
                                 - 0.147000295870625 z_k1_s1035
                                 - 0.0950001143584335 z_k1_s1036
                                 - 0.0660001434807073 z_k1_s1037
                                 - 0.134000379735235 z_k1_s1040
                                 - 0.0969998439124019 z_k1_s1041
                                 - 0.13100037683072 z_k1_s1042
                                 - 0.114000110371561 z_k1_s1043
                                 - 0.0660000638993247 z_k1_s1044
                                 - 0.0810000784218985 z_k1_s1045
                                 - 0.0380000617905445 z_k1_s1046
                                 - 0.0890003361675132 z_k1_s1047
                                 - 0.584001834849409 z_k1_s1050
                                 - 0.155000567237348 z_k1_s1051
                                 - 0.154000441978646 z_k1_s1052
                                 - 0.119000302937919 z_k1_s1053
                                 - 0.16400050596179 z_k1_s1054
                                 - 0.245000124282324 z_k1_s1055
                                 - 0.127000456004766 z_k1_s1056
                                 - 0.150000450345499 z_k1_s1057
                                 - 0.100000243729968 z_k1_s1060
                                 - 0.0520000532194529 z_k1_s1061
                                 - 0.107000279171099 z_k1_s1062
                                 - 0.138999916043905 z_k1_s1063
                                 - 0.0959998296200511 z_k1_s1064
                                 - 0.146999825262106 z_k1_s1065
                                 - 0.101999991284111 z_k1_s1066
                                 - 0.176000759805333 z_k1_s1067
                                 - 0.0990003919068163 z_k1_s1070
                                 - 0.15700097700909 z_k1_s1071
                                 - 0.227000166161413 z_k1_s1072
                                 - 0.11200037134987 z_k1_s1073
                                 - 0.14900059290421 z_k1_s1074
                                 - 0.390000900297206 z_k1_s1075
                                 - 0.118000431877692 z_k1_s1076
                                 - 0.26500115630652 z_k1_s1077
                                 - 0.230000835787653 z_k1_s1100
                                 - 0.962001266548762 z_k1_s1101
                                 - 0.789002998245854 z_k1_s1102
                                 - 0.352001033248207 z_k1_s1103
                                 - 0.281000791635545 z_k1_s1104
                                 - 0.666002616846563 z_k1_s1105
                                 - 0.254000726304047 z_k1_s1106
                                 - 0.236000808416751 z_k1_s1107
                                 - 0.291999917857474 z_k1_s1110
                                 - 0.394999979622226 z_k1_s1111
                                 - 0.85999794327969 z_k1_s1112
                                 - 0.516998547911261 z_k1_s1113
                                 - 0.374999924755304 z_k1_s1114
                                 - 0.549001285297127 z_k1_s1115
                                 - 1.03900042153774 z_k1_s1116
                                 - 0.388999963162149 z_k1_s1117
                                 - 0.3749995714611 z_k1_s1120
                                 - 0.937997315251217 z_k1_s1121
                                 - 0.478998899736112 z_k1_s1122
                                 - 1.60999420167215 z_k1_s1123
                                 - 0.751996369709406 z_k1_s1124
                                 - 0.361999195377556 z_k1_s1125
                                 - 1.09999596221092 z_k1_s1126
                                 - 0.517998801188964 z_k1_s1127
                                 - 0.169999894313114 z_k1_s1130
                                 - 0.133999970482095 z_k1_s1131
                                 - 0.221999149291326 z_k1_s1132
                                 - 0.101999974687864 z_k1_s1133
                                 - 0.508000576170849 z_k1_s1134
                                 - 0.107999961993034 z_k1_s1135
                                 - 0.386000580299093 z_k1_s1136
                                 - 0.275000053153541 z_k1_s1137
                                 - 0.230999077981184 z_k1_s1140
                                 - 0.468998634338588 z_k1_s1141
                                 - 0.479997760936204 z_k1_s1142
                                 - 0.175999179009941 z_k1_s1143
                                 - 0.322998559952641 z_k1_s1144
                                 - 0.758996615011008 z_k1_s1145
                                 - 0.739998769976213 z_k1_s1146
                                 - 0.376999152367132 z_k1_s1147
                                 - 0.237000601688795 z_k1_s1150
                                 - 0.2930004251764 z_k1_s1151
                                 - 0.440001010831503 z_k1_s1152
                                 - 0.322999731187537 z_k1_s1153
                                 - 0.260000978764121 z_k1_s1154
                                 - 0.246999671025747 z_k1_s1155
                                 - 0.139999978319069 z_k1_s1156
                                 - 0.157999981525807 z_k1_s1157
                                 - 0.138999883509122 z_k1_s1160
                                 - 0.137999930335637 z_k1_s1161
                                 - 0.0729999300616324 z_k1_s1162
                                 - 0.240999368801062 z_k1_s1163
                                 - 0.145999860123265 z_k1_s1164
                                 - 0.171999555833118 z_k1_s1165
                                 - 0.299999650034157 z_k1_s1166
                                 - 0.312999041289461 z_k1_s1167
                                 - 0.738002355245792 z_k1_s1170
                                 - 0.242000787197711 z_k1_s1171
                                 - 0.271000736658207 z_k1_s1172
                                 - 0.399000950553977 z_k1_s1173
                                 - 1.00200306040652 z_k1_s1174
                                 - 0.20400073689744 z_k1_s1175
                                 - 0.599999682632893 z_k1_s1176
                                 - 0.768001283374616 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.894339 w_k1_ADM_s000
                                 + 0.807535 w_k1_AXP_s000
                                 + 0.840515 w_k1_DIS_s000
                                 + 0.957971 w_k1_FDX_s000
                                 + 1.19289 w_k1_IBM_s000
                                 + 0.787105 w_k1_INTC_s000
                                 + 0.997442 w_k1_KO_s000
                                 + 1.00871 w_k1_MAT_s000
                                 + 0.90639 w_k1_MSFT_s000
                                 + 1.01774 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.934556 w_k1_ADM_s000
                                 + 1.04594 w_k1_AXP_s000
                                 + 1.23458 w_k1_DIS_s000
                                 + 0.93123 w_k1_FDX_s000
                                 + 1.39741 w_k1_IBM_s000
                                 + 0.886472 w_k1_INTC_s000
                                 + 1.16463 w_k1_KO_s000 + 1.68328 w_k1_MAT_s000
                                 + 0.864759 w_k1_MSFT_s000
                                 + 1.05749 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03569 w_k1_ADM_s000
                                 + 0.976669 w_k1_AXP_s000
                                 + 1.13702 w_k1_DIS_s000
                                 + 1.24604 w_k1_FDX_s000
                                 + 0.696674 w_k1_IBM_s000
                                 + 0.985896 w_k1_INTC_s000
                                 + 0.954813 w_k1_KO_s000
                                 + 1.07538 w_k1_MAT_s000
                                 + 1.07387 w_k1_MSFT_s000
                                 + 1.07928 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.34178 w_k1_ADM_s000
                                 + 1.03463 w_k1_AXP_s000
                                 + 1.63808 w_k1_DIS_s000
                                 + 1.36937 w_k1_FDX_s000
                                 + 1.10731 w_k1_IBM_s000
                                 + 1.07876 w_k1_INTC_s000
                                 + 1.30116 w_k1_KO_s000 + 1.08831 w_k1_MAT_s000
                                 + 1.20904 w_k1_MSFT_s000
                                 + 1.12858 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.27708 w_k1_ADM_s000
                                 + 1.62987 w_k1_AXP_s000 + 1.4001 w_k1_DIS_s000
                                 + 1.2629 w_k1_FDX_s000 + 1.39949 w_k1_IBM_s000
                                 + 0.993643 w_k1_INTC_s000
                                 + 1.23058 w_k1_KO_s000 + 1.64421 w_k1_MAT_s000
                                 + 0.998943 w_k1_MSFT_s000
                                 + 1.21497 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.750042 w_k1_ADM_s000
                                 + 0.675727 w_k1_AXP_s000
                                 + 0.642506 w_k1_DIS_s000
                                 + 0.724749 w_k1_FDX_s000
                                 + 0.772431 w_k1_IBM_s000
                                 + 0.692132 w_k1_INTC_s000
                                 + 0.624948 w_k1_KO_s000
                                 + 0.641344 w_k1_MAT_s000
                                 + 0.661259 w_k1_MSFT_s000
                                 + 0.885726 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.921221 w_k1_ADM_s000
                                 + 1.02003 w_k1_AXP_s000
                                 + 0.881749 w_k1_DIS_s000
                                 + 0.883888 w_k1_FDX_s000
                                 + 0.995019 w_k1_IBM_s000
                                 + 0.777865 w_k1_INTC_s000
                                 + 0.976917 w_k1_KO_s000
                                 + 1.32588 w_k1_MAT_s000
                                 + 0.800036 w_k1_MSFT_s000
                                 + 0.908869 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.852763 w_k1_ADM_s000
                                 + 0.760548 w_k1_AXP_s000
                                 + 0.881001 w_k1_DIS_s000
                                 + 0.966574 w_k1_FDX_s000
                                 + 0.694394 w_k1_IBM_s000
                                 + 0.872108 w_k1_INTC_s000
                                 + 0.690679 w_k1_KO_s000
                                 + 0.651141 w_k1_MAT_s000
                                 + 0.856906 w_k1_MSFT_s000
                                 + 0.864984 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.20751 w_k1_ADM_s001
                                 + 1.351 w_k1_AXP_s001 + 0.88216 w_k1_DIS_s001
                                 + 1.09282 w_k1_FDX_s001
                                 + 0.947263 w_k1_IBM_s001
                                 + 0.983411 w_k1_INTC_s001
                                 + 0.939028 w_k1_KO_s001
                                 + 1.74826 w_k1_MAT_s001
                                 + 1.03712 w_k1_MSFT_s001
                                 + 0.873743 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.09951 w_k1_ADM_s001
                                 + 0.960411 w_k1_AXP_s001
                                 + 0.937946 w_k1_DIS_s001
                                 + 0.937475 w_k1_FDX_s001
                                 + 1.42184 w_k1_IBM_s001
                                 + 0.875917 w_k1_INTC_s001
                                 + 1.0525 w_k1_KO_s001 + 1.36294 w_k1_MAT_s001
                                 + 0.91739 w_k1_MSFT_s001
                                 + 0.894941 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.10681 w_k1_ADM_s001
                                 + 1.27849 w_k1_AXP_s001
                                 + 1.62553 w_k1_DIS_s001 + 1.263 w_k1_FDX_s001
                                 + 1.47327 w_k1_IBM_s001
                                 + 1.0178 w_k1_INTC_s001 + 1.34525 w_k1_KO_s001
                                 + 1.46359 w_k1_MAT_s001
                                 + 1.22204 w_k1_MSFT_s001 + 1.072 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.11926 w_k1_ADM_s001
                                 + 0.793417 w_k1_AXP_s001
                                 + 1.13996 w_k1_DIS_s001
                                 + 1.06727 w_k1_FDX_s001
                                 + 0.842152 w_k1_IBM_s001
                                 + 1.01769 w_k1_INTC_s001
                                 + 1.00545 w_k1_KO_s001
                                 + 0.733961 w_k1_MAT_s001
                                 + 1.17654 w_k1_MSFT_s001
                                 + 0.955186 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.30045 w_k1_ADM_s001
                                 + 1.52382 w_k1_AXP_s001
                                 + 1.33871 w_k1_DIS_s001
                                 + 1.54284 w_k1_FDX_s001
                                 + 0.805243 w_k1_IBM_s001
                                 + 1.40745 w_k1_INTC_s001
                                 + 1.12495 w_k1_KO_s001 + 1.12066 w_k1_MAT_s001
                                 + 1.16165 w_k1_MSFT_s001 + 1.1053 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.720892 w_k1_ADM_s001
                                 + 0.714327 w_k1_AXP_s001
                                 + 0.653573 w_k1_DIS_s001
                                 + 0.722774 w_k1_FDX_s001
                                 + 0.961846 w_k1_IBM_s001
                                 + 0.821325 w_k1_INTC_s001
                                 + 0.874844 w_k1_KO_s001
                                 + 1.03089 w_k1_MAT_s001
                                 + 0.839325 w_k1_MSFT_s001
                                 + 0.724513 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.942717 w_k1_ADM_s001
                                 + 0.970478 w_k1_AXP_s001
                                 + 0.917817 w_k1_DIS_s001
                                 + 0.967776 w_k1_FDX_s001
                                 + 0.909478 w_k1_IBM_s001
                                 + 1.09862 w_k1_INTC_s001
                                 + 1.01145 w_k1_KO_s001
                                 + 0.946738 w_k1_MAT_s001
                                 + 0.997738 w_k1_MSFT_s001
                                 + 0.878472 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.23546 w_k1_ADM_s001
                                 + 0.969878 w_k1_AXP_s001
                                 + 1.24112 w_k1_DIS_s001
                                 + 1.16138 w_k1_FDX_s001
                                 + 0.760525 w_k1_IBM_s001
                                 + 1.12345 w_k1_INTC_s001 + 1.1745 w_k1_KO_s001
                                 + 0.677834 w_k1_MAT_s001
                                 + 1.36159 w_k1_MSFT_s001
                                 + 0.941119 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.24203 w_k1_ADM_s002
                                 + 1.22056 w_k1_AXP_s002
                                 + 0.983949 w_k1_DIS_s002
                                 + 1.10634 w_k1_FDX_s002
                                 + 0.94356 w_k1_IBM_s002
                                 + 1.01636 w_k1_INTC_s002
                                 + 0.96919 w_k1_KO_s002 + 1.25667 w_k1_MAT_s002
                                 + 0.94919 w_k1_MSFT_s002
                                 + 1.03287 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.993421 w_k1_ADM_s002
                                 + 0.991125 w_k1_AXP_s002
                                 + 1.14311 w_k1_DIS_s002
                                 + 0.998885 w_k1_FDX_s002
                                 + 1.75775 w_k1_IBM_s002
                                 + 0.885709 w_k1_INTC_s002
                                 + 1.24552 w_k1_KO_s002 + 1.41786 w_k1_MAT_s002
                                 + 0.912825 w_k1_MSFT_s002
                                 + 1.05769 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.17683 w_k1_ADM_s002
                                 + 1.17655 w_k1_AXP_s002
                                 + 1.41665 w_k1_DIS_s002
                                 + 1.40394 w_k1_FDX_s002
                                 + 0.776191 w_k1_IBM_s002
                                 + 0.993047 w_k1_INTC_s002
                                 + 0.962761 w_k1_KO_s002
                                 + 0.703177 w_k1_MAT_s002
                                 + 1.15348 w_k1_MSFT_s002
                                 + 1.23767 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.09116 w_k1_ADM_s002
                                 + 0.734767 w_k1_AXP_s002
                                 + 1.27376 w_k1_DIS_s002
                                 + 1.09145 w_k1_FDX_s002
                                 + 0.900557 w_k1_IBM_s002
                                 + 0.989422 w_k1_INTC_s002
                                 + 1.05799 w_k1_KO_s002
                                 + 0.700495 w_k1_MAT_s002
                                 + 1.20265 w_k1_MSFT_s002
                                 + 1.01622 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.986012 w_k1_ADM_s002
                                 + 0.681894 w_k1_AXP_s002
                                 + 0.643994 w_k1_DIS_s002
                                 + 0.769958 w_k1_FDX_s002
                                 + 0.756222 w_k1_IBM_s002
                                 + 0.79833 w_k1_INTC_s002
                                 + 0.711645 w_k1_KO_s002
                                 + 0.674471 w_k1_MAT_s002
                                 + 0.781156 w_k1_MSFT_s002
                                 + 0.8889 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.811244 w_k1_ADM_s002
                                 + 0.953585 w_k1_AXP_s002
                                 + 0.814315 w_k1_DIS_s002
                                 + 0.825342 w_k1_FDX_s002
                                 + 1.2114 w_k1_IBM_s002
                                 + 0.831349 w_k1_INTC_s002
                                 + 0.919679 w_k1_KO_s002
                                 + 1.00261 w_k1_MAT_s002
                                 + 0.847509 w_k1_MSFT_s002
                                 + 0.968997 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.0365 w_k1_ADM_s002
                                 + 1.01286 w_k1_AXP_s002
                                 + 1.11448 w_k1_DIS_s002 + 1.0357 w_k1_FDX_s002
                                 + 1.0527 w_k1_IBM_s002
                                 + 0.967766 w_k1_INTC_s002
                                 + 1.18323 w_k1_KO_s002 + 1.28047 w_k1_MAT_s002
                                 + 1.02853 w_k1_MSFT_s002
                                 + 1.09353 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.43785 w_k1_ADM_s002
                                 + 1.52866 w_k1_AXP_s002
                                 + 1.67154 w_k1_DIS_s002
                                 + 1.42397 w_k1_FDX_s002
                                 + 1.44856 w_k1_IBM_s002
                                 + 1.05083 w_k1_INTC_s002
                                 + 1.39971 w_k1_KO_s002 + 1.3987 w_k1_MAT_s002
                                 + 1.19898 w_k1_MSFT_s002
                                 + 1.33192 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.827031 w_k1_ADM_s003
                                 + 0.821782 w_k1_AXP_s003
                                 + 0.73614 w_k1_DIS_s003
                                 + 0.80462 w_k1_FDX_s003
                                 + 0.883843 w_k1_IBM_s003
                                 + 0.994689 w_k1_INTC_s003
                                 + 0.917619 w_k1_KO_s003
                                 + 1.00936 w_k1_MAT_s003
                                 + 0.864189 w_k1_MSFT_s003
                                 + 0.85087 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.985909 w_k1_ADM_s003
                                 + 1.0744 w_k1_AXP_s003 + 1.0056 w_k1_DIS_s003
                                 + 0.964814 w_k1_FDX_s003
                                 + 0.937192 w_k1_IBM_s003
                                 + 1.11773 w_k1_INTC_s003
                                 + 0.91054 w_k1_KO_s003
                                 + 0.927359 w_k1_MAT_s003
                                 + 0.920484 w_k1_MSFT_s003
                                 + 0.892556 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.913265 w_k1_ADM_s003
                                 + 1.33344 w_k1_AXP_s003
                                 + 1.64607 w_k1_DIS_s003
                                 + 1.14991 w_k1_FDX_s003
                                 + 1.08632 w_k1_IBM_s003 + 1.104 w_k1_INTC_s003
                                 + 1.28725 w_k1_KO_s003 + 1.31704 w_k1_MAT_s003
                                 + 1.19684 w_k1_MSFT_s003
                                 + 1.14189 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.07398 w_k1_ADM_s003
                                 + 1.66454 w_k1_AXP_s003
                                 + 1.47693 w_k1_DIS_s003
                                 + 1.40765 w_k1_FDX_s003
                                 + 0.892588 w_k1_IBM_s003
                                 + 1.26115 w_k1_INTC_s003
                                 + 1.10561 w_k1_KO_s003 + 1.18697 w_k1_MAT_s003
                                 + 1.22103 w_k1_MSFT_s003
                                 + 1.15955 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.852401 w_k1_ADM_s003
                                 + 0.941145 w_k1_AXP_s003
                                 + 0.759969 w_k1_DIS_s003
                                 + 0.8534 w_k1_FDX_s003
                                 + 0.598254 w_k1_IBM_s003
                                 + 0.88545 w_k1_INTC_s003
                                 + 0.760393 w_k1_KO_s003
                                 + 0.622082 w_k1_MAT_s003
                                 + 0.864816 w_k1_MSFT_s003
                                 + 0.819678 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.861104 w_k1_ADM_s003
                                 + 1.09328 w_k1_AXP_s003 + 1.0995 w_k1_DIS_s003
                                 + 1.08104 w_k1_FDX_s003
                                 + 0.684811 w_k1_IBM_s003
                                 + 1.01825 w_k1_INTC_s003
                                 + 0.925347 w_k1_KO_s003
                                 + 0.948547 w_k1_MAT_s003
                                 + 0.992697 w_k1_MSFT_s003
                                 + 0.980647 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.926989 w_k1_ADM_s003
                                 + 1.16759 w_k1_AXP_s003
                                 + 1.01858 w_k1_DIS_s003
                                 + 0.879303 w_k1_FDX_s003
                                 + 1.34068 w_k1_IBM_s003
                                 + 0.915679 w_k1_INTC_s003
                                 + 1.12471 w_k1_KO_s003 + 1.37203 w_k1_MAT_s003
                                 + 0.878511 w_k1_MSFT_s003
                                 + 1.06055 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.666885 w_k1_ADM_s003
                                 + 0.763528 w_k1_AXP_s003
                                 + 0.628849 w_k1_DIS_s003
                                 + 0.62209 w_k1_FDX_s003
                                 + 0.944433 w_k1_IBM_s003
                                 + 0.787973 w_k1_INTC_s003
                                 + 0.852533 w_k1_KO_s003
                                 + 1.06635 w_k1_MAT_s003
                                 + 0.725187 w_k1_MSFT_s003
                                 + 0.822669 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.01094 w_k1_ADM_s004
                                 + 0.843574 w_k1_AXP_s004
                                 + 1.06354 w_k1_DIS_s004
                                 + 1.12401 w_k1_FDX_s004
                                 + 0.600869 w_k1_IBM_s004
                                 + 1.04358 w_k1_INTC_s004
                                 + 1.00886 w_k1_KO_s004
                                 + 0.499644 w_k1_MAT_s004
                                 + 1.19598 w_k1_MSFT_s004
                                 + 1.05177 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.943253 w_k1_ADM_s004
                                 + 1.05106 w_k1_AXP_s004
                                 + 0.769111 w_k1_DIS_s004
                                 + 0.79852 w_k1_FDX_s004
                                 + 1.27042 w_k1_IBM_s004
                                 + 1.04314 w_k1_INTC_s004
                                 + 1.25512 w_k1_KO_s004 + 1.55835 w_k1_MAT_s004
                                 + 0.909012 w_k1_MSFT_s004
                                 + 0.896061 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.821396 w_k1_ADM_s004
                                 + 0.693898 w_k1_AXP_s004
                                 + 0.598333 w_k1_DIS_s004
                                 + 0.751616 w_k1_FDX_s004
                                 + 0.783997 w_k1_IBM_s004
                                 + 0.770251 w_k1_INTC_s004
                                 + 0.984231 w_k1_KO_s004 + 0.7889 w_k1_MAT_s004
                                 + 0.821639 w_k1_MSFT_s004
                                 + 0.81893 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.22489 w_k1_ADM_s004
                                 + 1.37579 w_k1_AXP_s004
                                 + 1.35318 w_k1_DIS_s004
                                 + 1.39494 w_k1_FDX_s004
                                 + 0.908659 w_k1_IBM_s004
                                 + 1.09445 w_k1_INTC_s004
                                 + 1.19412 w_k1_KO_s004 + 1.12161 w_k1_MAT_s004
                                 + 1.20564 w_k1_MSFT_s004 + 1.1934 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.884062 w_k1_ADM_s004
                                 + 1.08108 w_k1_AXP_s004
                                 + 1.02999 w_k1_DIS_s004
                                 + 1.08333 w_k1_FDX_s004
                                 + 0.90278 w_k1_IBM_s004
                                 + 0.932432 w_k1_INTC_s004
                                 + 1.0863 w_k1_KO_s004 + 1.14337 w_k1_MAT_s004
                                 + 1.03248 w_k1_MSFT_s004
                                 + 0.953545 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.31793 w_k1_ADM_s004
                                 + 1.51741 w_k1_AXP_s004
                                 + 1.59109 w_k1_DIS_s004
                                 + 1.31375 w_k1_FDX_s004 + 1.2848 w_k1_IBM_s004
                                 + 1.22282 w_k1_INTC_s004
                                 + 1.78256 w_k1_KO_s004 + 2.12217 w_k1_MAT_s004
                                 + 1.54938 w_k1_MSFT_s004
                                 + 1.33245 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.91592 w_k1_ADM_s004
                                 + 1.03199 w_k1_AXP_s004
                                 + 1.22444 w_k1_DIS_s004 + 1.025 w_k1_FDX_s004
                                 + 1.20899 w_k1_IBM_s004
                                 + 1.04142 w_k1_INTC_s004
                                 + 1.53192 w_k1_KO_s004 + 1.5425 w_k1_MAT_s004
                                 + 1.24712 w_k1_MSFT_s004
                                 + 1.20794 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.6951 w_k1_ADM_s004
                                 + 1.34262 w_k1_AXP_s004
                                 + 1.14432 w_k1_DIS_s004
                                 + 1.04565 w_k1_FDX_s004
                                 + 1.66921 w_k1_IBM_s004
                                 + 1.14863 w_k1_INTC_s004
                                 + 1.65907 w_k1_KO_s004 + 2.77116 w_k1_MAT_s004
                                 + 1.22572 w_k1_MSFT_s004
                                 + 1.12916 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.1748 w_k1_ADM_s005
                                 + 1.14955 w_k1_AXP_s005
                                 + 1.25246 w_k1_DIS_s005
                                 + 1.12318 w_k1_FDX_s005
                                 + 1.52387 w_k1_IBM_s005
                                 + 1.16152 w_k1_INTC_s005
                                 + 1.29454 w_k1_KO_s005 + 1.66951 w_k1_MAT_s005
                                 + 1.30161 w_k1_MSFT_s005
                                 + 1.24093 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.763229 w_k1_ADM_s005
                                 + 0.833457 w_k1_AXP_s005
                                 + 0.719184 w_k1_DIS_s005
                                 + 0.727175 w_k1_FDX_s005
                                 + 1.27804 w_k1_IBM_s005
                                 + 0.890098 w_k1_INTC_s005
                                 + 0.88909 w_k1_KO_s005
                                 + 0.925284 w_k1_MAT_s005
                                 + 0.937265 w_k1_MSFT_s005
                                 + 0.944887 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.825553 w_k1_ADM_s005
                                 + 0.643872 w_k1_AXP_s005
                                 + 0.64254 w_k1_DIS_s005
                                 + 0.806736 w_k1_FDX_s005
                                 + 0.773219 w_k1_IBM_s005
                                 + 0.904248 w_k1_INTC_s005
                                 + 0.695394 w_k1_KO_s005
                                 + 0.412073 w_k1_MAT_s005
                                 + 0.82354 w_k1_MSFT_s005
                                 + 0.890119 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.13361 w_k1_ADM_s005
                                 + 1.26215 w_k1_AXP_s005
                                 + 1.40196 w_k1_DIS_s005
                                 + 1.29458 w_k1_FDX_s005
                                 + 0.962694 w_k1_IBM_s005
                                 + 1.47537 w_k1_INTC_s005
                                 + 1.11792 w_k1_KO_s005
                                 + 0.739843 w_k1_MAT_s005
                                 + 1.26401 w_k1_MSFT_s005
                                 + 1.04505 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.25433 w_k1_ADM_s005
                                 + 1.23276 w_k1_AXP_s005
                                 + 0.924062 w_k1_DIS_s005
                                 + 1.07678 w_k1_FDX_s005
                                 + 1.11515 w_k1_IBM_s005
                                 + 1.02054 w_k1_INTC_s005
                                 + 0.925507 w_k1_KO_s005 + 1.0826 w_k1_MAT_s005
                                 + 1.00038 w_k1_MSFT_s005
                                 + 0.987602 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.09607 w_k1_ADM_s005
                                 + 0.970062 w_k1_AXP_s005
                                 + 0.949138 w_k1_DIS_s005
                                 + 0.948177 w_k1_FDX_s005
                                 + 0.995359 w_k1_IBM_s005
                                 + 0.957448 w_k1_INTC_s005
                                 + 0.791324 w_k1_KO_s005
                                 + 0.863666 w_k1_MAT_s005
                                 + 0.9035 w_k1_MSFT_s005 + 1.00992 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.868404 w_k1_ADM_s005
                                 + 0.706554 w_k1_AXP_s005
                                 + 0.949633 w_k1_DIS_s005
                                 + 0.946148 w_k1_FDX_s005
                                 + 0.656975 w_k1_IBM_s005
                                 + 1.02201 w_k1_INTC_s005
                                 + 0.912339 w_k1_KO_s005
                                 + 0.569415 w_k1_MAT_s005
                                 + 1.04605 w_k1_MSFT_s005
                                 + 0.866852 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.03596 w_k1_ADM_s005
                                 + 0.81439 w_k1_AXP_s005 + 1.1644 w_k1_DIS_s005
                                 + 0.945913 w_k1_FDX_s005
                                 + 1.2562 w_k1_IBM_s005
                                 + 0.841089 w_k1_INTC_s005
                                 + 1.10335 w_k1_KO_s005 + 0.90908 w_k1_MAT_s005
                                 + 1.10538 w_k1_MSFT_s005
                                 + 1.10868 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.810583 w_k1_ADM_s006
                                 + 0.726088 w_k1_AXP_s006
                                 + 0.600224 w_k1_DIS_s006
                                 + 0.823616 w_k1_FDX_s006
                                 + 0.673021 w_k1_IBM_s006
                                 + 0.784804 w_k1_INTC_s006
                                 + 0.765148 w_k1_KO_s006
                                 + 0.772756 w_k1_MAT_s006
                                 + 0.88829 w_k1_MSFT_s006
                                 + 0.814807 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.653428 w_k1_ADM_s006
                                 + 0.826395 w_k1_AXP_s006
                                 + 0.766513 w_k1_DIS_s006
                                 + 0.78077 w_k1_FDX_s006
                                 + 1.23379 w_k1_IBM_s006
                                 + 0.862471 w_k1_INTC_s006
                                 + 0.972031 w_k1_KO_s006
                                 + 0.995992 w_k1_MAT_s006
                                 + 0.79523 w_k1_MSFT_s006
                                 + 0.859097 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.22957 w_k1_ADM_s006
                                 + 1.16037 w_k1_AXP_s006
                                 + 1.33056 w_k1_DIS_s006
                                 + 1.27341 w_k1_FDX_s006
                                 + 0.827201 w_k1_IBM_s006
                                 + 1.24933 w_k1_INTC_s006
                                 + 1.14451 w_k1_KO_s006 + 1.00989 w_k1_MAT_s006
                                 + 1.36906 w_k1_MSFT_s006
                                 + 1.12325 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.896041 w_k1_ADM_s006
                                 + 0.921358 w_k1_AXP_s006
                                 + 1.11918 w_k1_DIS_s006
                                 + 1.06308 w_k1_FDX_s006
                                 + 0.854512 w_k1_IBM_s006
                                 + 0.991329 w_k1_INTC_s006
                                 + 1.05079 w_k1_KO_s006
                                 + 0.746707 w_k1_MAT_s006
                                 + 1.18356 w_k1_MSFT_s006
                                 + 1.11899 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.868441 w_k1_ADM_s006
                                 + 1.30099 w_k1_AXP_s006
                                 + 1.18854 w_k1_DIS_s006
                                 + 1.12194 w_k1_FDX_s006
                                 + 1.37749 w_k1_IBM_s006
                                 + 0.988233 w_k1_INTC_s006
                                 + 1.17127 w_k1_KO_s006 + 1.60886 w_k1_MAT_s006
                                 + 1.02348 w_k1_MSFT_s006
                                 + 1.10103 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.521367 w_k1_ADM_s006
                                 + 0.664177 w_k1_AXP_s006
                                 + 0.852051 w_k1_DIS_s006
                                 + 0.82484 w_k1_FDX_s006
                                 + 0.919057 w_k1_IBM_s006
                                 + 0.98565 w_k1_INTC_s006 + 1.0207 w_k1_KO_s006
                                 + 0.688851 w_k1_MAT_s006
                                 + 0.962863 w_k1_MSFT_s006
                                 + 0.943777 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.31287 w_k1_ADM_s006
                                 + 1.47937 w_k1_AXP_s006
                                 + 1.76941 w_k1_DIS_s006
                                 + 1.47683 w_k1_FDX_s006
                                 + 1.10207 w_k1_IBM_s006
                                 + 1.06756 w_k1_INTC_s006
                                 + 1.38576 w_k1_KO_s006 + 1.29445 w_k1_MAT_s006
                                 + 1.50588 w_k1_MSFT_s006
                                 + 1.23902 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.808305 w_k1_ADM_s006
                                 + 1.14494 w_k1_AXP_s006
                                 + 0.84675 w_k1_DIS_s006
                                 + 1.01343 w_k1_FDX_s006
                                 + 0.827124 w_k1_IBM_s006
                                 + 0.910492 w_k1_INTC_s006
                                 + 0.899197 w_k1_KO_s006
                                 + 1.04529 w_k1_MAT_s006
                                 + 0.983712 w_k1_MSFT_s006
                                 + 0.892917 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.07798 w_k1_ADM_s007
                                 + 1.24207 w_k1_AXP_s007
                                 + 1.37626 w_k1_DIS_s007
                                 + 1.12129 w_k1_FDX_s007
                                 + 1.87064 w_k1_IBM_s007
                                 + 1.41403 w_k1_INTC_s007
                                 + 1.32426 w_k1_KO_s007 + 1.23926 w_k1_MAT_s007
                                 + 1.21752 w_k1_MSFT_s007
                                 + 1.11142 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.843086 w_k1_ADM_s007
                                 + 1.20308 w_k1_AXP_s007
                                 + 1.08289 w_k1_DIS_s007
                                 + 1.00197 w_k1_FDX_s007
                                 + 1.27421 w_k1_IBM_s007
                                 + 1.21476 w_k1_INTC_s007
                                 + 1.03425 w_k1_KO_s007 + 1.39671 w_k1_MAT_s007
                                 + 1.03266 w_k1_MSFT_s007
                                 + 0.942345 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.14931 w_k1_ADM_s007
                                 + 1.45659 w_k1_AXP_s007
                                 + 1.78072 w_k1_DIS_s007 + 1.3892 w_k1_FDX_s007
                                 + 1.27199 w_k1_IBM_s007
                                 + 1.48465 w_k1_INTC_s007
                                 + 1.47296 w_k1_KO_s007 + 1.37305 w_k1_MAT_s007
                                 + 1.42587 w_k1_MSFT_s007 + 1.2932 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.767527 w_k1_ADM_s007
                                 + 0.724528 w_k1_AXP_s007
                                 + 0.740253 w_k1_DIS_s007
                                 + 0.753569 w_k1_FDX_s007
                                 + 0.962512 w_k1_IBM_s007
                                 + 0.941898 w_k1_INTC_s007
                                 + 0.80135 w_k1_KO_s007
                                 + 0.671275 w_k1_MAT_s007
                                 + 0.78721 w_k1_MSFT_s007
                                 + 0.767771 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1445 w_k1_ADM_s007
                                 + 0.953459 w_k1_AXP_s007
                                 + 1.29067 w_k1_DIS_s007
                                 + 1.23126 w_k1_FDX_s007
                                 + 0.769104 w_k1_IBM_s007
                                 + 1.35943 w_k1_INTC_s007
                                 + 0.990963 w_k1_KO_s007
                                 + 0.632633 w_k1_MAT_s007
                                 + 1.28003 w_k1_MSFT_s007
                                 + 0.981005 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.842066 w_k1_ADM_s007
                                 + 0.860601 w_k1_AXP_s007
                                 + 1.03994 w_k1_DIS_s007
                                 + 1.03001 w_k1_FDX_s007
                                 + 0.697535 w_k1_IBM_s007
                                 + 0.957638 w_k1_INTC_s007
                                 + 0.940952 w_k1_KO_s007
                                 + 0.648643 w_k1_MAT_s007
                                 + 1.17383 w_k1_MSFT_s007
                                 + 0.909839 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09506 w_k1_ADM_s007
                                 + 1.27438 w_k1_AXP_s007
                                 + 1.32498 w_k1_DIS_s007
                                 + 1.23821 w_k1_FDX_s007
                                 + 0.74468 w_k1_IBM_s007
                                 + 0.991869 w_k1_INTC_s007
                                 + 0.915124 w_k1_KO_s007
                                 + 0.860537 w_k1_MAT_s007
                                 + 0.981943 w_k1_MSFT_s007
                                 + 1.09482 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.844267 w_k1_ADM_s007
                                 + 0.816831 w_k1_AXP_s007
                                 + 1.05452 w_k1_DIS_s007
                                 + 0.914702 w_k1_FDX_s007
                                 + 0.926472 w_k1_IBM_s007
                                 + 0.979787 w_k1_INTC_s007
                                 + 1.01628 w_k1_KO_s007 + 0.77943 w_k1_MAT_s007
                                 + 0.937835 w_k1_MSFT_s007
                                 + 1.06151 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.969288 w_k1_ADM_s010
                                 + 0.988102 w_k1_AXP_s010
                                 + 0.924901 w_k1_DIS_s010
                                 + 0.974699 w_k1_FDX_s010
                                 + 1.15479 w_k1_IBM_s010
                                 + 1.04794 w_k1_INTC_s010
                                 + 1.11673 w_k1_KO_s010 + 1.29198 w_k1_MAT_s010
                                 + 0.856367 w_k1_MSFT_s010
                                 + 0.975013 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.30177 w_k1_ADM_s010
                                 + 1.59744 w_k1_AXP_s010
                                 + 1.86929 w_k1_DIS_s010
                                 + 1.47088 w_k1_FDX_s010
                                 + 1.71581 w_k1_IBM_s010
                                 + 1.33526 w_k1_INTC_s010 + 1.4456 w_k1_KO_s010
                                 + 1.82893 w_k1_MAT_s010
                                 + 1.3221 w_k1_MSFT_s010 + 1.342 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.01533 w_k1_ADM_s010
                                 + 0.845816 w_k1_AXP_s010
                                 + 1.18641 w_k1_DIS_s010
                                 + 0.984514 w_k1_FDX_s010
                                 + 1.43931 w_k1_IBM_s010
                                 + 1.09956 w_k1_INTC_s010
                                 + 1.25608 w_k1_KO_s010 + 1.51507 w_k1_MAT_s010
                                 + 1.12106 w_k1_MSFT_s010
                                 + 0.94456 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03286 w_k1_ADM_s010
                                 + 0.940946 w_k1_AXP_s010
                                 + 0.771214 w_k1_DIS_s010
                                 + 0.961244 w_k1_FDX_s010
                                 + 0.73644 w_k1_IBM_s010
                                 + 1.22793 w_k1_INTC_s010
                                 + 0.857366 w_k1_KO_s010
                                 + 1.06505 w_k1_MAT_s010
                                 + 0.910754 w_k1_MSFT_s010
                                 + 0.81745 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.31906 w_k1_ADM_s010
                                 + 1.07965 w_k1_AXP_s010
                                 + 1.52624 w_k1_DIS_s010
                                 + 1.31081 w_k1_FDX_s010
                                 + 0.863948 w_k1_IBM_s010
                                 + 1.57867 w_k1_INTC_s010
                                 + 1.23883 w_k1_KO_s010 + 1.18945 w_k1_MAT_s010
                                 + 1.35022 w_k1_MSFT_s010
                                 + 1.18968 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.22655 w_k1_ADM_s010
                                 + 1.33883 w_k1_AXP_s010
                                 + 1.41564 w_k1_DIS_s010
                                 + 1.13409 w_k1_FDX_s010
                                 + 1.45989 w_k1_IBM_s010
                                 + 1.40743 w_k1_INTC_s010
                                 + 1.44443 w_k1_KO_s010 + 1.73328 w_k1_MAT_s010
                                 + 1.20345 w_k1_MSFT_s010
                                 + 1.18053 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.878322 w_k1_ADM_s010
                                 + 0.609845 w_k1_AXP_s010
                                 + 0.735773 w_k1_DIS_s010
                                 + 0.864115 w_k1_FDX_s010
                                 + 0.742237 w_k1_IBM_s010
                                 + 1.01308 w_k1_INTC_s010
                                 + 0.764681 w_k1_KO_s010
                                 + 0.732772 w_k1_MAT_s010
                                 + 0.99158 w_k1_MSFT_s010
                                 + 0.820143 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.01892 w_k1_ADM_s010
                                 + 1.05954 w_k1_AXP_s010
                                 + 1.11203 w_k1_DIS_s010
                                 + 1.12507 w_k1_FDX_s010
                                 + 0.73187 w_k1_IBM_s010
                                 + 1.30919 w_k1_INTC_s010
                                 + 0.924568 w_k1_KO_s010
                                 + 0.944168 w_k1_MAT_s010
                                 + 1.11305 w_k1_MSFT_s010
                                 + 0.993995 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.27529 w_k1_ADM_s011
                                 + 1.01556 w_k1_AXP_s011
                                 + 1.18356 w_k1_DIS_s011
                                 + 1.11217 w_k1_FDX_s011
                                 + 0.947385 w_k1_IBM_s011
                                 + 1.34826 w_k1_INTC_s011
                                 + 1.28605 w_k1_KO_s011
                                 + 0.945682 w_k1_MAT_s011
                                 + 1.23094 w_k1_MSFT_s011
                                 + 1.16855 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.886708 w_k1_ADM_s011
                                 + 1.1621 w_k1_AXP_s011 + 1.00144 w_k1_DIS_s011
                                 + 0.949368 w_k1_FDX_s011
                                 + 1.31472 w_k1_IBM_s011
                                 + 1.43978 w_k1_INTC_s011
                                 + 1.06415 w_k1_KO_s011 + 1.38336 w_k1_MAT_s011
                                 + 1.12981 w_k1_MSFT_s011
                                 + 1.05593 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.936582 w_k1_ADM_s011
                                 + 0.756767 w_k1_AXP_s011
                                 + 0.738162 w_k1_DIS_s011
                                 + 0.894338 w_k1_FDX_s011
                                 + 0.671646 w_k1_IBM_s011
                                 + 1.17523 w_k1_INTC_s011
                                 + 0.725561 w_k1_KO_s011
                                 + 0.465194 w_k1_MAT_s011
                                 + 0.897241 w_k1_MSFT_s011
                                 + 0.813401 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.864438 w_k1_ADM_s011
                                 + 1.1304 w_k1_AXP_s011 + 1.71428 w_k1_DIS_s011
                                 + 1.13672 w_k1_FDX_s011 + 1.4881 w_k1_IBM_s011
                                 + 1.60426 w_k1_INTC_s011
                                 + 1.62747 w_k1_KO_s011 + 1.74164 w_k1_MAT_s011
                                 + 1.28909 w_k1_MSFT_s011
                                 + 1.24188 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.672085 w_k1_ADM_s011
                                 + 0.646552 w_k1_AXP_s011
                                 + 0.457399 w_k1_DIS_s011 + 0.653 w_k1_FDX_s011
                                 + 0.730455 w_k1_IBM_s011
                                 + 0.873225 w_k1_INTC_s011
                                 + 0.633351 w_k1_KO_s011
                                 + 0.491856 w_k1_MAT_s011
                                 + 0.657395 w_k1_MSFT_s011
                                 + 0.823029 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.33649 w_k1_ADM_s011
                                 + 1.4883 w_k1_AXP_s011 + 1.47586 w_k1_DIS_s011
                                 + 1.45018 w_k1_FDX_s011
                                 + 1.13685 w_k1_IBM_s011
                                 + 1.52623 w_k1_INTC_s011
                                 + 1.40646 w_k1_KO_s011 + 1.11518 w_k1_MAT_s011
                                 + 1.44498 w_k1_MSFT_s011
                                 + 1.44106 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.851479 w_k1_ADM_s011
                                 + 0.880596 w_k1_AXP_s011
                                 + 0.742787 w_k1_DIS_s011
                                 + 0.75055 w_k1_FDX_s011
                                 + 1.03278 w_k1_IBM_s011
                                 + 1.08226 w_k1_INTC_s011
                                 + 0.987592 w_k1_KO_s011
                                 + 0.963925 w_k1_MAT_s011
                                 + 0.84427 w_k1_MSFT_s011
                                 + 0.982518 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.954633 w_k1_ADM_s011
                                 + 0.763365 w_k1_AXP_s011
                                 + 0.967459 w_k1_DIS_s011
                                 + 0.91453 w_k1_FDX_s011
                                 + 0.850921 w_k1_IBM_s011
                                 + 1.17483 w_k1_INTC_s011
                                 + 0.943306 w_k1_KO_s011
                                 + 0.615891 w_k1_MAT_s011
                                 + 1.07528 w_k1_MSFT_s011
                                 + 0.959888 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.71747 w_k1_ADM_s012
                                 + 0.748888 w_k1_AXP_s012
                                 + 0.699847 w_k1_DIS_s012
                                 + 0.776881 w_k1_FDX_s012
                                 + 1.09688 w_k1_IBM_s012
                                 + 0.556614 w_k1_INTC_s012
                                 + 0.781759 w_k1_KO_s012
                                 + 1.21469 w_k1_MAT_s012
                                 + 0.721979 w_k1_MSFT_s012
                                 + 0.830518 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.997471 w_k1_ADM_s012
                                 + 1.37104 w_k1_AXP_s012
                                 + 1.58936 w_k1_DIS_s012
                                 + 1.34812 w_k1_FDX_s012
                                 + 1.45324 w_k1_IBM_s012
                                 + 1.15055 w_k1_INTC_s012
                                 + 1.28059 w_k1_KO_s012 + 1.37941 w_k1_MAT_s012
                                 + 1.43067 w_k1_MSFT_s012
                                 + 1.25237 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.966381 w_k1_ADM_s012
                                 + 1.07841 w_k1_AXP_s012
                                 + 1.13823 w_k1_DIS_s012
                                 + 0.985771 w_k1_FDX_s012
                                 + 1.57196 w_k1_IBM_s012
                                 + 0.907589 w_k1_INTC_s012
                                 + 1.24386 w_k1_KO_s012 + 1.68406 w_k1_MAT_s012
                                 + 1.07489 w_k1_MSFT_s012 + 1.1764 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.17117 w_k1_ADM_s012
                                 + 1.54243 w_k1_AXP_s012
                                 + 1.06414 w_k1_DIS_s012
                                 + 1.19529 w_k1_FDX_s012
                                 + 0.978041 w_k1_IBM_s012
                                 + 0.86079 w_k1_INTC_s012
                                 + 0.873653 w_k1_KO_s012
                                 + 1.24361 w_k1_MAT_s012
                                 + 0.99213 w_k1_MSFT_s012
                                 + 1.19994 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.1317 w_k1_ADM_s012
                                 + 1.20617 w_k1_AXP_s012
                                 + 1.40185 w_k1_DIS_s012
                                 + 1.23452 w_k1_FDX_s012
                                 + 0.944054 w_k1_IBM_s012
                                 + 1.1051 w_k1_INTC_s012 + 1.15625 w_k1_KO_s012
                                 + 0.95758 w_k1_MAT_s012
                                 + 1.18334 w_k1_MSFT_s012
                                 + 1.07096 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.930026 w_k1_ADM_s012
                                 + 0.856323 w_k1_AXP_s012
                                 + 1.05119 w_k1_DIS_s012
                                 + 1.00471 w_k1_FDX_s012
                                 + 1.02696 w_k1_IBM_s012
                                 + 0.898481 w_k1_INTC_s012
                                 + 0.988439 w_k1_KO_s012
                                 + 0.884954 w_k1_MAT_s012
                                 + 1.10548 w_k1_MSFT_s012
                                 + 1.02895 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.886565 w_k1_ADM_s012
                                 + 1.01448 w_k1_AXP_s012
                                 + 0.740879 w_k1_DIS_s012
                                 + 0.936706 w_k1_FDX_s012
                                 + 0.901981 w_k1_IBM_s012
                                 + 1.04857 w_k1_INTC_s012
                                 + 0.921805 w_k1_KO_s012
                                 + 0.935075 w_k1_MAT_s012
                                 + 0.878384 w_k1_MSFT_s012
                                 + 0.904414 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.632663 w_k1_ADM_s012
                                 + 0.713571 w_k1_AXP_s012
                                 + 0.576722 w_k1_DIS_s012
                                 + 0.754792 w_k1_FDX_s012
                                 + 0.65347 w_k1_IBM_s012
                                 + 0.675098 w_k1_INTC_s012
                                 + 0.652472 w_k1_KO_s012
                                 + 0.665525 w_k1_MAT_s012
                                 + 0.736208 w_k1_MSFT_s012
                                 + 0.818284 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.799965 w_k1_ADM_s013
                                 + 0.687343 w_k1_AXP_s013
                                 + 0.635994 w_k1_DIS_s013
                                 + 0.747752 w_k1_FDX_s013
                                 + 0.708648 w_k1_IBM_s013
                                 + 0.80382 w_k1_INTC_s013
                                 + 0.705726 w_k1_KO_s013
                                 + 0.834125 w_k1_MAT_s013
                                 + 0.705351 w_k1_MSFT_s013
                                 + 0.874451 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.951185 w_k1_ADM_s013
                                 + 0.960817 w_k1_AXP_s013
                                 + 0.995797 w_k1_DIS_s013
                                 + 1.09701 w_k1_FDX_s013
                                 + 0.698883 w_k1_IBM_s013
                                 + 0.78093 w_k1_INTC_s013
                                 + 0.761686 w_k1_KO_s013 + 0.7542 w_k1_MAT_s013
                                 + 0.86088 w_k1_MSFT_s013
                                 + 0.862955 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.695869 w_k1_ADM_s013
                                 + 0.87534 w_k1_AXP_s013
                                 + 1.01833 w_k1_DIS_s013
                                 + 0.884521 w_k1_FDX_s013
                                 + 1.48695 w_k1_IBM_s013
                                 + 0.975178 w_k1_INTC_s013
                                 + 1.31408 w_k1_KO_s013 + 1.4591 w_k1_MAT_s013
                                 + 0.961412 w_k1_MSFT_s013
                                 + 1.01257 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.03073 w_k1_ADM_s013
                                 + 0.88384 w_k1_AXP_s013
                                 + 1.23593 w_k1_DIS_s013
                                 + 1.04487 w_k1_FDX_s013
                                 + 1.16208 w_k1_IBM_s013
                                 + 1.06795 w_k1_INTC_s013
                                 + 1.23437 w_k1_KO_s013 + 1.05564 w_k1_MAT_s013
                                 + 1.06588 w_k1_MSFT_s013 + 1.101 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.07927 w_k1_ADM_s013
                                 + 1.18082 w_k1_AXP_s013
                                 + 1.18199 w_k1_DIS_s013 + 1.1353 w_k1_FDX_s013
                                 + 1.252 w_k1_IBM_s013
                                 + 0.982119 w_k1_INTC_s013
                                 + 1.12869 w_k1_KO_s013 + 1.17979 w_k1_MAT_s013
                                 + 1.04528 w_k1_MSFT_s013
                                 + 1.08545 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.770536 w_k1_ADM_s013
                                 + 0.76466 w_k1_AXP_s013
                                 + 0.80668 w_k1_DIS_s013
                                 + 0.954127 w_k1_FDX_s013
                                 + 0.877107 w_k1_IBM_s013
                                 + 0.911578 w_k1_INTC_s013
                                 + 0.93851 w_k1_KO_s013
                                 + 0.923488 w_k1_MAT_s013
                                 + 0.800603 w_k1_MSFT_s013
                                 + 0.90715 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.694584 w_k1_ADM_s013
                                 + 0.910636 w_k1_AXP_s013
                                 + 0.781441 w_k1_DIS_s013
                                 + 0.810789 w_k1_FDX_s013
                                 + 1.10363 w_k1_IBM_s013
                                 + 0.762739 w_k1_INTC_s013
                                 + 0.900384 w_k1_KO_s013
                                 + 1.01232 w_k1_MAT_s013
                                 + 0.782818 w_k1_MSFT_s013
                                 + 0.896897 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.31642 w_k1_ADM_s013
                                 + 1.51918 w_k1_AXP_s013
                                 + 1.55431 w_k1_DIS_s013
                                 + 1.48434 w_k1_FDX_s013
                                 + 1.32247 w_k1_IBM_s013
                                 + 1.31962 w_k1_INTC_s013
                                 + 1.12201 w_k1_KO_s013
                                 + 0.942143 w_k1_MAT_s013
                                 + 1.1318 w_k1_MSFT_s013 + 1.04606 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.89049 w_k1_ADM_s014
                                 + 0.687039 w_k1_AXP_s014
                                 + 0.919336 w_k1_DIS_s014
                                 + 0.927522 w_k1_FDX_s014
                                 + 0.687254 w_k1_IBM_s014
                                 + 0.923401 w_k1_INTC_s014
                                 + 0.76019 w_k1_KO_s014
                                 + 0.628547 w_k1_MAT_s014
                                 + 0.921928 w_k1_MSFT_s014
                                 + 0.855619 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.697824 w_k1_ADM_s014
                                 + 0.628269 w_k1_AXP_s014
                                 + 0.633906 w_k1_DIS_s014
                                 + 0.710606 w_k1_FDX_s014
                                 + 0.903205 w_k1_IBM_s014
                                 + 0.682191 w_k1_INTC_s014
                                 + 0.748042 w_k1_KO_s014
                                 + 0.88711 w_k1_MAT_s014
                                 + 0.700868 w_k1_MSFT_s014
                                 + 0.752633 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.33079 w_k1_ADM_s014
                                 + 1.13485 w_k1_AXP_s014
                                 + 1.26372 w_k1_DIS_s014
                                 + 1.21676 w_k1_FDX_s014
                                 + 0.891581 w_k1_IBM_s014
                                 + 0.983765 w_k1_INTC_s014
                                 + 0.982271 w_k1_KO_s014
                                 + 1.00807 w_k1_MAT_s014
                                 + 1.10485 w_k1_MSFT_s014
                                 + 1.09117 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.943558 w_k1_ADM_s014
                                 + 0.819992 w_k1_AXP_s014
                                 + 0.877533 w_k1_DIS_s014
                                 + 0.874095 w_k1_FDX_s014
                                 + 0.957671 w_k1_IBM_s014
                                 + 0.840755 w_k1_INTC_s014
                                 + 0.891403 w_k1_KO_s014
                                 + 1.15928 w_k1_MAT_s014
                                 + 0.902474 w_k1_MSFT_s014
                                 + 0.86617 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.03493 w_k1_ADM_s014
                                 + 0.927717 w_k1_AXP_s014
                                 + 1.17241 w_k1_DIS_s014
                                 + 0.962385 w_k1_FDX_s014
                                 + 1.47328 w_k1_IBM_s014
                                 + 0.96059 w_k1_INTC_s014
                                 + 1.16654 w_k1_KO_s014 + 1.15926 w_k1_MAT_s014
                                 + 0.965634 w_k1_MSFT_s014
                                 + 1.01729 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.39491 w_k1_ADM_s014
                                 + 0.963562 w_k1_AXP_s014
                                 + 1.94454 w_k1_DIS_s014
                                 + 1.39122 w_k1_FDX_s014
                                 + 0.968881 w_k1_IBM_s014
                                 + 1.37269 w_k1_INTC_s014
                                 + 1.32142 w_k1_KO_s014
                                 + 0.872599 w_k1_MAT_s014
                                 + 1.49679 w_k1_MSFT_s014
                                 + 1.15237 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.918251 w_k1_ADM_s014
                                 + 0.844332 w_k1_AXP_s014
                                 + 0.835353 w_k1_DIS_s014
                                 + 0.962592 w_k1_FDX_s014
                                 + 0.718548 w_k1_IBM_s014
                                 + 0.80614 w_k1_INTC_s014
                                 + 0.741224 w_k1_KO_s014
                                 + 0.716982 w_k1_MAT_s014
                                 + 0.799607 w_k1_MSFT_s014
                                 + 0.817849 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.10132 w_k1_ADM_s014
                                 + 1.14437 w_k1_AXP_s014
                                 + 1.17315 w_k1_DIS_s014
                                 + 1.14856 w_k1_FDX_s014 + 1.3157 w_k1_IBM_s014
                                 + 0.997921 w_k1_INTC_s014
                                 + 1.09267 w_k1_KO_s014 + 1.42557 w_k1_MAT_s014
                                 + 1.12212 w_k1_MSFT_s014
                                 + 1.00163 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.01945 w_k1_ADM_s015
                                 + 1.14036 w_k1_AXP_s015
                                 + 1.04299 w_k1_DIS_s015
                                 + 1.01379 w_k1_FDX_s015 + 1.5138 w_k1_IBM_s015
                                 + 0.988299 w_k1_INTC_s015
                                 + 1.06022 w_k1_KO_s015 + 1.45233 w_k1_MAT_s015
                                 + 0.823119 w_k1_MSFT_s015
                                 + 0.924471 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.23857 w_k1_ADM_s015
                                 + 1.06132 w_k1_AXP_s015
                                 + 1.33287 w_k1_DIS_s015 + 1.2311 w_k1_FDX_s015
                                 + 1.14949 w_k1_IBM_s015
                                 + 1.08452 w_k1_INTC_s015
                                 + 1.14884 w_k1_KO_s015 + 1.1769 w_k1_MAT_s015
                                 + 1.00458 w_k1_MSFT_s015
                                 + 1.01546 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.2904 w_k1_ADM_s015
                                 + 1.1123 w_k1_AXP_s015 + 1.32535 w_k1_DIS_s015
                                 + 1.17949 w_k1_FDX_s015
                                 + 0.861436 w_k1_IBM_s015
                                 + 1.11692 w_k1_INTC_s015
                                 + 0.924831 w_k1_KO_s015
                                 + 0.916543 w_k1_MAT_s015
                                 + 1.09519 w_k1_MSFT_s015
                                 + 0.988932 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.18594 w_k1_ADM_s015
                                 + 1.23916 w_k1_AXP_s015
                                 + 0.984478 w_k1_DIS_s015
                                 + 1.19639 w_k1_FDX_s015
                                 + 0.909212 w_k1_IBM_s015
                                 + 1.09492 w_k1_INTC_s015
                                 + 0.867759 w_k1_KO_s015
                                 + 1.23437 w_k1_MAT_s015
                                 + 0.960907 w_k1_MSFT_s015
                                 + 1.00106 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.951389 w_k1_ADM_s015
                                 + 0.765984 w_k1_AXP_s015
                                 + 0.866433 w_k1_DIS_s015
                                 + 0.85281 w_k1_FDX_s015
                                 + 0.963676 w_k1_IBM_s015
                                 + 0.813396 w_k1_INTC_s015
                                 + 0.921543 w_k1_KO_s015
                                 + 1.12136 w_k1_MAT_s015
                                 + 0.83501 w_k1_MSFT_s015
                                 + 0.814167 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.730781 w_k1_ADM_s015
                                 + 0.831016 w_k1_AXP_s015
                                 + 0.721739 w_k1_DIS_s015
                                 + 0.913307 w_k1_FDX_s015
                                 + 0.694161 w_k1_IBM_s015
                                 + 0.959688 w_k1_INTC_s015
                                 + 0.68133 w_k1_KO_s015
                                 + 0.877097 w_k1_MAT_s015
                                 + 0.817073 w_k1_MSFT_s015
                                 + 0.843427 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.1353 w_k1_ADM_s015
                                 + 1.43141 w_k1_AXP_s015
                                 + 1.64845 w_k1_DIS_s015
                                 + 1.39233 w_k1_FDX_s015
                                 + 1.31166 w_k1_IBM_s015
                                 + 1.38496 w_k1_INTC_s015
                                 + 1.36478 w_k1_KO_s015 + 1.63043 w_k1_MAT_s015
                                 + 1.23592 w_k1_MSFT_s015
                                 + 1.07886 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.01886 w_k1_ADM_s015
                                 + 0.686245 w_k1_AXP_s015
                                 + 1.14554 w_k1_DIS_s015 + 1.0523 w_k1_FDX_s015
                                 + 1.06558 w_k1_IBM_s015
                                 + 0.946269 w_k1_INTC_s015
                                 + 1.14894 w_k1_KO_s015
                                 + 0.875015 w_k1_MAT_s015
                                 + 0.96895 w_k1_MSFT_s015
                                 + 0.923471 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.12134 w_k1_ADM_s016
                                 + 1.57859 w_k1_AXP_s016
                                 + 1.12141 w_k1_DIS_s016 + 1.1097 w_k1_FDX_s016
                                 + 1.0808 w_k1_IBM_s016
                                 + 1.27076 w_k1_INTC_s016
                                 + 0.993609 w_k1_KO_s016
                                 + 1.34921 w_k1_MAT_s016
                                 + 1.07282 w_k1_MSFT_s016
                                 + 1.13512 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.641103 w_k1_ADM_s016
                                 + 0.89377 w_k1_AXP_s016
                                 + 0.740683 w_k1_DIS_s016
                                 + 0.746614 w_k1_FDX_s016
                                 + 0.966039 w_k1_IBM_s016
                                 + 0.867313 w_k1_INTC_s016
                                 + 0.899206 w_k1_KO_s016
                                 + 1.26774 w_k1_MAT_s016
                                 + 0.816566 w_k1_MSFT_s016
                                 + 0.834094 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.771395 w_k1_ADM_s016
                                 + 0.878532 w_k1_AXP_s016 + 1.022 w_k1_DIS_s016
                                 + 0.825321 w_k1_FDX_s016
                                 + 1.5394 w_k1_IBM_s016
                                 + 1.03025 w_k1_INTC_s016 + 1.2576 w_k1_KO_s016
                                 + 1.63347 w_k1_MAT_s016
                                 + 0.988021 w_k1_MSFT_s016
                                 + 1.02252 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.709301 w_k1_ADM_s016
                                 + 0.724402 w_k1_AXP_s016
                                 + 0.733266 w_k1_DIS_s016
                                 + 0.858621 w_k1_FDX_s016
                                 + 0.803605 w_k1_IBM_s016
                                 + 0.979919 w_k1_INTC_s016
                                 + 0.782274 w_k1_KO_s016
                                 + 0.664056 w_k1_MAT_s016
                                 + 0.867654 w_k1_MSFT_s016
                                 + 0.884818 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.03473 w_k1_ADM_s016
                                 + 1.23184 w_k1_AXP_s016
                                 + 1.49291 w_k1_DIS_s016
                                 + 1.18099 w_k1_FDX_s016
                                 + 1.31396 w_k1_IBM_s016
                                 + 1.12905 w_k1_INTC_s016
                                 + 1.10304 w_k1_KO_s016 + 1.30634 w_k1_MAT_s016
                                 + 1.12698 w_k1_MSFT_s016
                                 + 0.956716 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.806741 w_k1_ADM_s016
                                 + 0.908132 w_k1_AXP_s016
                                 + 1.26403 w_k1_DIS_s016 + 1.0593 w_k1_FDX_s016
                                 + 0.973821 w_k1_IBM_s016
                                 + 1.04956 w_k1_INTC_s016
                                 + 0.966696 w_k1_KO_s016
                                 + 0.655204 w_k1_MAT_s016
                                 + 1.10781 w_k1_MSFT_s016
                                 + 0.948132 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.726721 w_k1_ADM_s016
                                 + 1.01854 w_k1_AXP_s016
                                 + 0.926216 w_k1_DIS_s016
                                 + 0.936006 w_k1_FDX_s016
                                 + 1.13552 w_k1_IBM_s016
                                 + 1.0622 w_k1_INTC_s016
                                 + 0.981647 w_k1_KO_s016
                                 + 0.946688 w_k1_MAT_s016
                                 + 0.99877 w_k1_MSFT_s016
                                 + 0.958957 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.948191 w_k1_ADM_s016
                                 + 1.15124 w_k1_AXP_s016
                                 + 1.19454 w_k1_DIS_s016
                                 + 1.18222 w_k1_FDX_s016
                                 + 0.767423 w_k1_IBM_s016
                                 + 1.1138 w_k1_INTC_s016
                                 + 0.859327 w_k1_KO_s016
                                 + 0.776365 w_k1_MAT_s016
                                 + 1.355 w_k1_MSFT_s016 + 0.932897 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.807719 w_k1_ADM_s017
                                 + 0.687872 w_k1_AXP_s017
                                 + 0.742477 w_k1_DIS_s017
                                 + 0.868578 w_k1_FDX_s017
                                 + 0.685547 w_k1_IBM_s017
                                 + 0.929245 w_k1_INTC_s017
                                 + 0.68366 w_k1_KO_s017
                                 + 0.453827 w_k1_MAT_s017
                                 + 0.882541 w_k1_MSFT_s017
                                 + 0.876047 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.02985 w_k1_ADM_s017
                                 + 0.623567 w_k1_AXP_s017
                                 + 0.54187 w_k1_DIS_s017
                                 + 0.690927 w_k1_FDX_s017
                                 + 0.831616 w_k1_IBM_s017
                                 + 0.875176 w_k1_INTC_s017
                                 + 0.680163 w_k1_KO_s017
                                 + 0.627146 w_k1_MAT_s017
                                 + 0.80259 w_k1_MSFT_s017
                                 + 0.879783 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.0184 w_k1_ADM_s017
                                 + 1.05214 w_k1_AXP_s017 + 1.0654 w_k1_DIS_s017
                                 + 1.06412 w_k1_FDX_s017
                                 + 0.83583 w_k1_IBM_s017
                                 + 0.9906 w_k1_INTC_s017 + 0.9278 w_k1_KO_s017
                                 + 0.854434 w_k1_MAT_s017
                                 + 0.942485 w_k1_MSFT_s017
                                 + 1.02474 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.90107 w_k1_ADM_s017
                                 + 1.61869 w_k1_AXP_s017
                                 + 1.83852 w_k1_DIS_s017
                                 + 1.41245 w_k1_FDX_s017 + 1.2571 w_k1_IBM_s017
                                 + 1.30665 w_k1_INTC_s017
                                 + 1.36563 w_k1_KO_s017
                                 + 0.990567 w_k1_MAT_s017
                                 + 1.33284 w_k1_MSFT_s017
                                 + 1.38309 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.20101 w_k1_ADM_s017
                                 + 1.05057 w_k1_AXP_s017
                                 + 1.29962 w_k1_DIS_s017
                                 + 1.12745 w_k1_FDX_s017
                                 + 1.07831 w_k1_IBM_s017
                                 + 1.25325 w_k1_INTC_s017
                                 + 1.09231 w_k1_KO_s017
                                 + 0.773471 w_k1_MAT_s017
                                 + 1.15236 w_k1_MSFT_s017
                                 + 1.10565 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.905561 w_k1_ADM_s017
                                 + 1.05177 w_k1_AXP_s017
                                 + 0.86717 w_k1_DIS_s017
                                 + 0.86914 w_k1_FDX_s017
                                 + 1.55829 w_k1_IBM_s017
                                 + 0.957443 w_k1_INTC_s017
                                 + 0.962732 w_k1_KO_s017
                                 + 1.40736 w_k1_MAT_s017
                                 + 0.774786 w_k1_MSFT_s017
                                 + 1.13181 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.07105 w_k1_ADM_s017
                                 + 0.773139 w_k1_AXP_s017
                                 + 0.718614 w_k1_DIS_s017
                                 + 0.808081 w_k1_FDX_s017
                                 + 1.07705 w_k1_IBM_s017
                                 + 1.02724 w_k1_INTC_s017
                                 + 1.04259 w_k1_KO_s017
                                 + 0.909283 w_k1_MAT_s017
                                 + 0.89867 w_k1_MSFT_s017
                                 + 1.06096 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.982032 w_k1_ADM_s017
                                 + 1.3022 w_k1_AXP_s017 + 1.0281 w_k1_DIS_s017
                                 + 1.05705 w_k1_FDX_s017
                                 + 1.20335 w_k1_IBM_s017
                                 + 1.2386 w_k1_INTC_s017 + 1.13447 w_k1_KO_s017
                                 + 1.42505 w_k1_MAT_s017
                                 + 0.973332 w_k1_MSFT_s017
                                 + 1.00579 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.15262 w_k1_ADM_s100
                                 + 0.857346 w_k1_AXP_s100
                                 + 1.06353 w_k1_DIS_s100
                                 + 1.15312 w_k1_FDX_s100
                                 + 0.742713 w_k1_IBM_s100
                                 + 1.13526 w_k1_INTC_s100
                                 + 0.998178 w_k1_KO_s100
                                 + 0.614697 w_k1_MAT_s100
                                 + 1.10482 w_k1_MSFT_s100
                                 + 1.04001 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.04397 w_k1_ADM_s100
                                 + 0.968065 w_k1_AXP_s100
                                 + 1.15766 w_k1_DIS_s100
                                 + 0.991022 w_k1_FDX_s100
                                 + 1.22592 w_k1_IBM_s100
                                 + 1.11435 w_k1_INTC_s100
                                 + 1.12148 w_k1_KO_s100
                                 + 0.992485 w_k1_MAT_s100
                                 + 1.0437 w_k1_MSFT_s100 + 1.06561 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.918596 w_k1_ADM_s100
                                 + 0.603732 w_k1_AXP_s100
                                 + 0.943548 w_k1_DIS_s100
                                 + 0.819695 w_k1_FDX_s100
                                 + 0.854003 w_k1_IBM_s100
                                 + 0.903542 w_k1_INTC_s100
                                 + 0.897468 w_k1_KO_s100
                                 + 0.504495 w_k1_MAT_s100
                                 + 0.890339 w_k1_MSFT_s100
                                 + 0.860612 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.2713 w_k1_ADM_s100
                                 + 1.44806 w_k1_AXP_s100
                                 + 1.45511 w_k1_DIS_s100
                                 + 1.26766 w_k1_FDX_s100
                                 + 1.11823 w_k1_IBM_s100
                                 + 1.24136 w_k1_INTC_s100
                                 + 1.27646 w_k1_KO_s100 + 1.2839 w_k1_MAT_s100
                                 + 1.27508 w_k1_MSFT_s100
                                 + 1.22266 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.844405 w_k1_ADM_s100
                                 + 0.755871 w_k1_AXP_s100
                                 + 0.578604 w_k1_DIS_s100
                                 + 0.749363 w_k1_FDX_s100
                                 + 0.697224 w_k1_IBM_s100
                                 + 0.827401 w_k1_INTC_s100
                                 + 0.684686 w_k1_KO_s100
                                 + 0.714805 w_k1_MAT_s100
                                 + 0.816467 w_k1_MSFT_s100
                                 + 0.76628 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.09393 w_k1_ADM_s100
                                 + 0.77042 w_k1_AXP_s100
                                 + 0.989396 w_k1_DIS_s100
                                 + 0.910462 w_k1_FDX_s100
                                 + 0.990354 w_k1_IBM_s100
                                 + 0.791083 w_k1_INTC_s100
                                 + 1.09603 w_k1_KO_s100 + 1.00514 w_k1_MAT_s100
                                 + 0.852902 w_k1_MSFT_s100
                                 + 0.979484 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.990603 w_k1_ADM_s100
                                 + 1.29774 w_k1_AXP_s100
                                 + 1.13896 w_k1_DIS_s100
                                 + 1.11301 w_k1_FDX_s100
                                 + 1.52271 w_k1_IBM_s100
                                 + 1.08912 w_k1_INTC_s100
                                 + 1.12492 w_k1_KO_s100 + 1.46149 w_k1_MAT_s100
                                 + 1.02882 w_k1_MSFT_s100
                                 + 1.12736 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.02891 w_k1_ADM_s100
                                 + 0.924518 w_k1_AXP_s100
                                 + 1.02216 w_k1_DIS_s100
                                 + 1.03722 w_k1_FDX_s100
                                 + 0.727298 w_k1_IBM_s100
                                 + 0.899231 w_k1_INTC_s100
                                 + 0.824059 w_k1_KO_s100
                                 + 0.730006 w_k1_MAT_s100
                                 + 0.915016 w_k1_MSFT_s100
                                 + 0.873977 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.87868 w_k1_ADM_s101
                                 + 1.01399 w_k1_AXP_s101
                                 + 0.625436 w_k1_DIS_s101
                                 + 0.747775 w_k1_FDX_s101
                                 + 1.25955 w_k1_IBM_s101
                                 + 0.662304 w_k1_INTC_s101
                                 + 0.848428 w_k1_KO_s101
                                 + 1.27035 w_k1_MAT_s101
                                 + 0.639611 w_k1_MSFT_s101
                                 + 0.921567 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.26733 w_k1_ADM_s101
                                 + 1.17104 w_k1_AXP_s101
                                 + 1.81785 w_k1_DIS_s101
                                 + 1.33352 w_k1_FDX_s101
                                 + 1.29969 w_k1_IBM_s101
                                 + 1.25358 w_k1_INTC_s101
                                 + 1.41251 w_k1_KO_s101
                                 + 0.891705 w_k1_MAT_s101
                                 + 1.3051 w_k1_MSFT_s101 + 1.22459 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.992602 w_k1_ADM_s101
                                 + 0.658521 w_k1_AXP_s101
                                 + 0.786698 w_k1_DIS_s101
                                 + 0.8311 w_k1_FDX_s101
                                 + 0.772984 w_k1_IBM_s101
                                 + 0.735374 w_k1_INTC_s101
                                 + 0.813225 w_k1_KO_s101
                                 + 0.698935 w_k1_MAT_s101
                                 + 0.919275 w_k1_MSFT_s101
                                 + 0.830713 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.868173 w_k1_ADM_s101
                                 + 0.937033 w_k1_AXP_s101
                                 + 0.77115 w_k1_DIS_s101
                                 + 0.858791 w_k1_FDX_s101
                                 + 1.04961 w_k1_IBM_s101
                                 + 0.826207 w_k1_INTC_s101
                                 + 0.868481 w_k1_KO_s101
                                 + 0.95151 w_k1_MAT_s101
                                 + 0.82417 w_k1_MSFT_s101
                                 + 0.872175 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.12441 w_k1_ADM_s101
                                 + 0.991232 w_k1_AXP_s101
                                 + 1.03604 w_k1_DIS_s101
                                 + 1.07557 w_k1_FDX_s101
                                 + 1.25323 w_k1_IBM_s101
                                 + 0.826815 w_k1_INTC_s101
                                 + 1.02547 w_k1_KO_s101 + 1.08352 w_k1_MAT_s101
                                 + 1.00827 w_k1_MSFT_s101
                                 + 0.99393 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.27707 w_k1_ADM_s101
                                 + 1.18348 w_k1_AXP_s101
                                 + 1.38032 w_k1_DIS_s101
                                 + 1.26672 w_k1_FDX_s101
                                 + 0.994652 w_k1_IBM_s101
                                 + 1.04112 w_k1_INTC_s101
                                 + 1.17522 w_k1_KO_s101 + 1.15842 w_k1_MAT_s101
                                 + 1.22436 w_k1_MSFT_s101
                                 + 1.21901 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.14137 w_k1_ADM_s101
                                 + 0.983331 w_k1_AXP_s101
                                 + 1.14036 w_k1_DIS_s101
                                 + 1.07119 w_k1_FDX_s101
                                 + 0.795341 w_k1_IBM_s101
                                 + 0.733757 w_k1_INTC_s101
                                 + 0.943453 w_k1_KO_s101
                                 + 0.892901 w_k1_MAT_s101
                                 + 1.01329 w_k1_MSFT_s101
                                 + 0.97641 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.788038 w_k1_ADM_s101
                                 + 0.611116 w_k1_AXP_s101
                                 + 0.438023 w_k1_DIS_s101
                                 + 0.631903 w_k1_FDX_s101
                                 + 0.848578 w_k1_IBM_s101
                                 + 0.598613 w_k1_INTC_s101
                                 + 0.640359 w_k1_KO_s101
                                 + 0.805325 w_k1_MAT_s101
                                 + 0.579134 w_k1_MSFT_s101
                                 + 0.738115 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02228 w_k1_ADM_s102
                                 + 1.02121 w_k1_AXP_s102
                                 + 1.14281 w_k1_DIS_s102
                                 + 1.10342 w_k1_FDX_s102
                                 + 1.07587 w_k1_IBM_s102
                                 + 1.21341 w_k1_INTC_s102
                                 + 1.19274 w_k1_KO_s102 + 1.06342 w_k1_MAT_s102
                                 + 1.15856 w_k1_MSFT_s102
                                 + 1.04261 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.18398 w_k1_ADM_s102
                                 + 1.25663 w_k1_AXP_s102
                                 + 1.33574 w_k1_DIS_s102
                                 + 1.04706 w_k1_FDX_s102
                                 + 1.60443 w_k1_IBM_s102
                                 + 1.16125 w_k1_INTC_s102
                                 + 1.49582 w_k1_KO_s102 + 1.76927 w_k1_MAT_s102
                                 + 1.15152 w_k1_MSFT_s102
                                 + 1.14226 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.68859 w_k1_ADM_s102
                                 + 1.88241 w_k1_AXP_s102
                                 + 1.86312 w_k1_DIS_s102 + 1.4906 w_k1_FDX_s102
                                 + 1.39013 w_k1_IBM_s102
                                 + 1.43361 w_k1_INTC_s102
                                 + 1.37845 w_k1_KO_s102 + 1.68386 w_k1_MAT_s102
                                 + 1.55197 w_k1_MSFT_s102
                                 + 1.21913 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.973938 w_k1_ADM_s102
                                 + 1.03207 w_k1_AXP_s102
                                 + 0.898226 w_k1_DIS_s102
                                 + 1.01848 w_k1_FDX_s102
                                 + 0.738769 w_k1_IBM_s102
                                 + 1.10963 w_k1_INTC_s102
                                 + 0.824448 w_k1_KO_s102
                                 + 0.641917 w_k1_MAT_s102
                                 + 1.0605 w_k1_MSFT_s102
                                 + 0.906796 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.23104 w_k1_ADM_s102
                                 + 1.18425 w_k1_AXP_s102
                                 + 1.49745 w_k1_DIS_s102
                                 + 1.35039 w_k1_FDX_s102
                                 + 0.936963 w_k1_IBM_s102
                                 + 1.37451 w_k1_INTC_s102 + 1.3175 w_k1_KO_s102
                                 + 0.682663 w_k1_MAT_s102
                                 + 1.2688 w_k1_MSFT_s102 + 1.04767 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.961186 w_k1_ADM_s102
                                 + 0.78699 w_k1_AXP_s102
                                 + 0.804502 w_k1_DIS_s102
                                 + 0.734384 w_k1_FDX_s102
                                 + 1.47203 w_k1_IBM_s102
                                 + 0.703956 w_k1_INTC_s102
                                 + 1.05962 w_k1_KO_s102 + 1.02621 w_k1_MAT_s102
                                 + 0.733845 w_k1_MSFT_s102
                                 + 0.944774 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.711895 w_k1_ADM_s102
                                 + 0.573899 w_k1_AXP_s102
                                 + 0.610611 w_k1_DIS_s102
                                 + 0.733298 w_k1_FDX_s102
                                 + 0.75173 w_k1_IBM_s102
                                 + 0.767627 w_k1_INTC_s102
                                 + 0.829474 w_k1_KO_s102
                                 + 0.564308 w_k1_MAT_s102
                                 + 0.726697 w_k1_MSFT_s102
                                 + 0.822692 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.951633 w_k1_ADM_s102
                                 + 0.931858 w_k1_AXP_s102
                                 + 0.800201 w_k1_DIS_s102
                                 + 0.785043 w_k1_FDX_s102
                                 + 1.10077 w_k1_IBM_s102
                                 + 0.841445 w_k1_INTC_s102
                                 + 1.05501 w_k1_KO_s102
                                 + 0.973055 w_k1_MAT_s102
                                 + 0.782848 w_k1_MSFT_s102
                                 + 0.992535 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.813127 w_k1_ADM_s103
                                 + 0.57547 w_k1_AXP_s103
                                 + 0.694229 w_k1_DIS_s103
                                 + 0.827857 w_k1_FDX_s103
                                 + 0.803183 w_k1_IBM_s103
                                 + 0.79087 w_k1_INTC_s103
                                 + 0.741169 w_k1_KO_s103
                                 + 0.479461 w_k1_MAT_s103
                                 + 0.829159 w_k1_MSFT_s103
                                 + 0.833947 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.25847 w_k1_ADM_s103
                                 + 1.05412 w_k1_AXP_s103
                                 + 1.10096 w_k1_DIS_s103
                                 + 1.04128 w_k1_FDX_s103
                                 + 1.63342 w_k1_IBM_s103
                                 + 1.07191 w_k1_INTC_s103
                                 + 1.36106 w_k1_KO_s103 + 1.35853 w_k1_MAT_s103
                                 + 1.1479 w_k1_MSFT_s103 + 1.1512 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.07023 w_k1_ADM_s103
                                 + 1.22256 w_k1_AXP_s103
                                 + 1.17478 w_k1_DIS_s103
                                 + 1.20476 w_k1_FDX_s103
                                 + 0.960852 w_k1_IBM_s103
                                 + 1.24237 w_k1_INTC_s103
                                 + 1.07626 w_k1_KO_s103 + 1.09345 w_k1_MAT_s103
                                 + 1.32405 w_k1_MSFT_s103
                                 + 1.21215 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.00011 w_k1_ADM_s103
                                 + 1.23901 w_k1_AXP_s103 + 1.0034 w_k1_DIS_s103
                                 + 0.987973 w_k1_FDX_s103
                                 + 1.34848 w_k1_IBM_s103
                                 + 1.12604 w_k1_INTC_s103
                                 + 1.10333 w_k1_KO_s103 + 1.20122 w_k1_MAT_s103
                                 + 0.93917 w_k1_MSFT_s103
                                 + 1.13797 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.1776 w_k1_ADM_s103
                                 + 1.3649 w_k1_AXP_s103 + 1.73547 w_k1_DIS_s103
                                 + 1.29458 w_k1_FDX_s103 + 1.3142 w_k1_IBM_s103
                                 + 1.24354 w_k1_INTC_s103
                                 + 1.31924 w_k1_KO_s103 + 1.15778 w_k1_MAT_s103
                                 + 1.35366 w_k1_MSFT_s103
                                 + 1.06744 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.771396 w_k1_ADM_s103
                                 + 0.8797 w_k1_AXP_s103
                                 + 0.860259 w_k1_DIS_s103
                                 + 1.00036 w_k1_FDX_s103
                                 + 0.863972 w_k1_IBM_s103
                                 + 0.972173 w_k1_INTC_s103
                                 + 0.796692 w_k1_KO_s103
                                 + 0.99323 w_k1_MAT_s103
                                 + 0.997135 w_k1_MSFT_s103
                                 + 0.977544 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.0112 w_k1_ADM_s103
                                 + 1.10973 w_k1_AXP_s103 + 1.0712 w_k1_DIS_s103
                                 + 1.07311 w_k1_FDX_s103 + 0.9779 w_k1_IBM_s103
                                 + 1.10602 w_k1_INTC_s103
                                 + 1.07494 w_k1_KO_s103
                                 + 0.863641 w_k1_MAT_s103
                                 + 1.10651 w_k1_MSFT_s103 + 1.0827 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.832975 w_k1_ADM_s103
                                 + 0.805256 w_k1_AXP_s103
                                 + 0.893725 w_k1_DIS_s103
                                 + 0.805622 w_k1_FDX_s103
                                 + 1.12037 w_k1_IBM_s103
                                 + 0.999954 w_k1_INTC_s103
                                 + 1.15765 w_k1_KO_s103 + 1.08305 w_k1_MAT_s103
                                 + 0.878773 w_k1_MSFT_s103
                                 + 0.963178 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.844548 w_k1_ADM_s104
                                 + 0.708837 w_k1_AXP_s104
                                 + 0.952768 w_k1_DIS_s104
                                 + 0.773225 w_k1_FDX_s104
                                 + 1.07119 w_k1_IBM_s104
                                 + 0.859314 w_k1_INTC_s104
                                 + 1.0545 w_k1_KO_s104 + 1.34232 w_k1_MAT_s104
                                 + 0.853568 w_k1_MSFT_s104
                                 + 1.10271 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.16384 w_k1_ADM_s104
                                 + 1.16226 w_k1_AXP_s104
                                 + 1.44127 w_k1_DIS_s104
                                 + 1.13704 w_k1_FDX_s104 + 1.5289 w_k1_IBM_s104
                                 + 1.03258 w_k1_INTC_s104
                                 + 1.27556 w_k1_KO_s104 + 2.34246 w_k1_MAT_s104
                                 + 1.03445 w_k1_MSFT_s104
                                 + 1.08633 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.836917 w_k1_ADM_s104
                                 + 0.747505 w_k1_AXP_s104
                                 + 0.791186 w_k1_DIS_s104
                                 + 0.838361 w_k1_FDX_s104
                                 + 0.673947 w_k1_IBM_s104
                                 + 0.897656 w_k1_INTC_s104
                                 + 0.77479 w_k1_KO_s104
                                 + 0.782149 w_k1_MAT_s104
                                 + 0.89797 w_k1_MSFT_s104
                                 + 0.887696 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.05902 w_k1_ADM_s104
                                 + 0.963586 w_k1_AXP_s104
                                 + 1.12499 w_k1_DIS_s104
                                 + 1.11078 w_k1_FDX_s104
                                 + 0.808137 w_k1_IBM_s104
                                 + 1.01682 w_k1_INTC_s104
                                 + 1.07538 w_k1_KO_s104
                                 + 0.957788 w_k1_MAT_s104
                                 + 0.980932 w_k1_MSFT_s104
                                 + 1.0119 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.16925 w_k1_ADM_s104
                                 + 1.32687 w_k1_AXP_s104
                                 + 1.36816 w_k1_DIS_s104
                                 + 1.37876 w_k1_FDX_s104
                                 + 1.13891 w_k1_IBM_s104
                                 + 1.66182 w_k1_INTC_s104
                                 + 1.34724 w_k1_KO_s104 + 1.50756 w_k1_MAT_s104
                                 + 1.27897 w_k1_MSFT_s104
                                 + 1.23616 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.17609 w_k1_ADM_s104
                                 + 0.895777 w_k1_AXP_s104
                                 + 1.10783 w_k1_DIS_s104
                                 + 1.00076 w_k1_FDX_s104
                                 + 1.06418 w_k1_IBM_s104
                                 + 1.02423 w_k1_INTC_s104
                                 + 1.09874 w_k1_KO_s104 + 1.20836 w_k1_MAT_s104
                                 + 1.02952 w_k1_MSFT_s104
                                 + 1.15602 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.35397 w_k1_ADM_s104
                                 + 1.46709 w_k1_AXP_s104 + 1.8513 w_k1_DIS_s104
                                 + 1.45724 w_k1_FDX_s104 + 1.6724 w_k1_IBM_s104
                                 + 1.34541 w_k1_INTC_s104
                                 + 1.72514 w_k1_KO_s104 + 1.87423 w_k1_MAT_s104
                                 + 1.38948 w_k1_MSFT_s104
                                 + 1.28622 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.976289 w_k1_ADM_s104
                                 + 0.861466 w_k1_AXP_s104
                                 + 0.971738 w_k1_DIS_s104
                                 + 0.998247 w_k1_FDX_s104
                                 + 0.944457 w_k1_IBM_s104
                                 + 0.976197 w_k1_INTC_s104
                                 + 1.05202 w_k1_KO_s104 + 1.28368 w_k1_MAT_s104
                                 + 0.964517 w_k1_MSFT_s104
                                 + 1.0387 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.01113 w_k1_ADM_s105
                                 + 0.957485 w_k1_AXP_s105
                                 + 0.96447 w_k1_DIS_s105
                                 + 0.971031 w_k1_FDX_s105
                                 + 1.0041 w_k1_IBM_s105
                                 + 0.903219 w_k1_INTC_s105
                                 + 0.889133 w_k1_KO_s105
                                 + 0.914356 w_k1_MAT_s105
                                 + 0.880743 w_k1_MSFT_s105
                                 + 0.955435 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.838822 w_k1_ADM_s105
                                 + 1.23463 w_k1_AXP_s105
                                 + 1.19603 w_k1_DIS_s105
                                 + 0.999225 w_k1_FDX_s105
                                 + 1.5389 w_k1_IBM_s105
                                 + 0.998298 w_k1_INTC_s105
                                 + 1.03765 w_k1_KO_s105 + 1.3261 w_k1_MAT_s105
                                 + 0.868208 w_k1_MSFT_s105
                                 + 1.09618 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.833205 w_k1_ADM_s105
                                 + 0.625937 w_k1_AXP_s105
                                 + 0.537584 w_k1_DIS_s105
                                 + 0.816286 w_k1_FDX_s105
                                 + 0.586219 w_k1_IBM_s105
                                 + 0.94393 w_k1_INTC_s105
                                 + 0.593674 w_k1_KO_s105
                                 + 0.394701 w_k1_MAT_s105
                                 + 0.758765 w_k1_MSFT_s105
                                 + 0.767622 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.933301 w_k1_ADM_s105
                                 + 1.17784 w_k1_AXP_s105
                                 + 1.38979 w_k1_DIS_s105
                                 + 1.24911 w_k1_FDX_s105
                                 + 0.911158 w_k1_IBM_s105
                                 + 1.03968 w_k1_INTC_s105
                                 + 1.10127 w_k1_KO_s105
                                 + 0.991316 w_k1_MAT_s105
                                 + 1.13153 w_k1_MSFT_s105 + 1.1113 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.04105 w_k1_ADM_s105
                                 + 1.54851 w_k1_AXP_s105
                                 + 1.55701 w_k1_DIS_s105
                                 + 1.28476 w_k1_FDX_s105
                                 + 1.34072 w_k1_IBM_s105
                                 + 1.20853 w_k1_INTC_s105
                                 + 1.50699 w_k1_KO_s105 + 1.96389 w_k1_MAT_s105
                                 + 1.30045 w_k1_MSFT_s105
                                 + 1.24805 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.890739 w_k1_ADM_s105
                                 + 1.12581 w_k1_AXP_s105
                                 + 1.05104 w_k1_DIS_s105
                                 + 1.17544 w_k1_FDX_s105
                                 + 0.946914 w_k1_IBM_s105
                                 + 1.1749 w_k1_INTC_s105 + 1.03549 w_k1_KO_s105
                                 + 1.24626 w_k1_MAT_s105
                                 + 1.07785 w_k1_MSFT_s105
                                 + 1.04217 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.857976 w_k1_ADM_s105
                                 + 0.719539 w_k1_AXP_s105
                                 + 0.648823 w_k1_DIS_s105
                                 + 0.723351 w_k1_FDX_s105
                                 + 1.01839 w_k1_IBM_s105
                                 + 0.891971 w_k1_INTC_s105
                                 + 0.866593 w_k1_KO_s105
                                 + 0.782916 w_k1_MAT_s105
                                 + 0.801203 w_k1_MSFT_s105
                                 + 0.837341 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.935077 w_k1_ADM_s105
                                 + 0.739976 w_k1_AXP_s105
                                 + 0.838114 w_k1_DIS_s105
                                 + 0.855862 w_k1_FDX_s105
                                 + 0.739558 w_k1_IBM_s105
                                 + 1.00518 w_k1_INTC_s105
                                 + 0.867684 w_k1_KO_s105
                                 + 0.698103 w_k1_MAT_s105
                                 + 1.05569 w_k1_MSFT_s105
                                 + 0.940297 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.819971 w_k1_ADM_s106
                                 + 0.74389 w_k1_AXP_s106
                                 + 0.728922 w_k1_DIS_s106
                                 + 0.816687 w_k1_FDX_s106
                                 + 0.781046 w_k1_IBM_s106
                                 + 0.736861 w_k1_INTC_s106
                                 + 0.791816 w_k1_KO_s106
                                 + 0.605038 w_k1_MAT_s106
                                 + 0.802666 w_k1_MSFT_s106
                                 + 0.885047 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.0025 w_k1_ADM_s106
                                 + 1.36723 w_k1_AXP_s106
                                 + 1.43181 w_k1_DIS_s106
                                 + 1.40452 w_k1_FDX_s106
                                 + 1.01547 w_k1_IBM_s106
                                 + 1.52078 w_k1_INTC_s106
                                 + 1.11685 w_k1_KO_s106
                                 + 0.789241 w_k1_MAT_s106
                                 + 1.44946 w_k1_MSFT_s106
                                 + 1.14257 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.814544 w_k1_ADM_s106
                                 + 0.829726 w_k1_AXP_s106
                                 + 0.789535 w_k1_DIS_s106
                                 + 0.880798 w_k1_FDX_s106
                                 + 0.875087 w_k1_IBM_s106
                                 + 1.14095 w_k1_INTC_s106
                                 + 0.84094 w_k1_KO_s106
                                 + 0.799809 w_k1_MAT_s106
                                 + 0.987222 w_k1_MSFT_s106
                                 + 0.888554 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.05856 w_k1_ADM_s106
                                 + 0.953498 w_k1_AXP_s106
                                 + 0.667115 w_k1_DIS_s106
                                 + 0.935079 w_k1_FDX_s106
                                 + 1.17473 w_k1_IBM_s106
                                 + 0.855738 w_k1_INTC_s106
                                 + 0.885219 w_k1_KO_s106
                                 + 1.06966 w_k1_MAT_s106
                                 + 0.883529 w_k1_MSFT_s106
                                 + 0.991873 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.06095 w_k1_ADM_s106
                                 + 1.1276 w_k1_AXP_s106
                                 + 0.852632 w_k1_DIS_s106
                                 + 1.05022 w_k1_FDX_s106
                                 + 0.931815 w_k1_IBM_s106
                                 + 1.26969 w_k1_INTC_s106
                                 + 0.891905 w_k1_KO_s106
                                 + 0.783381 w_k1_MAT_s106
                                 + 1.05905 w_k1_MSFT_s106
                                 + 1.01809 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.09496 w_k1_ADM_s106
                                 + 1.1563 w_k1_AXP_s106 + 1.04194 w_k1_DIS_s106
                                 + 1.22885 w_k1_FDX_s106
                                 + 0.93757 w_k1_IBM_s106
                                 + 1.14287 w_k1_INTC_s106
                                 + 0.923802 w_k1_KO_s106
                                 + 0.945263 w_k1_MAT_s106
                                 + 1.0798 w_k1_MSFT_s106 + 1.07327 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.880856 w_k1_ADM_s106
                                 + 0.920741 w_k1_AXP_s106
                                 + 0.828266 w_k1_DIS_s106
                                 + 1.02987 w_k1_FDX_s106
                                 + 0.665948 w_k1_IBM_s106
                                 + 1.19181 w_k1_INTC_s106
                                 + 0.828405 w_k1_KO_s106
                                 + 0.853322 w_k1_MAT_s106
                                 + 0.979298 w_k1_MSFT_s106
                                 + 0.877213 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08299 w_k1_ADM_s106
                                 + 1.26212 w_k1_AXP_s106
                                 + 1.35116 w_k1_DIS_s106 + 1.2041 w_k1_FDX_s106
                                 + 1.51699 w_k1_IBM_s106
                                 + 1.25233 w_k1_INTC_s106
                                 + 1.42166 w_k1_KO_s106 + 1.8655 w_k1_MAT_s106
                                 + 1.21483 w_k1_MSFT_s106
                                 + 1.24741 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.954216 w_k1_ADM_s107
                                 + 1.13536 w_k1_AXP_s107
                                 + 0.849707 w_k1_DIS_s107
                                 + 0.940642 w_k1_FDX_s107
                                 + 0.69611 w_k1_IBM_s107
                                 + 0.987123 w_k1_INTC_s107
                                 + 0.863684 w_k1_KO_s107
                                 + 0.678002 w_k1_MAT_s107
                                 + 0.793071 w_k1_MSFT_s107
                                 + 0.853285 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.02555 w_k1_ADM_s107
                                 + 0.737128 w_k1_AXP_s107
                                 + 0.751633 w_k1_DIS_s107
                                 + 0.890788 w_k1_FDX_s107
                                 + 0.584481 w_k1_IBM_s107
                                 + 0.864492 w_k1_INTC_s107
                                 + 0.656559 w_k1_KO_s107
                                 + 0.45942 w_k1_MAT_s107
                                 + 0.871998 w_k1_MSFT_s107
                                 + 0.751487 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.19758 w_k1_ADM_s107
                                 + 1.18014 w_k1_AXP_s107
                                 + 0.955802 w_k1_DIS_s107
                                 + 1.01729 w_k1_FDX_s107
                                 + 1.11931 w_k1_IBM_s107
                                 + 1.02128 w_k1_INTC_s107
                                 + 1.04831 w_k1_KO_s107 + 1.04957 w_k1_MAT_s107
                                 + 1.01449 w_k1_MSFT_s107
                                 + 0.944986 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.980049 w_k1_ADM_s107
                                 + 0.659214 w_k1_AXP_s107
                                 + 0.659541 w_k1_DIS_s107
                                 + 0.695966 w_k1_FDX_s107
                                 + 0.740203 w_k1_IBM_s107
                                 + 0.720428 w_k1_INTC_s107
                                 + 0.742227 w_k1_KO_s107
                                 + 0.40303 w_k1_MAT_s107
                                 + 0.743249 w_k1_MSFT_s107
                                 + 0.748112 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.11399 w_k1_ADM_s107
                                 + 0.848942 w_k1_AXP_s107
                                 + 0.867112 w_k1_DIS_s107
                                 + 0.739935 w_k1_FDX_s107
                                 + 1.06905 w_k1_IBM_s107
                                 + 0.901051 w_k1_INTC_s107
                                 + 0.994139 w_k1_KO_s107
                                 + 0.754266 w_k1_MAT_s107
                                 + 0.819718 w_k1_MSFT_s107
                                 + 0.902119 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.47439 w_k1_ADM_s107
                                 + 1.29145 w_k1_AXP_s107
                                 + 1.37699 w_k1_DIS_s107
                                 + 1.14874 w_k1_FDX_s107
                                 + 1.08355 w_k1_IBM_s107
                                 + 1.17071 w_k1_INTC_s107
                                 + 1.25108 w_k1_KO_s107 + 1.04108 w_k1_MAT_s107
                                 + 1.12559 w_k1_MSFT_s107
                                 + 1.06555 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.40745 w_k1_ADM_s107
                                 + 1.7657 w_k1_AXP_s107 + 2.01095 w_k1_DIS_s107
                                 + 1.5028 w_k1_FDX_s107 + 1.64691 w_k1_IBM_s107
                                 + 1.03786 w_k1_INTC_s107 + 1.6422 w_k1_KO_s107
                                 + 2.22722 w_k1_MAT_s107
                                 + 1.47658 w_k1_MSFT_s107
                                 + 1.31191 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.37346 w_k1_ADM_s107
                                 + 0.854691 w_k1_AXP_s107
                                 + 1.07994 w_k1_DIS_s107
                                 + 1.09829 w_k1_FDX_s107
                                 + 0.735612 w_k1_IBM_s107
                                 + 0.948129 w_k1_INTC_s107
                                 + 1.07266 w_k1_KO_s107
                                 + 0.728062 w_k1_MAT_s107
                                 + 1.0766 w_k1_MSFT_s107
                                 + 0.820595 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.02929 w_k1_ADM_s110
                                 + 1.59745 w_k1_AXP_s110
                                 + 1.25575 w_k1_DIS_s110
                                 + 1.14465 w_k1_FDX_s110
                                 + 1.63687 w_k1_IBM_s110
                                 + 1.04564 w_k1_INTC_s110 + 1.3987 w_k1_KO_s110
                                 + 1.69945 w_k1_MAT_s110
                                 + 1.06943 w_k1_MSFT_s110
                                 + 1.25428 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.685335 w_k1_ADM_s110
                                 + 0.79279 w_k1_AXP_s110
                                 + 0.658889 w_k1_DIS_s110
                                 + 0.759316 w_k1_FDX_s110
                                 + 0.795043 w_k1_IBM_s110
                                 + 0.812991 w_k1_INTC_s110
                                 + 0.72114 w_k1_KO_s110 + 1.03329 w_k1_MAT_s110
                                 + 0.780214 w_k1_MSFT_s110
                                 + 0.905267 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.03836 w_k1_ADM_s110
                                 + 1.12417 w_k1_AXP_s110
                                 + 1.16729 w_k1_DIS_s110
                                 + 0.995355 w_k1_FDX_s110
                                 + 1.16193 w_k1_IBM_s110
                                 + 1.08434 w_k1_INTC_s110
                                 + 1.17302 w_k1_KO_s110 + 1.21845 w_k1_MAT_s110
                                 + 1.10276 w_k1_MSFT_s110
                                 + 1.02932 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.38934 w_k1_ADM_s110
                                 + 1.69746 w_k1_AXP_s110
                                 + 1.75284 w_k1_DIS_s110
                                 + 1.45292 w_k1_FDX_s110
                                 + 1.09563 w_k1_IBM_s110
                                 + 1.11257 w_k1_INTC_s110
                                 + 1.19441 w_k1_KO_s110 + 1.39324 w_k1_MAT_s110
                                 + 1.32867 w_k1_MSFT_s110 + 1.2437 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.24728 w_k1_ADM_s110
                                 + 1.43467 w_k1_AXP_s110
                                 + 1.18213 w_k1_DIS_s110
                                 + 1.22752 w_k1_FDX_s110
                                 + 0.785779 w_k1_IBM_s110
                                 + 1.14894 w_k1_INTC_s110
                                 + 0.906159 w_k1_KO_s110
                                 + 0.886667 w_k1_MAT_s110
                                 + 1.10566 w_k1_MSFT_s110
                                 + 1.17233 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06676 w_k1_ADM_s110
                                 + 1.00272 w_k1_AXP_s110
                                 + 0.899386 w_k1_DIS_s110
                                 + 1.0192 w_k1_FDX_s110
                                 + 0.812126 w_k1_IBM_s110
                                 + 0.963229 w_k1_INTC_s110
                                 + 0.919819 w_k1_KO_s110
                                 + 1.03868 w_k1_MAT_s110
                                 + 0.946153 w_k1_MSFT_s110
                                 + 0.911828 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.805789 w_k1_ADM_s110
                                 + 1.27604 w_k1_AXP_s110
                                 + 0.828606 w_k1_DIS_s110
                                 + 0.928904 w_k1_FDX_s110
                                 + 0.976688 w_k1_IBM_s110
                                 + 0.883784 w_k1_INTC_s110
                                 + 0.85253 w_k1_KO_s110 + 1.21197 w_k1_MAT_s110
                                 + 0.821399 w_k1_MSFT_s110
                                 + 0.990346 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.30148 w_k1_ADM_s110
                                 + 1.30249 w_k1_AXP_s110
                                 + 0.935344 w_k1_DIS_s110
                                 + 0.962968 w_k1_FDX_s110
                                 + 1.19232 w_k1_IBM_s110
                                 + 0.75598 w_k1_INTC_s110
                                 + 1.08225 w_k1_KO_s110 + 1.75403 w_k1_MAT_s110
                                 + 0.971711 w_k1_MSFT_s110
                                 + 1.20762 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.928429 w_k1_ADM_s111
                                 + 1.42283 w_k1_AXP_s111
                                 + 1.34549 w_k1_DIS_s111 + 1.1606 w_k1_FDX_s111
                                 + 1.69078 w_k1_IBM_s111
                                 + 1.2203 w_k1_INTC_s111 + 1.38186 w_k1_KO_s111
                                 + 1.80296 w_k1_MAT_s111
                                 + 1.24049 w_k1_MSFT_s111
                                 + 1.11472 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.923699 w_k1_ADM_s111
                                 + 0.78981 w_k1_AXP_s111
                                 + 0.620567 w_k1_DIS_s111
                                 + 0.784402 w_k1_FDX_s111
                                 + 0.977828 w_k1_IBM_s111
                                 + 0.998699 w_k1_INTC_s111
                                 + 0.853622 w_k1_KO_s111
                                 + 1.10011 w_k1_MAT_s111
                                 + 0.874642 w_k1_MSFT_s111
                                 + 0.885455 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.998048 w_k1_ADM_s111
                                 + 0.947934 w_k1_AXP_s111
                                 + 0.936049 w_k1_DIS_s111
                                 + 0.863755 w_k1_FDX_s111
                                 + 1.27758 w_k1_IBM_s111
                                 + 1.00005 w_k1_INTC_s111
                                 + 1.15058 w_k1_KO_s111 + 1.15481 w_k1_MAT_s111
                                 + 1.14747 w_k1_MSFT_s111
                                 + 0.960735 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.14453 w_k1_ADM_s111
                                 + 1.00675 w_k1_AXP_s111
                                 + 1.14972 w_k1_DIS_s111
                                 + 1.20924 w_k1_FDX_s111
                                 + 0.845695 w_k1_IBM_s111
                                 + 1.14193 w_k1_INTC_s111
                                 + 1.08991 w_k1_KO_s111
                                 + 0.912628 w_k1_MAT_s111
                                 + 1.17428 w_k1_MSFT_s111
                                 + 1.07943 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.54194 w_k1_ADM_s111
                                 + 1.33198 w_k1_AXP_s111 + 1.4381 w_k1_DIS_s111
                                 + 1.35429 w_k1_FDX_s111
                                 + 1.14977 w_k1_IBM_s111
                                 + 1.57319 w_k1_INTC_s111
                                 + 1.39752 w_k1_KO_s111 + 1.57444 w_k1_MAT_s111
                                 + 1.46382 w_k1_MSFT_s111
                                 + 1.24649 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.07032 w_k1_ADM_s111
                                 + 0.826915 w_k1_AXP_s111
                                 + 0.987165 w_k1_DIS_s111
                                 + 0.959429 w_k1_FDX_s111
                                 + 0.772386 w_k1_IBM_s111
                                 + 1.09656 w_k1_INTC_s111
                                 + 0.888933 w_k1_KO_s111
                                 + 0.811236 w_k1_MAT_s111
                                 + 1.07246 w_k1_MSFT_s111
                                 + 1.03631 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.841412 w_k1_ADM_s111
                                 + 0.692793 w_k1_AXP_s111
                                 + 0.71465 w_k1_DIS_s111
                                 + 0.78793 w_k1_FDX_s111
                                 + 0.739005 w_k1_IBM_s111
                                 + 0.886186 w_k1_INTC_s111
                                 + 0.764742 w_k1_KO_s111
                                 + 0.647027 w_k1_MAT_s111
                                 + 0.838254 w_k1_MSFT_s111
                                 + 0.733105 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.16775 w_k1_ADM_s111
                                 + 1.18934 w_k1_AXP_s111
                                 + 1.33035 w_k1_DIS_s111
                                 + 1.15765 w_k1_FDX_s111
                                 + 1.13625 w_k1_IBM_s111
                                 + 1.09257 w_k1_INTC_s111
                                 + 1.16244 w_k1_KO_s111 + 1.17185 w_k1_MAT_s111
                                 + 1.12867 w_k1_MSFT_s111
                                 + 1.11717 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.929062 w_k1_ADM_s112
                                 + 0.932462 w_k1_AXP_s112
                                 + 1.05253 w_k1_DIS_s112
                                 + 0.901008 w_k1_FDX_s112
                                 + 1.49466 w_k1_IBM_s112
                                 + 0.954027 w_k1_INTC_s112
                                 + 1.06493 w_k1_KO_s112 + 1.6675 w_k1_MAT_s112
                                 + 0.90422 w_k1_MSFT_s112
                                 + 0.91986 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.32396 w_k1_ADM_s112
                                 + 1.30249 w_k1_AXP_s112
                                 + 1.50463 w_k1_DIS_s112
                                 + 1.26362 w_k1_FDX_s112
                                 + 1.34522 w_k1_IBM_s112
                                 + 1.10674 w_k1_INTC_s112
                                 + 1.06749 w_k1_KO_s112 + 1.32168 w_k1_MAT_s112
                                 + 0.955479 w_k1_MSFT_s112
                                 + 1.14231 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.805013 w_k1_ADM_s112
                                 + 0.759567 w_k1_AXP_s112
                                 + 0.633572 w_k1_DIS_s112
                                 + 0.756084 w_k1_FDX_s112
                                 + 0.587736 w_k1_IBM_s112
                                 + 0.786303 w_k1_INTC_s112
                                 + 0.531494 w_k1_KO_s112
                                 + 0.640313 w_k1_MAT_s112
                                 + 0.698832 w_k1_MSFT_s112
                                 + 0.751599 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.754302 w_k1_ADM_s112
                                 + 0.792589 w_k1_AXP_s112
                                 + 0.809279 w_k1_DIS_s112
                                 + 0.869855 w_k1_FDX_s112
                                 + 0.911283 w_k1_IBM_s112
                                 + 0.839323 w_k1_INTC_s112
                                 + 0.705702 w_k1_KO_s112
                                 + 1.20139 w_k1_MAT_s112
                                 + 0.830369 w_k1_MSFT_s112
                                 + 0.831272 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.912483 w_k1_ADM_s112
                                 + 0.815459 w_k1_AXP_s112
                                 + 1.10848 w_k1_DIS_s112
                                 + 0.940538 w_k1_FDX_s112
                                 + 1.10897 w_k1_IBM_s112
                                 + 1.01025 w_k1_INTC_s112
                                 + 1.05069 w_k1_KO_s112 + 1.25161 w_k1_MAT_s112
                                 + 1.02319 w_k1_MSFT_s112
                                 + 0.926845 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10782 w_k1_ADM_s112
                                 + 1.55145 w_k1_AXP_s112
                                 + 0.984894 w_k1_DIS_s112
                                 + 1.07082 w_k1_FDX_s112
                                 + 1.22581 w_k1_IBM_s112
                                 + 1.13981 w_k1_INTC_s112
                                 + 0.924682 w_k1_KO_s112
                                 + 1.60309 w_k1_MAT_s112
                                 + 0.870257 w_k1_MSFT_s112
                                 + 1.04855 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.981959 w_k1_ADM_s112
                                 + 1.0205 w_k1_AXP_s112 + 1.02074 w_k1_DIS_s112
                                 + 1.03803 w_k1_FDX_s112
                                 + 1.00363 w_k1_IBM_s112
                                 + 1.09013 w_k1_INTC_s112
                                 + 0.925922 w_k1_KO_s112
                                 + 1.06945 w_k1_MAT_s112
                                 + 0.983539 w_k1_MSFT_s112
                                 + 1.06875 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.26822 w_k1_ADM_s112
                                 + 1.147 w_k1_AXP_s112 + 1.73958 w_k1_DIS_s112
                                 + 1.34063 w_k1_FDX_s112
                                 + 0.965537 w_k1_IBM_s112
                                 + 1.30215 w_k1_INTC_s112
                                 + 1.22608 w_k1_KO_s112 + 1.5431 w_k1_MAT_s112
                                 + 1.21598 w_k1_MSFT_s112
                                 + 1.19526 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.953224 w_k1_ADM_s113
                                 + 0.752791 w_k1_AXP_s113
                                 + 0.593473 w_k1_DIS_s113
                                 + 0.720201 w_k1_FDX_s113
                                 + 0.85839 w_k1_IBM_s113
                                 + 0.769118 w_k1_INTC_s113
                                 + 0.861391 w_k1_KO_s113
                                 + 0.740983 w_k1_MAT_s113
                                 + 0.826806 w_k1_MSFT_s113
                                 + 0.813555 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.18234 w_k1_ADM_s113
                                 + 1.05391 w_k1_AXP_s113
                                 + 0.910106 w_k1_DIS_s113
                                 + 0.990532 w_k1_FDX_s113
                                 + 0.996586 w_k1_IBM_s113
                                 + 0.866683 w_k1_INTC_s113
                                 + 1.02273 w_k1_KO_s113
                                 + 0.810184 w_k1_MAT_s113
                                 + 0.789708 w_k1_MSFT_s113
                                 + 0.861939 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.12185 w_k1_ADM_s113
                                 + 1.52043 w_k1_AXP_s113
                                 + 1.20347 w_k1_DIS_s113
                                 + 1.16866 w_k1_FDX_s113
                                 + 1.35216 w_k1_IBM_s113
                                 + 1.05436 w_k1_INTC_s113
                                 + 1.40742 w_k1_KO_s113 + 1.89931 w_k1_MAT_s113
                                 + 1.10256 w_k1_MSFT_s113
                                 + 1.10342 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.733061 w_k1_ADM_s113
                                 + 0.771829 w_k1_AXP_s113
                                 + 0.586946 w_k1_DIS_s113
                                 + 0.701296 w_k1_FDX_s113
                                 + 0.640366 w_k1_IBM_s113
                                 + 0.806317 w_k1_INTC_s113
                                 + 0.810112 w_k1_KO_s113
                                 + 0.572216 w_k1_MAT_s113
                                 + 0.835864 w_k1_MSFT_s113
                                 + 0.846875 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.38345 w_k1_ADM_s113
                                 + 1.32668 w_k1_AXP_s113
                                 + 1.33004 w_k1_DIS_s113
                                 + 1.27663 w_k1_FDX_s113
                                 + 0.982085 w_k1_IBM_s113
                                 + 1.18125 w_k1_INTC_s113
                                 + 1.28605 w_k1_KO_s113
                                 + 0.921105 w_k1_MAT_s113
                                 + 1.24516 w_k1_MSFT_s113
                                 + 1.17424 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.79448 w_k1_ADM_s113
                                 + 0.829179 w_k1_AXP_s113
                                 + 0.745858 w_k1_DIS_s113
                                 + 0.892315 w_k1_FDX_s113
                                 + 0.586895 w_k1_IBM_s113
                                 + 1.03441 w_k1_INTC_s113
                                 + 0.868006 w_k1_KO_s113
                                 + 0.465032 w_k1_MAT_s113
                                 + 1.02673 w_k1_MSFT_s113
                                 + 0.832723 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.10735 w_k1_ADM_s113
                                 + 0.99539 w_k1_AXP_s113
                                 + 0.905201 w_k1_DIS_s113
                                 + 0.878408 w_k1_FDX_s113
                                 + 1.30695 w_k1_IBM_s113
                                 + 0.845891 w_k1_INTC_s113
                                 + 1.22064 w_k1_KO_s113 + 1.19032 w_k1_MAT_s113
                                 + 1.0244 w_k1_MSFT_s113 + 1.01544 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.07262 w_k1_ADM_s113
                                 + 0.919436 w_k1_AXP_s113
                                 + 0.954092 w_k1_DIS_s113
                                 + 0.971137 w_k1_FDX_s113
                                 + 0.813544 w_k1_IBM_s113
                                 + 0.882494 w_k1_INTC_s113
                                 + 0.998129 w_k1_KO_s113
                                 + 0.794757 w_k1_MAT_s113
                                 + 0.963937 w_k1_MSFT_s113
                                 + 0.941174 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.01879 w_k1_ADM_s114
                                 + 1.37372 w_k1_AXP_s114
                                 + 2.04343 w_k1_DIS_s114
                                 + 1.54409 w_k1_FDX_s114
                                 + 1.15928 w_k1_IBM_s114
                                 + 1.30651 w_k1_INTC_s114
                                 + 1.44047 w_k1_KO_s114 + 1.18298 w_k1_MAT_s114
                                 + 1.43494 w_k1_MSFT_s114
                                 + 1.21069 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.867408 w_k1_ADM_s114
                                 + 1.38737 w_k1_AXP_s114
                                 + 1.13101 w_k1_DIS_s114
                                 + 1.04139 w_k1_FDX_s114
                                 + 1.41686 w_k1_IBM_s114
                                 + 1.16557 w_k1_INTC_s114
                                 + 1.20885 w_k1_KO_s114 + 1.35824 w_k1_MAT_s114
                                 + 0.989926 w_k1_MSFT_s114
                                 + 0.979802 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.03851 w_k1_ADM_s114
                                 + 0.819609 w_k1_AXP_s114
                                 + 0.943717 w_k1_DIS_s114
                                 + 0.935636 w_k1_FDX_s114
                                 + 1.02589 w_k1_IBM_s114
                                 + 0.864969 w_k1_INTC_s114
                                 + 0.979844 w_k1_KO_s114
                                 + 0.659183 w_k1_MAT_s114
                                 + 0.934485 w_k1_MSFT_s114
                                 + 0.963782 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.15839 w_k1_ADM_s114
                                 + 1.57782 w_k1_AXP_s114
                                 + 1.61199 w_k1_DIS_s114
                                 + 1.29198 w_k1_FDX_s114
                                 + 1.91148 w_k1_IBM_s114
                                 + 1.33082 w_k1_INTC_s114
                                 + 1.65948 w_k1_KO_s114 + 2.32208 w_k1_MAT_s114
                                 + 1.44621 w_k1_MSFT_s114
                                 + 1.32628 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.615222 w_k1_ADM_s114
                                 + 0.612581 w_k1_AXP_s114
                                 + 0.55389 w_k1_DIS_s114
                                 + 0.674245 w_k1_FDX_s114
                                 + 0.889866 w_k1_IBM_s114
                                 + 0.75284 w_k1_INTC_s114
                                 + 0.744734 w_k1_KO_s114
                                 + 0.742759 w_k1_MAT_s114
                                 + 0.697758 w_k1_MSFT_s114
                                 + 0.802772 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.846226 w_k1_ADM_s114
                                 + 1.23463 w_k1_AXP_s114
                                 + 0.925656 w_k1_DIS_s114
                                 + 1.01805 w_k1_FDX_s114
                                 + 1.05118 w_k1_IBM_s114
                                 + 0.897807 w_k1_INTC_s114
                                 + 0.988316 w_k1_KO_s114
                                 + 1.29168 w_k1_MAT_s114
                                 + 0.944362 w_k1_MSFT_s114
                                 + 0.946705 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.85006 w_k1_ADM_s114
                                 + 0.867629 w_k1_AXP_s114
                                 + 0.802871 w_k1_DIS_s114
                                 + 0.930529 w_k1_FDX_s114
                                 + 0.758161 w_k1_IBM_s114
                                 + 0.758372 w_k1_INTC_s114
                                 + 0.765186 w_k1_KO_s114
                                 + 0.697641 w_k1_MAT_s114
                                 + 0.774369 w_k1_MSFT_s114
                                 + 0.842904 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.01137 w_k1_ADM_s114
                                 + 1.09139 w_k1_AXP_s114
                                 + 1.35967 w_k1_DIS_s114
                                 + 1.13133 w_k1_FDX_s114
                                 + 1.23227 w_k1_IBM_s114
                                 + 1.14976 w_k1_INTC_s114
                                 + 1.28971 w_k1_KO_s114 + 1.03912 w_k1_MAT_s114
                                 + 1.27151 w_k1_MSFT_s114
                                 + 1.01986 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.820428 w_k1_ADM_s115
                                 + 0.647719 w_k1_AXP_s115
                                 + 0.646542 w_k1_DIS_s115
                                 + 0.791317 w_k1_FDX_s115
                                 + 0.705648 w_k1_IBM_s115
                                 + 0.842336 w_k1_INTC_s115
                                 + 0.820453 w_k1_KO_s115
                                 + 0.620922 w_k1_MAT_s115
                                 + 0.911125 w_k1_MSFT_s115
                                 + 0.826498 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.817832 w_k1_ADM_s115
                                 + 0.842008 w_k1_AXP_s115
                                 + 0.781202 w_k1_DIS_s115
                                 + 0.884091 w_k1_FDX_s115
                                 + 0.815379 w_k1_IBM_s115
                                 + 1.22549 w_k1_INTC_s115
                                 + 1.00227 w_k1_KO_s115
                                 + 0.667533 w_k1_MAT_s115
                                 + 0.870422 w_k1_MSFT_s115
                                 + 0.965577 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.29772 w_k1_ADM_s115
                                 + 1.2151 w_k1_AXP_s115 + 1.38433 w_k1_DIS_s115
                                 + 1.20394 w_k1_FDX_s115
                                 + 1.06551 w_k1_IBM_s115
                                 + 1.05777 w_k1_INTC_s115
                                 + 1.15537 w_k1_KO_s115 + 1.06386 w_k1_MAT_s115
                                 + 1.06984 w_k1_MSFT_s115
                                 + 1.14796 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.962689 w_k1_ADM_s115
                                 + 0.885705 w_k1_AXP_s115
                                 + 0.873683 w_k1_DIS_s115
                                 + 0.891028 w_k1_FDX_s115
                                 + 1.23388 w_k1_IBM_s115
                                 + 1.08654 w_k1_INTC_s115
                                 + 1.19445 w_k1_KO_s115 + 1.18974 w_k1_MAT_s115
                                 + 0.859221 w_k1_MSFT_s115
                                 + 1.01626 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04584 w_k1_ADM_s115
                                 + 0.799997 w_k1_AXP_s115
                                 + 1.19625 w_k1_DIS_s115
                                 + 1.00111 w_k1_FDX_s115
                                 + 0.949184 w_k1_IBM_s115
                                 + 1.18694 w_k1_INTC_s115
                                 + 1.27867 w_k1_KO_s115
                                 + 0.637922 w_k1_MAT_s115
                                 + 1.11027 w_k1_MSFT_s115
                                 + 0.952672 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.871174 w_k1_ADM_s115
                                 + 1.13937 w_k1_AXP_s115
                                 + 1.24709 w_k1_DIS_s115
                                 + 0.981088 w_k1_FDX_s115
                                 + 1.51538 w_k1_IBM_s115
                                 + 1.18004 w_k1_INTC_s115
                                 + 1.45643 w_k1_KO_s115 + 1.43491 w_k1_MAT_s115
                                 + 1.09236 w_k1_MSFT_s115
                                 + 1.12147 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.21836 w_k1_ADM_s115
                                 + 1.36869 w_k1_AXP_s115
                                 + 0.931806 w_k1_DIS_s115
                                 + 0.961889 w_k1_FDX_s115
                                 + 1.07192 w_k1_IBM_s115
                                 + 1.02009 w_k1_INTC_s115
                                 + 1.17627 w_k1_KO_s115 + 1.08279 w_k1_MAT_s115
                                 + 1.01782 w_k1_MSFT_s115
                                 + 1.09194 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.850877 w_k1_ADM_s115
                                 + 0.923082 w_k1_AXP_s115
                                 + 0.982126 w_k1_DIS_s115
                                 + 0.947745 w_k1_FDX_s115
                                 + 1.00448 w_k1_IBM_s115
                                 + 1.22768 w_k1_INTC_s115
                                 + 1.01846 w_k1_KO_s115
                                 + 0.832005 w_k1_MAT_s115
                                 + 1.01921 w_k1_MSFT_s115
                                 + 0.941419 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.40867 w_k1_ADM_s116
                                 + 1.42865 w_k1_AXP_s116
                                 + 1.31375 w_k1_DIS_s116
                                 + 1.14523 w_k1_FDX_s116
                                 + 1.28404 w_k1_IBM_s116
                                 + 0.863383 w_k1_INTC_s116
                                 + 1.36862 w_k1_KO_s116 + 1.25024 w_k1_MAT_s116
                                 + 0.98813 w_k1_MSFT_s116
                                 + 1.21165 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.801241 w_k1_ADM_s116
                                 + 1.48481 w_k1_AXP_s116
                                 + 0.972828 w_k1_DIS_s116
                                 + 0.938511 w_k1_FDX_s116
                                 + 1.61946 w_k1_IBM_s116
                                 + 0.961577 w_k1_INTC_s116
                                 + 1.29333 w_k1_KO_s116 + 1.75844 w_k1_MAT_s116
                                 + 1.00877 w_k1_MSFT_s116
                                 + 1.10155 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.60942 w_k1_ADM_s116
                                 + 1.39356 w_k1_AXP_s116
                                 + 2.23332 w_k1_DIS_s116
                                 + 1.46203 w_k1_FDX_s116
                                 + 1.36966 w_k1_IBM_s116
                                 + 1.32648 w_k1_INTC_s116
                                 + 1.72957 w_k1_KO_s116 + 1.63206 w_k1_MAT_s116
                                 + 1.3305 w_k1_MSFT_s116 + 1.28405 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.894973 w_k1_ADM_s116
                                 + 0.888892 w_k1_AXP_s116
                                 + 1.06902 w_k1_DIS_s116
                                 + 0.960291 w_k1_FDX_s116
                                 + 0.8884 w_k1_IBM_s116
                                 + 0.999017 w_k1_INTC_s116
                                 + 1.09814 w_k1_KO_s116 + 1.05118 w_k1_MAT_s116
                                 + 1.23343 w_k1_MSFT_s116
                                 + 0.986185 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.28048 w_k1_ADM_s116
                                 + 1.22121 w_k1_AXP_s116
                                 + 1.31454 w_k1_DIS_s116
                                 + 1.22263 w_k1_FDX_s116
                                 + 0.785832 w_k1_IBM_s116
                                 + 0.993047 w_k1_INTC_s116
                                 + 1.06927 w_k1_KO_s116
                                 + 0.745801 w_k1_MAT_s116
                                 + 1.12464 w_k1_MSFT_s116
                                 + 0.97653 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.20304 w_k1_ADM_s116
                                 + 1.41808 w_k1_AXP_s116
                                 + 1.30465 w_k1_DIS_s116
                                 + 1.17381 w_k1_FDX_s116
                                 + 1.00454 w_k1_IBM_s116
                                 + 0.955331 w_k1_INTC_s116
                                 + 1.09341 w_k1_KO_s116
                                 + 0.906684 w_k1_MAT_s116
                                 + 1.13209 w_k1_MSFT_s116
                                 + 1.18478 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.981861 w_k1_ADM_s116
                                 + 0.923937 w_k1_AXP_s116
                                 + 0.796665 w_k1_DIS_s116
                                 + 0.821792 w_k1_FDX_s116
                                 + 0.697059 w_k1_IBM_s116
                                 + 0.631372 w_k1_INTC_s116
                                 + 0.857546 w_k1_KO_s116
                                 + 0.921802 w_k1_MAT_s116
                                 + 0.853851 w_k1_MSFT_s116
                                 + 0.880059 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.695301 w_k1_ADM_s116
                                 + 0.962912 w_k1_AXP_s116
                                 + 0.881809 w_k1_DIS_s116
                                 + 0.820511 w_k1_FDX_s116
                                 + 1.10968 w_k1_IBM_s116
                                 + 1.00843 w_k1_INTC_s116
                                 + 0.93939 w_k1_KO_s116 + 1.1702 w_k1_MAT_s116
                                 + 1.01025 w_k1_MSFT_s116
                                 + 0.910407 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.14901 w_k1_ADM_s117
                                 + 1.1997 w_k1_AXP_s117 + 1.17402 w_k1_DIS_s117
                                 + 1.09253 w_k1_FDX_s117
                                 + 0.938333 w_k1_IBM_s117
                                 + 1.01881 w_k1_INTC_s117
                                 + 1.24608 w_k1_KO_s117 + 1.02328 w_k1_MAT_s117
                                 + 1.02678 w_k1_MSFT_s117
                                 + 1.21357 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.40509 w_k1_ADM_s117
                                 + 1.27037 w_k1_AXP_s117
                                 + 1.80289 w_k1_DIS_s117
                                 + 1.36185 w_k1_FDX_s117
                                 + 1.10372 w_k1_IBM_s117
                                 + 1.20429 w_k1_INTC_s117 + 1.349 w_k1_KO_s117
                                 + 1.10625 w_k1_MAT_s117
                                 + 1.37658 w_k1_MSFT_s117
                                 + 1.22986 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.04085 w_k1_ADM_s117
                                 + 1.09853 w_k1_AXP_s117
                                 + 1.55903 w_k1_DIS_s117
                                 + 1.12547 w_k1_FDX_s117
                                 + 1.59825 w_k1_IBM_s117
                                 + 1.23043 w_k1_INTC_s117
                                 + 1.50207 w_k1_KO_s117 + 1.30881 w_k1_MAT_s117
                                 + 1.32568 w_k1_MSFT_s117
                                 + 1.17798 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.938362 w_k1_ADM_s117
                                 + 1.24589 w_k1_AXP_s117
                                 + 1.18955 w_k1_DIS_s117
                                 + 0.964424 w_k1_FDX_s117 + 1.41 w_k1_IBM_s117
                                 + 1.00073 w_k1_INTC_s117
                                 + 1.31726 w_k1_KO_s117 + 1.37437 w_k1_MAT_s117
                                 + 0.961357 w_k1_MSFT_s117
                                 + 1.18304 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.817762 w_k1_ADM_s117
                                 + 0.714872 w_k1_AXP_s117
                                 + 0.700997 w_k1_DIS_s117
                                 + 0.792941 w_k1_FDX_s117
                                 + 0.790896 w_k1_IBM_s117
                                 + 0.888428 w_k1_INTC_s117
                                 + 0.888191 w_k1_KO_s117
                                 + 0.591519 w_k1_MAT_s117
                                 + 0.804504 w_k1_MSFT_s117
                                 + 0.939718 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.15813 w_k1_ADM_s117
                                 + 1.64534 w_k1_AXP_s117
                                 + 1.40544 w_k1_DIS_s117
                                 + 1.19219 w_k1_FDX_s117
                                 + 1.28333 w_k1_IBM_s117
                                 + 1.10778 w_k1_INTC_s117
                                 + 1.31577 w_k1_KO_s117 + 1.43914 w_k1_MAT_s117
                                 + 1.04178 w_k1_MSFT_s117 + 1.1103 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.938662 w_k1_ADM_s117
                                 + 0.942497 w_k1_AXP_s117
                                 + 1.06208 w_k1_DIS_s117
                                 + 0.889354 w_k1_FDX_s117
                                 + 1.08038 w_k1_IBM_s117
                                 + 0.896758 w_k1_INTC_s117
                                 + 1.1093 w_k1_KO_s117 + 0.977637 w_k1_MAT_s117
                                 + 0.930062 w_k1_MSFT_s117
                                 + 1.17452 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.922724 w_k1_ADM_s117
                                 + 0.95783 w_k1_AXP_s117
                                 + 0.910335 w_k1_DIS_s117
                                 + 0.8882 w_k1_FDX_s117
                                 + 0.951792 w_k1_IBM_s117
                                 + 1.03416 w_k1_INTC_s117
                                 + 1.06355 w_k1_KO_s117
                                 + 0.826375 w_k1_MAT_s117
                                 + 0.93447 w_k1_MSFT_s117
                                 + 0.964707 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0211142810006998 w_k1_ADM_s000
                                 + 0.0523540552399412 w_k1_ADM_s001
                                 + 0.0835989731467625 w_k1_ADM_s002
                                 + 0.0873904945609725 w_k1_ADM_s003
                                 + 0.0367620771203895 w_k1_ADM_s004
                                 + 0.0550202174380615 w_k1_ADM_s005
                                 + 0.0693219157077201 w_k1_ADM_s006
                                 + 0.0209389659259318 w_k1_ADM_s007
                                 + 0.0205712762976718 w_k1_ADM_s010
                                 + 0.0259289134734726 w_k1_ADM_s011
                                 + 0.0237836802794076 w_k1_ADM_s012
                                 + 0.0100351484694532 w_k1_ADM_s013
                                 + 0.0517822861870968 w_k1_ADM_s014
                                 + 0.0161515826950034 w_k1_ADM_s015
                                 + 0.00877442654718223 w_k1_ADM_s016
                                 + 0.01802585714333 w_k1_ADM_s017
                                 + 0.0251887467096865 w_k1_ADM_s100
                                 + 0.0126742104375081 w_k1_ADM_s101
                                 + 0.0161303200770151 w_k1_ADM_s102
                                 + 0.00844535282221793 w_k1_ADM_s103
                                 + 0.00767201840110054 w_k1_ADM_s104
                                 + 0.0159808590785703 w_k1_ADM_s105
                                 + 0.0091168785481711 w_k1_ADM_s106
                                 + 0.0190814361473365 w_k1_ADM_s107
                                 + 0.0377712190939417 w_k1_ADM_s110
                                 + 0.0458031842257478 w_k1_ADM_s111
                                 + 0.0601459219168903 w_k1_ADM_s112
                                 + 0.021553089599492 w_k1_ADM_s113
                                 + 0.0319583835847502 w_k1_ADM_s114
                                 + 0.0210812149226107 w_k1_ADM_s115
                                 + 0.0154561039294854 w_k1_ADM_s116
                                 + 0.0417199293740634 w_k1_ADM_s117
                                 + 0.0206446652122587 w_k1_AXP_s000
                                 + 0.0504019190690257 w_k1_AXP_s001
                                 + 0.0784168582369887 w_k1_AXP_s002
                                 + 0.109563391578471 w_k1_AXP_s003
                                 + 0.0411756606107524 w_k1_AXP_s004
                                 + 0.0521693949971881 w_k1_AXP_s005
                                 + 0.0837274472753736 w_k1_AXP_s006
                                 + 0.0235318079362746 w_k1_AXP_s007
                                 + 0.0193642874783135 w_k1_AXP_s010
                                 + 0.0255825804526668 w_k1_AXP_s011
                                 + 0.0260762470871225 w_k1_AXP_s012
                                 + 0.010374995982428 w_k1_AXP_s013
                                 + 0.0456662109957725 w_k1_AXP_s014
                                 + 0.0155148289187851 w_k1_AXP_s015
                                 + 0.0107806807274497 w_k1_AXP_s016
                                 + 0.0164099592344075 w_k1_AXP_s017
                                 + 0.0233043588426464 w_k1_AXP_s100
                                 + 0.0112047918186328 w_k1_AXP_s101
                                 + 0.0159742073651415 w_k1_AXP_s102
                                 + 0.00890685367620703 w_k1_AXP_s103
                                 + 0.00708047599075154 w_k1_AXP_s104
                                 + 0.0171925776639547 w_k1_AXP_s105
                                 + 0.00961077228425643 w_k1_AXP_s106
                                 + 0.0163486207991017 w_k1_AXP_s107
                                 + 0.0431702232311963 w_k1_AXP_s110
                                 + 0.0419908598833784 w_k1_AXP_s111
                                 + 0.0610283678884567 w_k1_AXP_s112
                                 + 0.0208603057572598 w_k1_AXP_s113
                                 + 0.0382754846753832 w_k1_AXP_s114
                                 + 0.0204783372735731 w_k1_AXP_s115
                                 + 0.0172021498242265 w_k1_AXP_s116
                                 + 0.043406927735305 w_k1_AXP_s117
                                 + 0.0229286297188042 w_k1_DIS_s000
                                 + 0.0519200671835122 w_k1_DIS_s001
                                 + 0.0840306387689148 w_k1_DIS_s002
                                 + 0.103559885430025 w_k1_DIS_s003
                                 + 0.0399847166756147 w_k1_DIS_s004
                                 + 0.0549579933190104 w_k1_DIS_s005
                                 + 0.0807028945549146 w_k1_DIS_s006
                                 + 0.0266481328008195 w_k1_DIS_s007
                                 + 0.0213561445564073 w_k1_DIS_s010
                                 + 0.0261855190344247 w_k1_DIS_s011
                                 + 0.0258320439517565 w_k1_DIS_s012
                                 + 0.0112478782361355 w_k1_DIS_s013
                                 + 0.0530521273612686 w_k1_DIS_s014
                                 + 0.0163000493842392 w_k1_DIS_s015
                                 + 0.0106487756525917 w_k1_DIS_s016
                                 + 0.016525931694034 w_k1_DIS_s017
                                 + 0.0251079338688407 w_k1_DIS_s100
                                 + 0.0126700961858075 w_k1_DIS_s101
                                 + 0.0165237428387879 w_k1_DIS_s102
                                 + 0.00920424469849769 w_k1_DIS_s103
                                 + 0.00836236374507514 w_k1_DIS_s104
                                 + 0.0171778483543134 w_k1_DIS_s105
                                 + 0.00881861804328966 w_k1_DIS_s106
                                 + 0.0168267035013683 w_k1_DIS_s107
                                 + 0.0382305376433552 w_k1_DIS_s110
                                 + 0.0437868070498148 w_k1_DIS_s111
                                 + 0.0662646599230218 w_k1_DIS_s112
                                 + 0.019178795017372 w_k1_DIS_s113
                                 + 0.0372735698785744 w_k1_DIS_s114
                                 + 0.021781150348729 w_k1_DIS_s115
                                 + 0.0166885311166862 w_k1_DIS_s116
                                 + 0.0452535822977278 w_k1_DIS_s117
                                 + 0.0220807960731769 w_k1_FDX_s000
                                 + 0.0516991955568362 w_k1_FDX_s001
                                 + 0.081603876345533 w_k1_FDX_s002
                                 + 0.0947773237015815 w_k1_FDX_s003
                                 + 0.0402245788547972 w_k1_FDX_s004
                                 + 0.0539574073626302 w_k1_FDX_s005
                                 + 0.0820600631651408 w_k1_FDX_s006
                                 + 0.0238638554189294 w_k1_FDX_s007
                                 + 0.0206125804496364 w_k1_FDX_s010
                                 + 0.025075779800912 w_k1_FDX_s011
                                 + 0.0259208164245676 w_k1_FDX_s012
                                 + 0.0112004743341859 w_k1_FDX_s013
                                 + 0.0503057979954261 w_k1_FDX_s014
                                 + 0.016370341994741 w_k1_FDX_s015
                                 + 0.0102554835103183 w_k1_FDX_s016
                                 + 0.0161619311031776 w_k1_FDX_s017
                                 + 0.02414794271048 w_k1_FDX_s100
                                 + 0.0119803862288662 w_k1_FDX_s101
                                 + 0.0159469540123269 w_k1_FDX_s102
                                 + 0.00907631172464829 w_k1_FDX_s103
                                 + 0.00766639210866833 w_k1_FDX_s104
                                 + 0.0171524702908815 w_k1_FDX_s105
                                 + 0.00977356470428925 w_k1_FDX_s106
                                 + 0.0156849151933689 w_k1_FDX_s107
                                 + 0.0377742830692574 w_k1_FDX_s110
                                 + 0.0432030853799987 w_k1_FDX_s111
                                 + 0.0621696537324409 w_k1_FDX_s112
                                 + 0.0193716653870681 w_k1_FDX_s113
                                 + 0.0362716189292433 w_k1_FDX_s114
                                 + 0.0205114439021663 w_k1_FDX_s115
                                 + 0.0151061397820704 w_k1_FDX_s116
                                 + 0.0407915558193087 w_k1_FDX_s117
                                 + 0.0219706750107044 w_k1_IBM_s000
                                 + 0.0526509935830958 w_k1_IBM_s001
                                 + 0.0830291489188306 w_k1_IBM_s002
                                 + 0.0945426477927796 w_k1_IBM_s003
                                 + 0.038656934060216 w_k1_IBM_s004
                                 + 0.0537694314749093 w_k1_IBM_s005
                                 + 0.0806798329776378 w_k1_IBM_s006
                                 + 0.0235715175282919 w_k1_IBM_s007
                                 + 0.0206648168070684 w_k1_IBM_s010
                                 + 0.0270780147042526 w_k1_IBM_s011
                                 + 0.0272788724166535 w_k1_IBM_s012
                                 + 0.0115300252795014 w_k1_IBM_s013
                                 + 0.0508497214573777 w_k1_IBM_s014
                                 + 0.0161332375210222 w_k1_IBM_s015
                                 + 0.0107533061128486 w_k1_IBM_s016
                                 + 0.0170680090940893 w_k1_IBM_s017
                                 + 0.0235167133161173 w_k1_IBM_s100
                                 + 0.0117796420320912 w_k1_IBM_s101
                                 + 0.0160636016483136 w_k1_IBM_s102
                                 + 0.0091789029851603 w_k1_IBM_s103
                                 + 0.00781233036350396 w_k1_IBM_s104
                                 + 0.0171577551189951 w_k1_IBM_s105
                                 + 0.00950025999301495 w_k1_IBM_s106
                                 + 0.0146881449893185 w_k1_IBM_s107
                                 + 0.0373489349436963 w_k1_IBM_s110
                                 + 0.0448092120543309 w_k1_IBM_s111
                                 + 0.0645281445517974 w_k1_IBM_s112
                                 + 0.0193545687456633 w_k1_IBM_s113
                                 + 0.0387201228108124 w_k1_IBM_s114
                                 + 0.0220337750833822 w_k1_IBM_s115
                                 + 0.0156001372914172 w_k1_IBM_s116
                                 + 0.0438879695327028 w_k1_IBM_s117
                                 + 0.019022104281276 w_k1_INTC_s000
                                 + 0.0509816202433448 w_k1_INTC_s001
                                 + 0.0727392413992149 w_k1_INTC_s002
                                 + 0.0980109450675598 w_k1_INTC_s003
                                 + 0.0381519235058928 w_k1_INTC_s004
                                 + 0.0571758817226466 w_k1_INTC_s005
                                 + 0.0774429053264218 w_k1_INTC_s006
                                 + 0.0257297891614535 w_k1_INTC_s007
                                 + 0.0235141515526511 w_k1_INTC_s010
                                 + 0.033347529748309 w_k1_INTC_s011
                                 + 0.0234731252649432 w_k1_INTC_s012
                                 + 0.0103872853386045 w_k1_INTC_s013
                                 + 0.0456571356185583 w_k1_INTC_s014
                                 + 0.0155967277654686 w_k1_INTC_s015
                                 + 0.0110934793584104 w_k1_INTC_s016
                                 + 0.0172282390845475 w_k1_INTC_s017
                                 + 0.0234328737003084 w_k1_INTC_s100
                                 + 0.00996272349116465 w_k1_INTC_s101
                                 + 0.0168478618501965 w_k1_INTC_s102
                                 + 0.00928772088604883 w_k1_INTC_s103
                                 + 0.00779469828441811 w_k1_INTC_s104
                                 + 0.0170141453560878 w_k1_INTC_s105
                                 + 0.0102566375431295 w_k1_INTC_s106
                                 + 0.0151052888248515 w_k1_INTC_s107
                                 + 0.0363702949951165 w_k1_INTC_s110
                                 + 0.047389469777475 w_k1_INTC_s111
                                 + 0.0616978767266816 w_k1_INTC_s112
                                 + 0.0184418904683334 w_k1_INTC_s113
                                 + 0.0341711135670643 w_k1_INTC_s114
                                 + 0.0231193627575992 w_k1_INTC_s115
                                 + 0.0140465334210413 w_k1_INTC_s116
                                 + 0.0422454919910083 w_k1_INTC_s117
                                 + 0.0210339553925514 w_k1_KO_s000
                                 + 0.0531143345832429 w_k1_KO_s001
                                 + 0.0801302613533603 w_k1_KO_s002
                                 + 0.0976471380331816 w_k1_KO_s003
                                 + 0.047478339166031 w_k1_KO_s004
                                 + 0.0504432403390917 w_k1_KO_s005
                                 + 0.0823650805346517 w_k1_KO_s006
                                 + 0.0240733198572196 w_k1_KO_s007
                                 + 0.0211223975052776 w_k1_KO_s010
                                 + 0.0282976069005248 w_k1_KO_s011
                                 + 0.0254616613717602 w_k1_KO_s012
                                 + 0.0111724093598363 w_k1_KO_s013
                                 + 0.0476768995065075 w_k1_KO_s014
                                 + 0.0151055228273104 w_k1_KO_s015
                                 + 0.00995052911217149 w_k1_KO_s016
                                 + 0.0160754686262955 w_k1_KO_s017
                                 + 0.0240563037372778 w_k1_KO_s100
                                 + 0.0116533309724168 w_k1_KO_s101
                                 + 0.0169203624658632 w_k1_KO_s102
                                 + 0.00896187892512494 w_k1_KO_s103
                                 + 0.00826225254493042 w_k1_KO_s104
                                 + 0.0164362216212725 w_k1_KO_s105
                                 + 0.00906416237452667 w_k1_KO_s106
                                 + 0.0162376777908865 w_k1_KO_s107
                                 + 0.0370057798497739 w_k1_KO_s110
                                 + 0.0455251363196121 w_k1_KO_s111
                                 + 0.055698798541725 w_k1_KO_s112
                                 + 0.0217127164734798 w_k1_KO_s113
                                 + 0.0370521906834138 w_k1_KO_s114
                                 + 0.0240007200567077 w_k1_KO_s115
                                 + 0.0165509798042373 w_k1_KO_s116
                                 + 0.0481949466808413 w_k1_KO_s117
                                 + 0.0241665188316046 w_k1_MAT_s000
                                 + 0.056260596599547 w_k1_MAT_s001
                                 + 0.0810338585495497 w_k1_MAT_s002
                                 + 0.106807278560903 w_k1_MAT_s003
                                 + 0.0500866397486888 w_k1_MAT_s004
                                 + 0.0448007248416117 w_k1_MAT_s005
                                 + 0.0852175377233888 w_k1_MAT_s006
                                 + 0.0220291463963088 w_k1_MAT_s007
                                 + 0.0237397453132055 w_k1_MAT_s010
                                 + 0.025805882497673 w_k1_MAT_s011
                                 + 0.0276258559136576 w_k1_MAT_s012
                                 + 0.0110305083985244 w_k1_MAT_s013
                                 + 0.0511945799337 w_k1_MAT_s014
                                 + 0.0180403158406778 w_k1_MAT_s015
                                 + 0.0107945344699765 w_k1_MAT_s016
                                 + 0.0147868414192659 w_k1_MAT_s017
                                 + 0.0227894319619691 w_k1_MAT_s100
                                 + 0.0111292899771764 w_k1_MAT_s101
                                 + 0.0145445473911173 w_k1_MAT_s102
                                 + 0.0085735880646749 w_k1_MAT_s103
                                 + 0.0100158451398273 w_k1_MAT_s104
                                 + 0.0174984212861799 w_k1_MAT_s105
                                 + 0.00965333150281656 w_k1_MAT_s106
                                 + 0.0139680233887267 w_k1_MAT_s107
                                 + 0.0449939531709614 w_k1_MAT_s110
                                 + 0.045898645959334 w_k1_MAT_s111
                                 + 0.0760322233177721 w_k1_MAT_s112
                                 + 0.0191071062255454 w_k1_MAT_s113
                                 + 0.0376366220817884 w_k1_MAT_s114
                                 + 0.0199846150946756 w_k1_MAT_s115
                                 + 0.0169656963668903 w_k1_MAT_s116
                                 + 0.0403348825053794 w_k1_MAT_s117
                                 + 0.0196953261245576 w_k1_MSFT_s000
                                 + 0.0533129478220238 w_k1_MSFT_s001
                                 + 0.0771293103509305 w_k1_MSFT_s002
                                 + 0.0925378004846979 w_k1_MSFT_s003
                                 + 0.0422303037624977 w_k1_MSFT_s004
                                 + 0.0555814042719547 w_k1_MSFT_s005
                                 + 0.0836374493100565 w_k1_MSFT_s006
                                 + 0.0247900128732555 w_k1_MSFT_s007
                                 + 0.0207818298633055 w_k1_MSFT_s010
                                 + 0.0280899979021469 w_k1_MSFT_s011
                                 + 0.0260515592202005 w_k1_MSFT_s012
                                 + 0.0101162124113681 w_k1_MSFT_s013
                                 + 0.0486885665072591 w_k1_MSFT_s014
                                 + 0.0144056891651541 w_k1_MSFT_s015
                                 + 0.0108303892258439 w_k1_MSFT_s016
                                 + 0.0157563645123553 w_k1_MSFT_s017
                                 + 0.0234447782179469 w_k1_MSFT_s100
                                 + 0.0116280908185467 w_k1_MSFT_s101
                                 + 0.0162704614988649 w_k1_MSFT_s102
                                 + 0.00944146951360123 w_k1_MSFT_s103
                                 + 0.00750626654872946 w_k1_MSFT_s104
                                 + 0.0163788578236539 w_k1_MSFT_s105
                                 + 0.0095818070057543 w_k1_MSFT_s106
                                 + 0.0154960934429576 w_k1_MSFT_s107
                                 + 0.0371310304253681 w_k1_MSFT_s110
                                 + 0.0474934260115913 w_k1_MSFT_s111
                                 + 0.0570318049387555 w_k1_MSFT_s112
                                 + 0.0198033378928912 w_k1_MSFT_s113
                                 + 0.0349335892963352 w_k1_MSFT_s114
                                 + 0.0208124380919302 w_k1_MSFT_s115
                                 + 0.0160222094551998 w_k1_MSFT_s116
                                 + 0.0412809253725297 w_k1_MSFT_s117
                                 + 0.0216996296370568 w_k1_PG_s000
                                 + 0.0456710911746684 w_k1_PG_s001
                                 + 0.0822655852583993 w_k1_PG_s002
                                 + 0.095271965609436 w_k1_PG_s003
                                 + 0.0395583777028781 w_k1_PG_s004
                                 + 0.0531060520311629 w_k1_PG_s005
                                 + 0.0790976066470831 w_k1_PG_s006
                                 + 0.0230944407071632 w_k1_PG_s007
                                 + 0.0194720780004952 w_k1_PG_s010
                                 + 0.0275907349523534 w_k1_PG_s011
                                 + 0.026288422089815 w_k1_PG_s012
                                 + 0.0107230864450813 w_k1_PG_s013
                                 + 0.0470700508463377 w_k1_PG_s014
                                 + 0.014352759088764 w_k1_PG_s015
                                 + 0.00997551892415352 w_k1_PG_s016
                                 + 0.0173893774405329 w_k1_PG_s017
                                 + 0.023694604479271 w_k1_PG_s100
                                 + 0.011617865676373 w_k1_PG_s101
                                 + 0.0153624398033274 w_k1_PG_s102
                                 + 0.00912114284931861 w_k1_PG_s103
                                 + 0.00801328059671808 w_k1_PG_s104
                                 + 0.0168574375207995 w_k1_PG_s105
                                 + 0.00945392201444258 w_k1_PG_s106
                                 + 0.0142300683907519 w_k1_PG_s107
                                 + 0.0388252709180637 w_k1_PG_s110
                                 + 0.0429219139305094 w_k1_PG_s111
                                 + 0.059861112763625 w_k1_PG_s112
                                 + 0.0192237557264932 w_k1_PG_s113
                                 + 0.0342130583593665 w_k1_PG_s114
                                 + 0.021384643964861 w_k1_PG_s115
                                 + 0.0154716734219052 w_k1_PG_s116
                                 + 0.0459822231013341 w_k1_PG_s117 >= 1.1
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
