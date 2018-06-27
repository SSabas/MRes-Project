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
 w_k1_ADM_s0_return_constraint:  0.952121 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.01615 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.30166 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.82077 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.16299 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.899256 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.895202 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.0775 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.25459 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.872025 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.741663 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.977207 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.884794 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.749198 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.86799 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.15973 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.39288 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.05415 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.28616 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.787571 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.12591 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.43522 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.952708 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.858389 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.00127 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.35275 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.928274 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.13787 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.13521 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.749172 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.817426 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.737507 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.15184 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.829582 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.18583 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.819291 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.905227 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.988703 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.00545 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.08967 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.20226 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.778036 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.26231 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.828678 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.0568 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.20531 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.52497 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.913703 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.696822 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.0442 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.00457 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.70401 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.696667 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.949345 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.37861 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.886747 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.03958 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.718053 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.02973 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.48058 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.878554 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.742121 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.35383 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.71509 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.20591 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.096 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.29364 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.663917 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.627235 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.766932 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.33624 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.735007 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.25567 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.02902 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.893528 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.35377 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.982525 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.11937 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.32649 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.709474 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.28212 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.760047 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.983519 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.28893 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.55664 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.01896 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.688614 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.682952 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.775749 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.627967 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.627281 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.00797 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.49191 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.916857 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.14959 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.67602 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.02963 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.98463 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.917258 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.717949 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.11576 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 2.02098 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.27303 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.22113 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.14736 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.640762 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.563239 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.669113 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.36333 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.673161 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.31436 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.941355 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.890643 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.4198 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.02216 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.01975 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.17546 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.819502 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.14952 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.870175 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.00022 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.16361 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.44367 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0873 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.765485 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.940329 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.7946 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.790908 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.760625 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.08921 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.21891 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.870504 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.07421 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.683391 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.928285 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.55266 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.892906 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.896025 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.08112 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.59088 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.0456 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.15954 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.3089 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.716374 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.684686 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.792875 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.26767 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.689291 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.33744 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.934768 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.925663 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12458 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.865428 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.24687 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.18135 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.86317 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.22319 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.808432 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.966606 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.45024 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.935458 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.596126 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.948708 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.726227 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.50293 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.633674 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.690135 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.580457 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.42958 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.96934 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.05274 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.1533 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.69033 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.18769 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.22742 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.738759 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.40404 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.18938 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.86579 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.997188 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.702277 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.00302 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.894317 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.600483 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.18751 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.29477 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.87817 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.25352 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.88809 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.70914 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.998622 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.04109 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.15723 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.907436 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.01266 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.890858 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.961851 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.04549 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.11348 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06675 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.816911 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.818264 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.819041 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.746867 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.716615 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.930017 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.09927 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.963303 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.09186 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.800348 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.876508 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.2174 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.989313 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.04584 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.14499 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.53767 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.20964 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.20053 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.38983 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.842261 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.743832 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.83203 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.18456 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.859082 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.1192 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.04814 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.04882 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.979833 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.920363 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.14729 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.16278 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.837844 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.15288 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.794287 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.10303 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.30314 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.25917 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.849268 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.899437 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.756757 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.06753 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.808666 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.733135 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.786005 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.34279 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.941149 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.05615 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.85778 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.24978 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.39084 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.03823 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.8032 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.20688 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.44262 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.41097 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.10527 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.94956 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.86158 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.776434 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.690813 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.20303 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.960371 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.05929 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.11524 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.981955 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.34556 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.899386 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.17471 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.12167 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.760151 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.23919 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.680304 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.25878 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.53216 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.68041 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.693672 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.07235 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.919119 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.46636 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.685804 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.711371 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.724298 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.43007 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.769376 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.08292 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.00195 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.88625 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.42119 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.31389 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.742601 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.9319 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.21326 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 2.29258 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.34479 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.884127 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.01804 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.99389 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.830093 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.73612 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.21294 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.30392 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.47747 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.11467 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.23011 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.983284 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.995416 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.20276 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.824104 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15719 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.936248 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.01862 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.083 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.3278 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.0638 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.798343 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.801757 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.752547 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.799947 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.763943 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.06935 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.35907 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.94554 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.15806 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.740694 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.00855 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.46976 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.888675 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.882409 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.14312 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.38134 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.05621 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.14173 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.29257 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.798395 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.715267 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.785847 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.19583 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.774826 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.24059 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.919798 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.884522 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.11043 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.00515 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.03347 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.05936 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.854776 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.19985 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.876105 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.03579 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.24274 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.15109 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0206 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.860407 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.901811 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.986165 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.814572 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.860207 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.893596 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.20334 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.980918 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.05991 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.85859 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.07045 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.42061 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.885102 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.815199 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.11139 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.20845 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.01921 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.930128 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.996561 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.817683 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.831102 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.80939 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.22438 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.912929 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.16108 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.05311 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.0044 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.29481 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.382999622579379 z_k1_s0000
                                 - 0.804998702289065 z_k1_s0001
                                 - 0.83999864586685 z_k1_s0002
                                 - 0.864997005567848 z_k1_s0003
                                 - 1.20999564940731 z_k1_s0004
                                 - 0.392999526458876 z_k1_s0005
                                 - 0.460999096839089 z_k1_s0006
                                 - 0.642996563447184 z_k1_s0007
                                 - 0.971004065492154 z_k1_s0010
                                 - 0.919003177650248 z_k1_s0011
                                 - 1.06600464474035 z_k1_s0012
                                 - 0.848003529560575 z_k1_s0013
                                 - 0.386000855280073 z_k1_s0014
                                 - 1.3330039896137 z_k1_s0015
                                 - 0.785998299897155 z_k1_s0016
                                 - 0.469000968949007 z_k1_s0017
                                 - 0.392000033061355 z_k1_s0020
                                 - 0.317999983860976 z_k1_s0021
                                 - 0.638000560727664 z_k1_s0022
                                 - 0.881999653387837 z_k1_s0023
                                 - 0.272000237736558 z_k1_s0024
                                 - 0.361000103477458 z_k1_s0025
                                 - 0.71999877194878 z_k1_s0026
                                 - 0.626999825197512 z_k1_s0027
                                 - 0.497999440881754 z_k1_s0030
                                 - 0.182999973998721 z_k1_s0031
                                 - 0.23300010689127 z_k1_s0032
                                 - 0.342999088494851 z_k1_s0033
                                 - 0.582001284717273 z_k1_s0034
                                 - 0.423000135242949 z_k1_s0035
                                 - 0.138000092715431 z_k1_s0036
                                 - 0.579001431651723 z_k1_s0037
                                 - 0.403000986234421 z_k1_s0040
                                 - 0.413001529784054 z_k1_s0041
                                 - 0.668001894289509 z_k1_s0042
                                 - 0.624001527072651 z_k1_s0043
                                 - 0.223000312147903 z_k1_s0044
                                 - 0.346999291957494 z_k1_s0045
                                 - 0.220000149083013 z_k1_s0046
                                 - 0.476000905343685 z_k1_s0047
                                 - 0.174999379649747 z_k1_s0050
                                 - 0.192000164574774 z_k1_s0051
                                 - 0.113999891291504 z_k1_s0052
                                 - 0.316999516324824 z_k1_s0053
                                 - 0.242000192474472 z_k1_s0054
                                 - 0.160999831357316 z_k1_s0055
                                 - 0.112999862013542 z_k1_s0056
                                 - 0.12199983831552 z_k1_s0057
                                 - 0.165999969940044 z_k1_s0060
                                 - 0.113000114029108 z_k1_s0061
                                 - 0.34399976446365 z_k1_s0062
                                 - 0.131999962419789 z_k1_s0063
                                 - 0.1990006227452 z_k1_s0064
                                 - 0.182000051384894 z_k1_s0065
                                 - 0.54300038720885 z_k1_s0066
                                 - 0.212999535159078 z_k1_s0067
                                 - 0.326000293184312 z_k1_s0070
                                 - 0.259000132828561 z_k1_s0071
                                 - 0.468998970888541 z_k1_s0072
                                 - 0.225000163851105 z_k1_s0073
                                 - 0.567000843225128 z_k1_s0074
                                 - 0.680999874348847 z_k1_s0075
                                 - 0.707998889930177 z_k1_s0076
                                 - 0.351000112167609 z_k1_s0077
                                 - 0.791000592015076 z_k1_s0100
                                 - 0.456002941604274 z_k1_s0101
                                 - 0.449000202817265 z_k1_s0102
                                 - 0.310000743809703 z_k1_s0103
                                 - 0.247000805999978 z_k1_s0104
                                 - 0.195000655442144 z_k1_s0105
                                 - 0.346001071673483 z_k1_s0106
                                 - 0.840001593470202 z_k1_s0107
                                 - 0.285999549577624 z_k1_s0110
                                 - 0.443998705539371 z_k1_s0111
                                 - 0.367000133716975 z_k1_s0112
                                 - 0.661000079032667 z_k1_s0113
                                 - 0.788997833451566 z_k1_s0114
                                 - 0.187999897705349 z_k1_s0115
                                 - 0.255999797532888 z_k1_s0116
                                 - 0.307999565683537 z_k1_s0117
                                 - 0.354001638714373 z_k1_s0120
                                 - 0.353002584795181 z_k1_s0121
                                 - 0.318001247488302 z_k1_s0122
                                 - 0.411000712316256 z_k1_s0123
                                 - 0.350000123016812 z_k1_s0124
                                 - 0.289002183727764 z_k1_s0125
                                 - 0.287001425878984 z_k1_s0126
                                 - 0.288000479798176 z_k1_s0127
                                 - 0.204999964158649 z_k1_s0130
                                 - 0.56099955185899 z_k1_s0131
                                 - 0.360998780240693 z_k1_s0132
                                 - 0.411000626645479 z_k1_s0133
                                 - 0.406999156132926 z_k1_s0134
                                 - 0.367999700137447 z_k1_s0135
                                 - 0.522998809991437 z_k1_s0136
                                 - 0.470999535330799 z_k1_s0137
                                 - 0.628999653283965 z_k1_s0140
                                 - 0.58099935893324 z_k1_s0141
                                 - 0.84300271604237 z_k1_s0142
                                 - 0.745999380758216 z_k1_s0143
                                 - 1.04200245853691 z_k1_s0144
                                 - 1.85999849297355 z_k1_s0145
                                 - 0.786997666519421 z_k1_s0146
                                 - 0.398999476773481 z_k1_s0147
                                 - 0.14000006608991 z_k1_s0150
                                 - 0.398000581461415 z_k1_s0151
                                 - 0.169999992137725 z_k1_s0152
                                 - 0.47799956185538 z_k1_s0153
                                 - 0.220999496338917 z_k1_s0154
                                 - 0.125999936120887 z_k1_s0155
                                 - 0.406000479500812 z_k1_s0156
                                 - 0.116999999426553 z_k1_s0157
                                 - 0.454999036550241 z_k1_s0160
                                 - 0.329000400952023 z_k1_s0161
                                 - 0.58599939756159 z_k1_s0162
                                 - 0.165000118217252 z_k1_s0163
                                 - 0.154000144309445 z_k1_s0164
                                 - 0.365999919405445 z_k1_s0165
                                 - 0.142999915601566 z_k1_s0166
                                 - 0.350001022885299 z_k1_s0167
                                 - 0.202999803363919 z_k1_s0170
                                 - 0.294999423612486 z_k1_s0171
                                 - 0.153000291294147 z_k1_s0172
                                 - 0.136000083050289 z_k1_s0173
                                 - 0.165000116630871 z_k1_s0174
                                 - 0.141999995718651 z_k1_s0175
                                 - 0.184000679497655 z_k1_s0176
                                 - 0.160999695185123 z_k1_s0177
                                 - 1.06099804790507 z_k1_s1000
                                 - 0.70699811121173 z_k1_s1001
                                 - 0.233999166604611 z_k1_s1002
                                 - 0.363999733493158 z_k1_s1003
                                 - 0.274999046481842 z_k1_s1004
                                 - 0.259999351282119 z_k1_s1005
                                 - 0.286999159140614 z_k1_s1006
                                 - 0.376997686837944 z_k1_s1007
                                 - 0.6339989289093 z_k1_s1010
                                 - 0.96099949677434 z_k1_s1011
                                 - 0.476000970255142 z_k1_s1012
                                 - 0.304000321224673 z_k1_s1013
                                 - 0.479000728477377 z_k1_s1014
                                 - 0.912002109275784 z_k1_s1015
                                 - 1.1260005190193 z_k1_s1016
                                 - 0.83600288817094 z_k1_s1017
                                 - 0.135000033187492 z_k1_s1020
                                 - 0.167000293427498 z_k1_s1021
                                 - 0.351999659439965 z_k1_s1022
                                 - 0.121999991164992 z_k1_s1023
                                 - 0.147999492809974 z_k1_s1024
                                 - 0.237000407702497 z_k1_s1025
                                 - 0.121000010332492 z_k1_s1026
                                 - 0.173999285654965 z_k1_s1027
                                 - 0.0919999671476926 z_k1_s1030
                                 - 0.2030000969943 z_k1_s1031
                                 - 0.260999449707217 z_k1_s1032
                                 - 0.138999996587257 z_k1_s1033
                                 - 0.23000031161908 z_k1_s1034
                                 - 0.26399982355428 z_k1_s1035
                                 - 0.258000650859548 z_k1_s1036
                                 - 0.127999885814207 z_k1_s1037
                                 - 0.105999973029419 z_k1_s1040
                                 - 0.192999567615934 z_k1_s1041
                                 - 0.144999540644132 z_k1_s1042
                                 - 0.11499991318665 z_k1_s1043
                                 - 0.276000155087861 z_k1_s1044
                                 - 0.113999934258065 z_k1_s1045
                                 - 0.209000009273084 z_k1_s1046
                                 - 0.139999646001206 z_k1_s1047
                                 - 0.378999393453518 z_k1_s1050
                                 - 0.45099917904377 z_k1_s1051
                                 - 0.487999215044113 z_k1_s1052
                                 - 0.642001220675383 z_k1_s1053
                                 - 0.965996496836762 z_k1_s1054
                                 - 0.826998894680591 z_k1_s1055
                                 - 0.701999789002341 z_k1_s1056
                                 - 0.556997819219765 z_k1_s1057
                                 - 0.244999969578541 z_k1_s1060
                                 - 0.108999902753414 z_k1_s1061
                                 - 0.179999850812963 z_k1_s1062
                                 - 0.134000206593702 z_k1_s1063
                                 - 0.101999832594114 z_k1_s1064
                                 - 0.227999479563495 z_k1_s1065
                                 - 0.0579999243064331 z_k1_s1066
                                 - 0.186999175173182 z_k1_s1067
                                 - 0.400998166721627 z_k1_s1070
                                 - 0.487997768977941 z_k1_s1071
                                 - 0.715996726615175 z_k1_s1072
                                 - 0.243999250787662 z_k1_s1073
                                 - 0.620997160932994 z_k1_s1074
                                 - 0.545997503815483 z_k1_s1075
                                 - 0.241999259931195 z_k1_s1076
                                 - 0.404998148434561 z_k1_s1077
                                 - 0.21500024262381 z_k1_s1100
                                 - 0.345000827829427 z_k1_s1101
                                 - 0.340001606876319 z_k1_s1102
                                 - 0.432999521618894 z_k1_s1103
                                 - 0.462999809343449 z_k1_s1104
                                 - 0.31400069591425 z_k1_s1105
                                 - 0.101000141671683 z_k1_s1106
                                 - 0.27000010851804 z_k1_s1107
                                 - 0.11300020158395 z_k1_s1110
                                 - 0.202000637474244 z_k1_s1111
                                 - 0.36699975149386 z_k1_s1112
                                 - 0.345999618236569 z_k1_s1113
                                 - 0.11600002732051 z_k1_s1114
                                 - 0.492001189026008 z_k1_s1115
                                 - 0.108000105422625 z_k1_s1116
                                 - 0.189000232814642 z_k1_s1117
                                 - 0.443999741179115 z_k1_s1120
                                 - 0.296999431417125 z_k1_s1121
                                 - 0.140999898975801 z_k1_s1122
                                 - 0.482000484825048 z_k1_s1123
                                 - 0.52599885225727 z_k1_s1124
                                 - 0.152999884448209 z_k1_s1125
                                 - 0.591000050941105 z_k1_s1126
                                 - 0.152999884448209 z_k1_s1127
                                 - 0.152000031467606 z_k1_s1130
                                 - 0.101999977832223 z_k1_s1131
                                 - 0.115999482962293 z_k1_s1132
                                 - 0.125999863769252 z_k1_s1133
                                 - 0.197999932780206 z_k1_s1134
                                 - 0.0849999472601971 z_k1_s1135
                                 - 0.0909999701444378 z_k1_s1136
                                 - 0.157999437552044 z_k1_s1137
                                 - 0.792002446670168 z_k1_s1140
                                 - 0.280000524114951 z_k1_s1141
                                 - 0.256000602276803 z_k1_s1142
                                 - 0.444001426012186 z_k1_s1143
                                 - 0.707999848230206 z_k1_s1144
                                 - 0.259000413006169 z_k1_s1145
                                 - 0.37200022449452 z_k1_s1146
                                 - 0.479000414524138 z_k1_s1147
                                 - 0.594998526299192 z_k1_s1150
                                 - 0.406998961092127 z_k1_s1151
                                 - 0.602997791862934 z_k1_s1152
                                 - 0.213999472533653 z_k1_s1153
                                 - 0.237999410519677 z_k1_s1154
                                 - 0.294999148524263 z_k1_s1155
                                 - 0.185999595866501 z_k1_s1156
                                 - 0.520998437101299 z_k1_s1157
                                 - 0.720998356508963 z_k1_s1160
                                 - 1.04300113610202 z_k1_s1161
                                 - 0.524001791787446 z_k1_s1162
                                 - 0.417999838217379 z_k1_s1163
                                 - 0.282999722790148 z_k1_s1164
                                 - 0.955000205741339 z_k1_s1165
                                 - 0.366999611371702 z_k1_s1166
                                 - 0.269999772754878 z_k1_s1167
                                 - 0.197999739169898 z_k1_s1170
                                 - 0.241000613404203 z_k1_s1171
                                 - 0.231999721168782 z_k1_s1172
                                 - 0.147000429171884 z_k1_s1173
                                 - 0.217999806581043 z_k1_s1174
                                 - 0.10300001566724 z_k1_s1175
                                 - 0.0990000021850114 z_k1_s1176
                                 - 0.0880000314089129 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.14845 w_k1_ADM_s000
                                 + 1.35084 w_k1_AXP_s000
                                 + 1.00596 w_k1_DIS_s000 + 1.0941 w_k1_FDX_s000
                                 + 1.262 w_k1_IBM_s000 + 1.23597 w_k1_INTC_s000
                                 + 0.999189 w_k1_KO_s000
                                 + 1.20607 w_k1_MAT_s000
                                 + 1.04402 w_k1_MSFT_s000
                                 + 1.02508 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.756201 w_k1_ADM_s000
                                 + 0.899995 w_k1_AXP_s000
                                 + 0.860265 w_k1_DIS_s000
                                 + 0.836381 w_k1_FDX_s000
                                 + 1.25454 w_k1_IBM_s000
                                 + 0.891253 w_k1_INTC_s000
                                 + 0.983178 w_k1_KO_s000 + 1.1296 w_k1_MAT_s000
                                 + 0.833002 w_k1_MSFT_s000
                                 + 0.970864 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.772299 w_k1_ADM_s000
                                 + 0.69016 w_k1_AXP_s000
                                 + 0.826072 w_k1_DIS_s000
                                 + 0.855738 w_k1_FDX_s000
                                 + 0.699118 w_k1_IBM_s000
                                 + 0.991145 w_k1_INTC_s000
                                 + 0.737411 w_k1_KO_s000
                                 + 0.545016 w_k1_MAT_s000
                                 + 0.889005 w_k1_MSFT_s000
                                 + 0.890317 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.09052 w_k1_ADM_s000
                                 + 1.27467 w_k1_AXP_s000
                                 + 1.50567 w_k1_DIS_s000
                                 + 1.28324 w_k1_FDX_s000
                                 + 1.17051 w_k1_IBM_s000
                                 + 1.33186 w_k1_INTC_s000
                                 + 1.14357 w_k1_KO_s000 + 1.38743 w_k1_MAT_s000
                                 + 1.11148 w_k1_MSFT_s000
                                 + 1.07072 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.93981 w_k1_ADM_s000
                                 + 0.90233 w_k1_AXP_s000
                                 + 1.12809 w_k1_DIS_s000
                                 + 0.982331 w_k1_FDX_s000
                                 + 1.11001 w_k1_IBM_s000
                                 + 1.22571 w_k1_INTC_s000
                                 + 1.16542 w_k1_KO_s000
                                 + 0.732411 w_k1_MAT_s000
                                 + 0.952497 w_k1_MSFT_s000
                                 + 1.02264 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.653884 w_k1_ADM_s000
                                 + 0.629774 w_k1_AXP_s000
                                 + 0.670522 w_k1_DIS_s000
                                 + 0.697528 w_k1_FDX_s000
                                 + 0.906895 w_k1_IBM_s000
                                 + 0.794067 w_k1_INTC_s000
                                 + 0.682731 w_k1_KO_s000
                                 + 0.647487 w_k1_MAT_s000
                                 + 0.710811 w_k1_MSFT_s000
                                 + 0.813518 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.19044 w_k1_ADM_s000
                                 + 0.981372 w_k1_AXP_s000
                                 + 1.31916 w_k1_DIS_s000
                                 + 1.15074 w_k1_FDX_s000
                                 + 0.794943 w_k1_IBM_s000
                                 + 1.25061 w_k1_INTC_s000
                                 + 0.977361 w_k1_KO_s000
                                 + 0.82518 w_k1_MAT_s000
                                 + 1.22431 w_k1_MSFT_s000
                                 + 1.11295 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.26429 w_k1_ADM_s000
                                 + 0.933478 w_k1_AXP_s000
                                 + 0.894499 w_k1_DIS_s000
                                 + 0.989666 w_k1_FDX_s000
                                 + 0.832296 w_k1_IBM_s000
                                 + 0.8622 w_k1_INTC_s000 + 0.79259 w_k1_KO_s000
                                 + 0.706745 w_k1_MAT_s000
                                 + 0.869182 w_k1_MSFT_s000
                                 + 0.938493 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.868394 w_k1_ADM_s001
                                 + 0.866885 w_k1_AXP_s001
                                 + 0.719579 w_k1_DIS_s001
                                 + 0.771025 w_k1_FDX_s001
                                 + 0.914796 w_k1_IBM_s001
                                 + 0.98726 w_k1_INTC_s001
                                 + 0.925604 w_k1_KO_s001
                                 + 0.975129 w_k1_MAT_s001
                                 + 0.857006 w_k1_MSFT_s001
                                 + 0.918874 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.963229 w_k1_ADM_s001
                                 + 1.3275 w_k1_AXP_s001 + 1.33649 w_k1_DIS_s001
                                 + 1.06917 w_k1_FDX_s001
                                 + 1.65651 w_k1_IBM_s001
                                 + 1.17183 w_k1_INTC_s001
                                 + 1.38377 w_k1_KO_s001 + 1.41645 w_k1_MAT_s001
                                 + 1.11621 w_k1_MSFT_s001
                                 + 1.11166 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.931478 w_k1_ADM_s001
                                 + 1.20991 w_k1_AXP_s001
                                 + 1.54827 w_k1_DIS_s001
                                 + 1.22313 w_k1_FDX_s001
                                 + 1.12783 w_k1_IBM_s001
                                 + 1.39195 w_k1_INTC_s001
                                 + 1.30336 w_k1_KO_s001
                                 + 0.950473 w_k1_MAT_s001
                                 + 1.28427 w_k1_MSFT_s001
                                 + 1.08786 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.817074 w_k1_ADM_s001
                                 + 1.10597 w_k1_AXP_s001
                                 + 0.817209 w_k1_DIS_s001
                                 + 0.880851 w_k1_FDX_s001
                                 + 1.16402 w_k1_IBM_s001
                                 + 1.03485 w_k1_INTC_s001
                                 + 0.93657 w_k1_KO_s001 + 1.18193 w_k1_MAT_s001
                                 + 0.872457 w_k1_MSFT_s001
                                 + 0.861602 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.27311 w_k1_ADM_s001
                                 + 1.15676 w_k1_AXP_s001
                                 + 1.54802 w_k1_DIS_s001
                                 + 1.33701 w_k1_FDX_s001
                                 + 0.826753 w_k1_IBM_s001
                                 + 1.33579 w_k1_INTC_s001
                                 + 1.10022 w_k1_KO_s001
                                 + 0.779255 w_k1_MAT_s001
                                 + 1.33992 w_k1_MSFT_s001
                                 + 0.983117 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.903383 w_k1_ADM_s001
                                 + 0.914034 w_k1_AXP_s001
                                 + 0.868968 w_k1_DIS_s001
                                 + 0.900606 w_k1_FDX_s001
                                 + 0.723469 w_k1_IBM_s001
                                 + 1.01486 w_k1_INTC_s001
                                 + 0.951653 w_k1_KO_s001
                                 + 0.738609 w_k1_MAT_s001
                                 + 0.973377 w_k1_MSFT_s001
                                 + 0.934953 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.930328 w_k1_ADM_s001
                                 + 1.44774 w_k1_AXP_s001
                                 + 1.07053 w_k1_DIS_s001
                                 + 1.13475 w_k1_FDX_s001
                                 + 1.01286 w_k1_IBM_s001
                                 + 1.0128 w_k1_INTC_s001 + 1.03163 w_k1_KO_s001
                                 + 1.25319 w_k1_MAT_s001
                                 + 0.996531 w_k1_MSFT_s001
                                 + 1.09518 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.872217 w_k1_ADM_s001
                                 + 0.727093 w_k1_AXP_s001
                                 + 0.56637 w_k1_DIS_s001
                                 + 0.725091 w_k1_FDX_s001
                                 + 0.670738 w_k1_IBM_s001
                                 + 0.887856 w_k1_INTC_s001
                                 + 0.72012 w_k1_KO_s001 + 0.67717 w_k1_MAT_s001
                                 + 0.860217 w_k1_MSFT_s001
                                 + 0.777107 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.840379 w_k1_ADM_s002
                                 + 1.05804 w_k1_AXP_s002 + 0.9511 w_k1_DIS_s002
                                 + 0.95959 w_k1_FDX_s002
                                 + 1.35609 w_k1_IBM_s002
                                 + 1.05886 w_k1_INTC_s002
                                 + 1.00282 w_k1_KO_s002 + 1.28646 w_k1_MAT_s002
                                 + 0.909928 w_k1_MSFT_s002
                                 + 1.07872 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.940958 w_k1_ADM_s002
                                 + 0.67196 w_k1_AXP_s002
                                 + 0.921325 w_k1_DIS_s002
                                 + 0.889035 w_k1_FDX_s002
                                 + 0.870498 w_k1_IBM_s002
                                 + 1.06099 w_k1_INTC_s002
                                 + 0.821057 w_k1_KO_s002
                                 + 0.464639 w_k1_MAT_s002
                                 + 0.927958 w_k1_MSFT_s002
                                 + 0.924132 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.17277 w_k1_ADM_s002
                                 + 1.41664 w_k1_AXP_s002
                                 + 1.87249 w_k1_DIS_s002
                                 + 1.39385 w_k1_FDX_s002 + 1.5387 w_k1_IBM_s002
                                 + 1.407 w_k1_INTC_s002 + 1.54286 w_k1_KO_s002
                                 + 1.33509 w_k1_MAT_s002
                                 + 1.52382 w_k1_MSFT_s002
                                 + 1.36023 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.16009 w_k1_ADM_s002
                                 + 0.989927 w_k1_AXP_s002
                                 + 1.18362 w_k1_DIS_s002
                                 + 1.16345 w_k1_FDX_s002
                                 + 1.00968 w_k1_IBM_s002
                                 + 1.20786 w_k1_INTC_s002
                                 + 0.985678 w_k1_KO_s002
                                 + 0.503935 w_k1_MAT_s002
                                 + 1.09964 w_k1_MSFT_s002
                                 + 1.09961 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.985863 w_k1_ADM_s002
                                 + 1.11235 w_k1_AXP_s002
                                 + 0.877599 w_k1_DIS_s002
                                 + 0.921979 w_k1_FDX_s002
                                 + 1.12477 w_k1_IBM_s002
                                 + 1.06112 w_k1_INTC_s002
                                 + 0.906128 w_k1_KO_s002
                                 + 0.977348 w_k1_MAT_s002
                                 + 1.03814 w_k1_MSFT_s002
                                 + 0.98429 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.993596 w_k1_ADM_s002
                                 + 1.19397 w_k1_AXP_s002 + 1.1698 w_k1_DIS_s002
                                 + 0.999174 w_k1_FDX_s002
                                 + 1.64886 w_k1_IBM_s002
                                 + 1.17659 w_k1_INTC_s002
                                 + 1.13604 w_k1_KO_s002 + 1.00544 w_k1_MAT_s002
                                 + 0.840764 w_k1_MSFT_s002
                                 + 1.10975 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.02625 w_k1_ADM_s002
                                 + 0.911101 w_k1_AXP_s002
                                 + 0.685272 w_k1_DIS_s002
                                 + 0.851936 w_k1_FDX_s002
                                 + 0.899028 w_k1_IBM_s002
                                 + 0.944966 w_k1_INTC_s002
                                 + 0.849067 w_k1_KO_s002 + 0.4637 w_k1_MAT_s002
                                 + 0.770348 w_k1_MSFT_s002
                                 + 0.970506 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.820996 w_k1_ADM_s002
                                 + 0.700483 w_k1_AXP_s002
                                 + 0.664512 w_k1_DIS_s002
                                 + 0.818957 w_k1_FDX_s002
                                 + 0.723759 w_k1_IBM_s002
                                 + 0.95902 w_k1_INTC_s002
                                 + 0.743662 w_k1_KO_s002
                                 + 0.473988 w_k1_MAT_s002
                                 + 0.770255 w_k1_MSFT_s002
                                 + 0.874852 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.18442 w_k1_ADM_s003
                                 + 0.98871 w_k1_AXP_s003
                                 + 1.33699 w_k1_DIS_s003 + 1.1829 w_k1_FDX_s003
                                 + 1.21038 w_k1_IBM_s003
                                 + 0.964923 w_k1_INTC_s003
                                 + 1.24869 w_k1_KO_s003 + 1.04376 w_k1_MAT_s003
                                 + 1.08457 w_k1_MSFT_s003 + 1.1096 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02086 w_k1_ADM_s003
                                 + 0.984494 w_k1_AXP_s003
                                 + 0.851308 w_k1_DIS_s003
                                 + 0.831229 w_k1_FDX_s003
                                 + 1.49959 w_k1_IBM_s003
                                 + 0.841889 w_k1_INTC_s003
                                 + 1.13753 w_k1_KO_s003 + 1.4491 w_k1_MAT_s003
                                 + 0.87639 w_k1_MSFT_s003
                                 + 0.985492 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.63986 w_k1_ADM_s003
                                 + 1.77042 w_k1_AXP_s003 + 1.3811 w_k1_DIS_s003
                                 + 1.40568 w_k1_FDX_s003
                                 + 1.22126 w_k1_IBM_s003
                                 + 1.44194 w_k1_INTC_s003
                                 + 1.29062 w_k1_KO_s003 + 1.03827 w_k1_MAT_s003
                                 + 1.25408 w_k1_MSFT_s003
                                 + 1.19926 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.986554 w_k1_ADM_s003
                                 + 1.14037 w_k1_AXP_s003
                                 + 1.23369 w_k1_DIS_s003
                                 + 0.993209 w_k1_FDX_s003
                                 + 1.62739 w_k1_IBM_s003
                                 + 0.921203 w_k1_INTC_s003
                                 + 1.2745 w_k1_KO_s003 + 1.23499 w_k1_MAT_s003
                                 + 0.978114 w_k1_MSFT_s003
                                 + 1.05136 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.911741 w_k1_ADM_s003
                                 + 0.700565 w_k1_AXP_s003
                                 + 0.670883 w_k1_DIS_s003
                                 + 0.856687 w_k1_FDX_s003
                                 + 0.886906 w_k1_IBM_s003
                                 + 0.84112 w_k1_INTC_s003
                                 + 0.924586 w_k1_KO_s003
                                 + 0.66926 w_k1_MAT_s003
                                 + 0.784518 w_k1_MSFT_s003
                                 + 0.819777 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.18083 w_k1_ADM_s003
                                 + 0.883208 w_k1_AXP_s003
                                 + 0.852418 w_k1_DIS_s003
                                 + 0.919604 w_k1_FDX_s003
                                 + 0.941124 w_k1_IBM_s003
                                 + 1.13595 w_k1_INTC_s003
                                 + 0.900052 w_k1_KO_s003
                                 + 0.805443 w_k1_MAT_s003
                                 + 0.918843 w_k1_MSFT_s003
                                 + 0.880142 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.848195 w_k1_ADM_s003
                                 + 0.636326 w_k1_AXP_s003
                                 + 0.582349 w_k1_DIS_s003
                                 + 0.66018 w_k1_FDX_s003
                                 + 0.906255 w_k1_IBM_s003
                                 + 0.654414 w_k1_INTC_s003
                                 + 0.871676 w_k1_KO_s003
                                 + 0.71859 w_k1_MAT_s003
                                 + 0.613108 w_k1_MSFT_s003
                                 + 0.726204 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.27403 w_k1_ADM_s003
                                 + 0.955083 w_k1_AXP_s003
                                 + 1.00079 w_k1_DIS_s003
                                 + 1.06077 w_k1_FDX_s003
                                 + 0.965377 w_k1_IBM_s003
                                 + 1.06902 w_k1_INTC_s003 + 1.1514 w_k1_KO_s003
                                 + 0.841809 w_k1_MAT_s003
                                 + 1.11148 w_k1_MSFT_s003
                                 + 0.960503 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.806295 w_k1_ADM_s004
                                 + 0.945561 w_k1_AXP_s004
                                 + 0.876244 w_k1_DIS_s004
                                 + 0.784154 w_k1_FDX_s004
                                 + 1.27325 w_k1_IBM_s004
                                 + 0.829604 w_k1_INTC_s004
                                 + 0.958223 w_k1_KO_s004
                                 + 0.933301 w_k1_MAT_s004
                                 + 0.896726 w_k1_MSFT_s004
                                 + 1.04026 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.17228 w_k1_ADM_s004
                                 + 1.31086 w_k1_AXP_s004
                                 + 0.994713 w_k1_DIS_s004
                                 + 1.15048 w_k1_FDX_s004
                                 + 0.963607 w_k1_IBM_s004
                                 + 1.13413 w_k1_INTC_s004 + 0.9799 w_k1_KO_s004
                                 + 1.24817 w_k1_MAT_s004
                                 + 1.12666 w_k1_MSFT_s004
                                 + 1.19236 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.91109 w_k1_ADM_s004
                                 + 0.91952 w_k1_AXP_s004
                                 + 0.830808 w_k1_DIS_s004
                                 + 0.914185 w_k1_FDX_s004
                                 + 0.850326 w_k1_IBM_s004
                                 + 0.982252 w_k1_INTC_s004
                                 + 0.820055 w_k1_KO_s004
                                 + 0.673519 w_k1_MAT_s004
                                 + 0.945025 w_k1_MSFT_s004
                                 + 0.947337 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.21071 w_k1_ADM_s004
                                 + 1.21238 w_k1_AXP_s004 + 1.1512 w_k1_DIS_s004
                                 + 1.05409 w_k1_FDX_s004 + 1.0841 w_k1_IBM_s004
                                 + 1.1865 w_k1_INTC_s004 + 1.11209 w_k1_KO_s004
                                 + 0.91503 w_k1_MAT_s004
                                 + 1.05747 w_k1_MSFT_s004
                                 + 1.14872 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.12417 w_k1_ADM_s004
                                 + 1.21914 w_k1_AXP_s004
                                 + 0.789196 w_k1_DIS_s004
                                 + 1.04532 w_k1_FDX_s004
                                 + 0.834682 w_k1_IBM_s004
                                 + 0.97834 w_k1_INTC_s004
                                 + 0.773287 w_k1_KO_s004
                                 + 0.687904 w_k1_MAT_s004
                                 + 0.810303 w_k1_MSFT_s004
                                 + 0.995604 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.769338 w_k1_ADM_s004
                                 + 0.777397 w_k1_AXP_s004
                                 + 0.721378 w_k1_DIS_s004
                                 + 0.723098 w_k1_FDX_s004
                                 + 0.876259 w_k1_IBM_s004
                                 + 0.829583 w_k1_INTC_s004
                                 + 0.814572 w_k1_KO_s004
                                 + 0.723503 w_k1_MAT_s004
                                 + 0.804222 w_k1_MSFT_s004
                                 + 0.936401 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.06035 w_k1_ADM_s004
                                 + 1.72289 w_k1_AXP_s004
                                 + 1.77337 w_k1_DIS_s004
                                 + 1.44011 w_k1_FDX_s004
                                 + 1.30972 w_k1_IBM_s004
                                 + 1.46556 w_k1_INTC_s004 + 1.4477 w_k1_KO_s004
                                 + 1.24227 w_k1_MAT_s004
                                 + 1.41312 w_k1_MSFT_s004
                                 + 1.29403 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.14542 w_k1_ADM_s004
                                 + 1.40014 w_k1_AXP_s004
                                 + 1.30491 w_k1_DIS_s004
                                 + 1.21057 w_k1_FDX_s004
                                 + 1.30002 w_k1_IBM_s004
                                 + 1.43314 w_k1_INTC_s004
                                 + 1.33228 w_k1_KO_s004 + 1.05442 w_k1_MAT_s004
                                 + 1.26673 w_k1_MSFT_s004
                                 + 1.26268 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.19029 w_k1_ADM_s005
                                 + 1.62599 w_k1_AXP_s005
                                 + 1.67204 w_k1_DIS_s005
                                 + 1.39761 w_k1_FDX_s005
                                 + 1.18933 w_k1_IBM_s005
                                 + 1.27269 w_k1_INTC_s005
                                 + 1.20638 w_k1_KO_s005 + 1.35981 w_k1_MAT_s005
                                 + 1.23841 w_k1_MSFT_s005
                                 + 1.40256 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.01158 w_k1_ADM_s005
                                 + 1.28815 w_k1_AXP_s005
                                 + 1.27906 w_k1_DIS_s005
                                 + 1.21517 w_k1_FDX_s005
                                 + 0.849679 w_k1_IBM_s005
                                 + 1.29602 w_k1_INTC_s005
                                 + 1.04673 w_k1_KO_s005
                                 + 0.778244 w_k1_MAT_s005
                                 + 1.13882 w_k1_MSFT_s005
                                 + 1.12148 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.745749 w_k1_ADM_s005
                                 + 0.753841 w_k1_AXP_s005
                                 + 0.961567 w_k1_DIS_s005
                                 + 0.789344 w_k1_FDX_s005
                                 + 1.08275 w_k1_IBM_s005
                                 + 0.860671 w_k1_INTC_s005
                                 + 1.06875 w_k1_KO_s005 + 1.01216 w_k1_MAT_s005
                                 + 0.819719 w_k1_MSFT_s005
                                 + 1.04015 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.910502 w_k1_ADM_s005
                                 + 0.927693 w_k1_AXP_s005
                                 + 1.08934 w_k1_DIS_s005
                                 + 0.974868 w_k1_FDX_s005
                                 + 1.06109 w_k1_IBM_s005
                                 + 1.02861 w_k1_INTC_s005
                                 + 1.22312 w_k1_KO_s005
                                 + 0.874203 w_k1_MAT_s005
                                 + 1.12047 w_k1_MSFT_s005
                                 + 1.10633 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.93719 w_k1_ADM_s005
                                 + 0.79836 w_k1_AXP_s005
                                 + 0.78129 w_k1_DIS_s005
                                 + 0.912875 w_k1_FDX_s005
                                 + 0.875533 w_k1_IBM_s005
                                 + 0.868605 w_k1_INTC_s005
                                 + 0.842044 w_k1_KO_s005
                                 + 0.645318 w_k1_MAT_s005
                                 + 0.90497 w_k1_MSFT_s005
                                 + 0.94833 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.17164 w_k1_ADM_s005
                                 + 1.26064 w_k1_AXP_s005
                                 + 1.03861 w_k1_DIS_s005
                                 + 1.03138 w_k1_FDX_s005
                                 + 1.53791 w_k1_IBM_s005
                                 + 1.00808 w_k1_INTC_s005
                                 + 1.01655 w_k1_KO_s005 + 1.55584 w_k1_MAT_s005
                                 + 0.975473 w_k1_MSFT_s005
                                 + 1.12043 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.858342 w_k1_ADM_s005
                                 + 0.724215 w_k1_AXP_s005
                                 + 0.748641 w_k1_DIS_s005
                                 + 0.813001 w_k1_FDX_s005
                                 + 0.616772 w_k1_IBM_s005
                                 + 0.786924 w_k1_INTC_s005
                                 + 0.744444 w_k1_KO_s005
                                 + 0.571136 w_k1_MAT_s005
                                 + 0.740577 w_k1_MSFT_s005
                                 + 0.892562 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.94311 w_k1_ADM_s005
                                 + 1.14646 w_k1_AXP_s005
                                 + 1.43013 w_k1_DIS_s005
                                 + 1.14693 w_k1_FDX_s005
                                 + 1.20546 w_k1_IBM_s005
                                 + 1.02038 w_k1_INTC_s005
                                 + 1.25228 w_k1_KO_s005 + 1.3666 w_k1_MAT_s005
                                 + 1.21263 w_k1_MSFT_s005
                                 + 1.13936 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.714616 w_k1_ADM_s006
                                 + 0.734729 w_k1_AXP_s006
                                 + 0.673229 w_k1_DIS_s006
                                 + 0.7645 w_k1_FDX_s006 + 1.07649 w_k1_IBM_s006
                                 + 0.839704 w_k1_INTC_s006
                                 + 0.857994 w_k1_KO_s006
                                 + 1.12889 w_k1_MAT_s006
                                 + 0.895809 w_k1_MSFT_s006
                                 + 0.815684 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.926518 w_k1_ADM_s006
                                 + 1.34205 w_k1_AXP_s006 + 1.3399 w_k1_DIS_s006
                                 + 1.16899 w_k1_FDX_s006
                                 + 1.09406 w_k1_IBM_s006
                                 + 1.30051 w_k1_INTC_s006
                                 + 1.11598 w_k1_KO_s006 + 1.33725 w_k1_MAT_s006
                                 + 1.11018 w_k1_MSFT_s006
                                 + 1.05839 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.945963 w_k1_ADM_s006
                                 + 0.673437 w_k1_AXP_s006
                                 + 0.664864 w_k1_DIS_s006
                                 + 0.801097 w_k1_FDX_s006
                                 + 0.683568 w_k1_IBM_s006
                                 + 0.829533 w_k1_INTC_s006
                                 + 0.853736 w_k1_KO_s006
                                 + 0.762389 w_k1_MAT_s006
                                 + 0.885336 w_k1_MSFT_s006
                                 + 0.895781 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.10109 w_k1_ADM_s006
                                 + 1.23395 w_k1_AXP_s006 + 1.6937 w_k1_DIS_s006
                                 + 1.24626 w_k1_FDX_s006
                                 + 1.24404 w_k1_IBM_s006
                                 + 1.33987 w_k1_INTC_s006
                                 + 1.32767 w_k1_KO_s006 + 1.30894 w_k1_MAT_s006
                                 + 1.42628 w_k1_MSFT_s006
                                 + 1.10773 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.816773 w_k1_ADM_s006
                                 + 0.956433 w_k1_AXP_s006
                                 + 0.974439 w_k1_DIS_s006
                                 + 0.814517 w_k1_FDX_s006
                                 + 1.19427 w_k1_IBM_s006
                                 + 0.875322 w_k1_INTC_s006
                                 + 1.04213 w_k1_KO_s006 + 1.25439 w_k1_MAT_s006
                                 + 0.970156 w_k1_MSFT_s006
                                 + 0.940794 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.17348 w_k1_ADM_s006
                                 + 1.24955 w_k1_AXP_s006
                                 + 1.42376 w_k1_DIS_s006
                                 + 1.08319 w_k1_FDX_s006
                                 + 1.68321 w_k1_IBM_s006
                                 + 1.33762 w_k1_INTC_s006
                                 + 1.63364 w_k1_KO_s006 + 1.74999 w_k1_MAT_s006
                                 + 1.4117 w_k1_MSFT_s006 + 1.3842 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.849914 w_k1_ADM_s006
                                 + 0.854818 w_k1_AXP_s006
                                 + 1.21249 w_k1_DIS_s006
                                 + 1.00638 w_k1_FDX_s006
                                 + 0.820901 w_k1_IBM_s006
                                 + 0.947587 w_k1_INTC_s006
                                 + 1.01959 w_k1_KO_s006 + 0.78694 w_k1_MAT_s006
                                 + 1.10202 w_k1_MSFT_s006
                                 + 1.00431 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.983581 w_k1_ADM_s006
                                 + 0.936267 w_k1_AXP_s006
                                 + 1.25977 w_k1_DIS_s006
                                 + 1.05375 w_k1_FDX_s006
                                 + 1.18809 w_k1_IBM_s006
                                 + 1.16366 w_k1_INTC_s006
                                 + 1.22517 w_k1_KO_s006 + 1.12323 w_k1_MAT_s006
                                 + 1.28566 w_k1_MSFT_s006
                                 + 1.08955 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.995022 w_k1_ADM_s007
                                 + 1.43439 w_k1_AXP_s007
                                 + 1.14866 w_k1_DIS_s007
                                 + 1.15036 w_k1_FDX_s007
                                 + 1.14942 w_k1_IBM_s007
                                 + 1.07396 w_k1_INTC_s007
                                 + 0.888237 w_k1_KO_s007
                                 + 1.40222 w_k1_MAT_s007
                                 + 1.00781 w_k1_MSFT_s007
                                 + 1.10385 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.872058 w_k1_ADM_s007
                                 + 1.02086 w_k1_AXP_s007
                                 + 0.930111 w_k1_DIS_s007
                                 + 0.853013 w_k1_FDX_s007 + 1.592 w_k1_IBM_s007
                                 + 0.930979 w_k1_INTC_s007
                                 + 0.924206 w_k1_KO_s007
                                 + 1.66031 w_k1_MAT_s007
                                 + 0.827664 w_k1_MSFT_s007
                                 + 0.963139 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.01432 w_k1_ADM_s007
                                 + 1.09291 w_k1_AXP_s007
                                 + 1.11847 w_k1_DIS_s007
                                 + 1.17416 w_k1_FDX_s007
                                 + 0.681228 w_k1_IBM_s007
                                 + 0.981422 w_k1_INTC_s007
                                 + 0.834084 w_k1_KO_s007
                                 + 0.788221 w_k1_MAT_s007
                                 + 1.05729 w_k1_MSFT_s007
                                 + 1.11122 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.838076 w_k1_ADM_s007
                                 + 0.68248 w_k1_AXP_s007
                                 + 0.659311 w_k1_DIS_s007
                                 + 0.848692 w_k1_FDX_s007
                                 + 0.571834 w_k1_IBM_s007
                                 + 1.03962 w_k1_INTC_s007
                                 + 0.602964 w_k1_KO_s007
                                 + 0.503667 w_k1_MAT_s007
                                 + 0.803623 w_k1_MSFT_s007
                                 + 0.80464 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.859709 w_k1_ADM_s007
                                 + 0.628613 w_k1_AXP_s007
                                 + 0.663258 w_k1_DIS_s007
                                 + 0.729256 w_k1_FDX_s007
                                 + 0.896277 w_k1_IBM_s007
                                 + 0.989408 w_k1_INTC_s007
                                 + 0.801581 w_k1_KO_s007
                                 + 0.566515 w_k1_MAT_s007
                                 + 0.823731 w_k1_MSFT_s007
                                 + 0.854953 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.975945 w_k1_ADM_s007
                                 + 0.817627 w_k1_AXP_s007
                                 + 0.842612 w_k1_DIS_s007
                                 + 0.941994 w_k1_FDX_s007
                                 + 0.900379 w_k1_IBM_s007
                                 + 1.13533 w_k1_INTC_s007
                                 + 0.745544 w_k1_KO_s007
                                 + 0.655234 w_k1_MAT_s007
                                 + 0.917828 w_k1_MSFT_s007
                                 + 0.942452 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.11801 w_k1_ADM_s007
                                 + 1.04227 w_k1_AXP_s007
                                 + 1.20937 w_k1_DIS_s007
                                 + 1.06395 w_k1_FDX_s007
                                 + 1.25981 w_k1_IBM_s007
                                 + 1.17019 w_k1_INTC_s007
                                 + 1.04006 w_k1_KO_s007
                                 + 0.834032 w_k1_MAT_s007
                                 + 1.10443 w_k1_MSFT_s007
                                 + 1.05988 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.40784 w_k1_ADM_s007
                                 + 1.31248 w_k1_AXP_s007 + 1.6761 w_k1_DIS_s007
                                 + 1.26143 w_k1_FDX_s007
                                 + 1.18651 w_k1_IBM_s007
                                 + 1.39577 w_k1_INTC_s007
                                 + 1.30964 w_k1_KO_s007 + 1.14712 w_k1_MAT_s007
                                 + 1.35974 w_k1_MSFT_s007
                                 + 1.18438 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.997849 w_k1_ADM_s010
                                 + 0.920566 w_k1_AXP_s010
                                 + 0.857613 w_k1_DIS_s010
                                 + 1.00117 w_k1_FDX_s010
                                 + 0.636862 w_k1_IBM_s010
                                 + 0.91545 w_k1_INTC_s010
                                 + 0.797005 w_k1_KO_s010
                                 + 0.589521 w_k1_MAT_s010
                                 + 0.951853 w_k1_MSFT_s010
                                 + 0.831206 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.31688 w_k1_ADM_s010
                                 + 1.46283 w_k1_AXP_s010
                                 + 1.63955 w_k1_DIS_s010 + 1.2654 w_k1_FDX_s010
                                 + 1.3752 w_k1_IBM_s010
                                 + 1.45378 w_k1_INTC_s010
                                 + 1.53646 w_k1_KO_s010 + 1.48349 w_k1_MAT_s010
                                 + 1.43524 w_k1_MSFT_s010
                                 + 1.15313 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.621526 w_k1_ADM_s010
                                 + 0.796159 w_k1_AXP_s010
                                 + 0.753125 w_k1_DIS_s010
                                 + 0.771664 w_k1_FDX_s010
                                 + 1.1254 w_k1_IBM_s010
                                 + 0.959424 w_k1_INTC_s010
                                 + 0.768075 w_k1_KO_s010
                                 + 1.02349 w_k1_MAT_s010
                                 + 0.875159 w_k1_MSFT_s010
                                 + 0.81303 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10085 w_k1_ADM_s010
                                 + 1.14816 w_k1_AXP_s010
                                 + 1.17173 w_k1_DIS_s010
                                 + 1.16472 w_k1_FDX_s010
                                 + 0.999473 w_k1_IBM_s010
                                 + 1.21764 w_k1_INTC_s010
                                 + 1.04144 w_k1_KO_s010 + 1.07764 w_k1_MAT_s010
                                 + 1.2884 w_k1_MSFT_s010 + 1.03894 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.21413 w_k1_ADM_s010
                                 + 1.36923 w_k1_AXP_s010
                                 + 1.30878 w_k1_DIS_s010
                                 + 1.02498 w_k1_FDX_s010
                                 + 1.57126 w_k1_IBM_s010
                                 + 1.16055 w_k1_INTC_s010
                                 + 1.31895 w_k1_KO_s010 + 1.32142 w_k1_MAT_s010
                                 + 1.08932 w_k1_MSFT_s010
                                 + 1.04847 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.24083 w_k1_ADM_s010
                                 + 1.37298 w_k1_AXP_s010
                                 + 1.28742 w_k1_DIS_s010
                                 + 1.35919 w_k1_FDX_s010
                                 + 0.918899 w_k1_IBM_s010
                                 + 1.24723 w_k1_INTC_s010
                                 + 1.20086 w_k1_KO_s010 + 1.16841 w_k1_MAT_s010
                                 + 1.25275 w_k1_MSFT_s010
                                 + 1.16896 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.02042 w_k1_ADM_s010
                                 + 0.964162 w_k1_AXP_s010
                                 + 1.2234 w_k1_DIS_s010 + 1.01339 w_k1_FDX_s010
                                 + 1.11043 w_k1_IBM_s010
                                 + 1.14236 w_k1_INTC_s010
                                 + 1.13522 w_k1_KO_s010 + 1.37569 w_k1_MAT_s010
                                 + 1.16175 w_k1_MSFT_s010
                                 + 0.963727 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.878148 w_k1_ADM_s010
                                 + 0.803088 w_k1_AXP_s010
                                 + 0.849064 w_k1_DIS_s010
                                 + 0.85515 w_k1_FDX_s010
                                 + 0.925644 w_k1_IBM_s010
                                 + 0.921097 w_k1_INTC_s010
                                 + 0.938605 w_k1_KO_s010
                                 + 0.667506 w_k1_MAT_s010
                                 + 0.949161 w_k1_MSFT_s010
                                 + 0.80421 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.20225 w_k1_ADM_s011
                                 + 1.1523 w_k1_AXP_s011 + 1.44282 w_k1_DIS_s011
                                 + 1.14479 w_k1_FDX_s011
                                 + 1.66883 w_k1_IBM_s011
                                 + 1.16491 w_k1_INTC_s011
                                 + 1.38694 w_k1_KO_s011 + 1.48533 w_k1_MAT_s011
                                 + 1.33467 w_k1_MSFT_s011
                                 + 1.27663 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.07978 w_k1_ADM_s011
                                 + 0.898211 w_k1_AXP_s011
                                 + 0.939508 w_k1_DIS_s011
                                 + 0.921321 w_k1_FDX_s011
                                 + 1.44185 w_k1_IBM_s011
                                 + 0.919921 w_k1_INTC_s011
                                 + 1.07254 w_k1_KO_s011 + 1.03451 w_k1_MAT_s011
                                 + 0.894279 w_k1_MSFT_s011
                                 + 1.16349 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.995621 w_k1_ADM_s011
                                 + 0.979883 w_k1_AXP_s011
                                 + 1.10474 w_k1_DIS_s011
                                 + 1.08053 w_k1_FDX_s011
                                 + 1.05593 w_k1_IBM_s011
                                 + 1.10229 w_k1_INTC_s011
                                 + 1.03527 w_k1_KO_s011
                                 + 0.985722 w_k1_MAT_s011
                                 + 1.11782 w_k1_MSFT_s011
                                 + 1.10288 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.876447 w_k1_ADM_s011
                                 + 0.799734 w_k1_AXP_s011
                                 + 0.664919 w_k1_DIS_s011
                                 + 0.76133 w_k1_FDX_s011
                                 + 1.04299 w_k1_IBM_s011
                                 + 0.934269 w_k1_INTC_s011
                                 + 0.940691 w_k1_KO_s011
                                 + 0.771485 w_k1_MAT_s011
                                 + 0.832173 w_k1_MSFT_s011
                                 + 0.937495 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.04646 w_k1_ADM_s011
                                 + 0.745416 w_k1_AXP_s011
                                 + 0.833167 w_k1_DIS_s011
                                 + 0.952811 w_k1_FDX_s011
                                 + 0.863005 w_k1_IBM_s011
                                 + 0.97632 w_k1_INTC_s011
                                 + 1.00012 w_k1_KO_s011
                                 + 0.587954 w_k1_MAT_s011
                                 + 0.964617 w_k1_MSFT_s011
                                 + 1.06784 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.21126 w_k1_ADM_s011
                                 + 1.45032 w_k1_AXP_s011
                                 + 0.944648 w_k1_DIS_s011
                                 + 1.19997 w_k1_FDX_s011
                                 + 1.20907 w_k1_IBM_s011
                                 + 1.12808 w_k1_INTC_s011
                                 + 1.01435 w_k1_KO_s011 + 1.62266 w_k1_MAT_s011
                                 + 1.01241 w_k1_MSFT_s011
                                 + 1.05322 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.23953 w_k1_ADM_s011
                                 + 1.57777 w_k1_AXP_s011
                                 + 1.63553 w_k1_DIS_s011
                                 + 1.50042 w_k1_FDX_s011
                                 + 1.31155 w_k1_IBM_s011
                                 + 1.28119 w_k1_INTC_s011
                                 + 1.48551 w_k1_KO_s011 + 1.14184 w_k1_MAT_s011
                                 + 1.39045 w_k1_MSFT_s011 + 1.3016 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.24682 w_k1_ADM_s011
                                 + 1.19872 w_k1_AXP_s011
                                 + 1.28013 w_k1_DIS_s011
                                 + 1.22781 w_k1_FDX_s011
                                 + 0.936386 w_k1_IBM_s011
                                 + 1.09057 w_k1_INTC_s011
                                 + 1.12665 w_k1_KO_s011
                                 + 0.880709 w_k1_MAT_s011
                                 + 1.16797 w_k1_MSFT_s011 + 1.1546 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.998989 w_k1_ADM_s012
                                 + 0.866599 w_k1_AXP_s012
                                 + 0.851096 w_k1_DIS_s012
                                 + 0.923056 w_k1_FDX_s012
                                 + 0.910841 w_k1_IBM_s012
                                 + 0.96961 w_k1_INTC_s012
                                 + 1.00166 w_k1_KO_s012
                                 + 0.918791 w_k1_MAT_s012
                                 + 0.875082 w_k1_MSFT_s012
                                 + 0.903826 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.954885 w_k1_ADM_s012
                                 + 1.17777 w_k1_AXP_s012
                                 + 1.31126 w_k1_DIS_s012
                                 + 1.27165 w_k1_FDX_s012
                                 + 0.913514 w_k1_IBM_s012
                                 + 1.28573 w_k1_INTC_s012
                                 + 1.03269 w_k1_KO_s012 + 1.1809 w_k1_MAT_s012
                                 + 1.00513 w_k1_MSFT_s012
                                 + 1.16256 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.771917 w_k1_ADM_s012
                                 + 0.950781 w_k1_AXP_s012
                                 + 1.06371 w_k1_DIS_s012
                                 + 0.958619 w_k1_FDX_s012
                                 + 1.3515 w_k1_IBM_s012
                                 + 1.20716 w_k1_INTC_s012
                                 + 1.24331 w_k1_KO_s012 + 1.39598 w_k1_MAT_s012
                                 + 0.925951 w_k1_MSFT_s012
                                 + 1.00871 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.779289 w_k1_ADM_s012
                                 + 0.749646 w_k1_AXP_s012
                                 + 0.640069 w_k1_DIS_s012
                                 + 0.793894 w_k1_FDX_s012
                                 + 0.660209 w_k1_IBM_s012
                                 + 0.781342 w_k1_INTC_s012
                                 + 0.66059 w_k1_KO_s012
                                 + 0.598855 w_k1_MAT_s012
                                 + 0.6768 w_k1_MSFT_s012
                                 + 0.844906 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.853153 w_k1_ADM_s012
                                 + 1.33712 w_k1_AXP_s012
                                 + 0.855271 w_k1_DIS_s012
                                 + 1.02702 w_k1_FDX_s012
                                 + 1.24524 w_k1_IBM_s012
                                 + 0.998803 w_k1_INTC_s012
                                 + 1.00207 w_k1_KO_s012 + 1.67507 w_k1_MAT_s012
                                 + 0.870894 w_k1_MSFT_s012
                                 + 1.01395 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.987397 w_k1_ADM_s012
                                 + 1.05079 w_k1_AXP_s012
                                 + 1.16889 w_k1_DIS_s012
                                 + 1.09426 w_k1_FDX_s012
                                 + 0.935999 w_k1_IBM_s012
                                 + 1.06993 w_k1_INTC_s012
                                 + 1.02501 w_k1_KO_s012
                                 + 0.922423 w_k1_MAT_s012
                                 + 1.10303 w_k1_MSFT_s012
                                 + 1.00748 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.58321 w_k1_ADM_s012
                                 + 1.52401 w_k1_AXP_s012
                                 + 1.60572 w_k1_DIS_s012
                                 + 1.34057 w_k1_FDX_s012
                                 + 1.52998 w_k1_IBM_s012
                                 + 1.20362 w_k1_INTC_s012
                                 + 1.38891 w_k1_KO_s012 + 1.85972 w_k1_MAT_s012
                                 + 1.16106 w_k1_MSFT_s012
                                 + 1.14353 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.05595 w_k1_ADM_s012
                                 + 1.08615 w_k1_AXP_s012
                                 + 0.863376 w_k1_DIS_s012
                                 + 1.08958 w_k1_FDX_s012
                                 + 0.657842 w_k1_IBM_s012
                                 + 1.05309 w_k1_INTC_s012
                                 + 0.802567 w_k1_KO_s012
                                 + 0.72241 w_k1_MAT_s012
                                 + 0.856085 w_k1_MSFT_s012
                                 + 0.936699 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.51344 w_k1_ADM_s013
                                 + 1.49168 w_k1_AXP_s013
                                 + 1.15426 w_k1_DIS_s013
                                 + 1.34735 w_k1_FDX_s013
                                 + 1.00332 w_k1_IBM_s013
                                 + 1.06519 w_k1_INTC_s013
                                 + 0.918814 w_k1_KO_s013
                                 + 1.19204 w_k1_MAT_s013
                                 + 1.16413 w_k1_MSFT_s013
                                 + 1.02922 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.945602 w_k1_ADM_s013
                                 + 1.04167 w_k1_AXP_s013
                                 + 0.947665 w_k1_DIS_s013
                                 + 0.952781 w_k1_FDX_s013
                                 + 1.23637 w_k1_IBM_s013
                                 + 1.00763 w_k1_INTC_s013
                                 + 0.904262 w_k1_KO_s013
                                 + 1.66633 w_k1_MAT_s013
                                 + 0.974408 w_k1_MSFT_s013
                                 + 0.967644 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.23459 w_k1_ADM_s013
                                 + 1.15175 w_k1_AXP_s013
                                 + 1.50994 w_k1_DIS_s013 + 1.1956 w_k1_FDX_s013
                                 + 1.61452 w_k1_IBM_s013
                                 + 1.1816 w_k1_INTC_s013 + 1.41851 w_k1_KO_s013
                                 + 1.59297 w_k1_MAT_s013
                                 + 1.18545 w_k1_MSFT_s013
                                 + 1.09338 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.873361 w_k1_ADM_s013
                                 + 0.689032 w_k1_AXP_s013
                                 + 0.723025 w_k1_DIS_s013
                                 + 0.809031 w_k1_FDX_s013
                                 + 0.694028 w_k1_IBM_s013
                                 + 0.774944 w_k1_INTC_s013
                                 + 0.77957 w_k1_KO_s013
                                 + 0.688351 w_k1_MAT_s013
                                 + 0.858136 w_k1_MSFT_s013
                                 + 0.831685 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.902831 w_k1_ADM_s013
                                 + 0.932232 w_k1_AXP_s013
                                 + 0.703533 w_k1_DIS_s013
                                 + 0.832036 w_k1_FDX_s013
                                 + 1.01143 w_k1_IBM_s013
                                 + 0.760336 w_k1_INTC_s013
                                 + 0.867425 w_k1_KO_s013
                                 + 1.12677 w_k1_MAT_s013
                                 + 0.793297 w_k1_MSFT_s013
                                 + 0.869615 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.924454 w_k1_ADM_s013
                                 + 1.13096 w_k1_AXP_s013
                                 + 0.846158 w_k1_DIS_s013
                                 + 0.996971 w_k1_FDX_s013
                                 + 0.746737 w_k1_IBM_s013
                                 + 0.896547 w_k1_INTC_s013
                                 + 0.793094 w_k1_KO_s013
                                 + 1.08867 w_k1_MAT_s013
                                 + 0.904792 w_k1_MSFT_s013
                                 + 0.933452 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.2339 w_k1_ADM_s013
                                 + 1.12739 w_k1_AXP_s013
                                 + 1.35333 w_k1_DIS_s013
                                 + 1.24571 w_k1_FDX_s013
                                 + 0.923243 w_k1_IBM_s013
                                 + 0.910206 w_k1_INTC_s013
                                 + 1.21807 w_k1_KO_s013 + 1.11863 w_k1_MAT_s013
                                 + 1.18387 w_k1_MSFT_s013
                                 + 1.08792 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.862353 w_k1_ADM_s013
                                 + 0.739747 w_k1_AXP_s013
                                 + 0.996621 w_k1_DIS_s013
                                 + 0.960323 w_k1_FDX_s013
                                 + 0.935839 w_k1_IBM_s013
                                 + 0.828671 w_k1_INTC_s013
                                 + 0.944839 w_k1_KO_s013
                                 + 1.04064 w_k1_MAT_s013
                                 + 1.03555 w_k1_MSFT_s013
                                 + 0.867125 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.21851 w_k1_ADM_s014
                                 + 1.52355 w_k1_AXP_s014
                                 + 1.69037 w_k1_DIS_s014
                                 + 1.33336 w_k1_FDX_s014
                                 + 1.28809 w_k1_IBM_s014
                                 + 1.0638 w_k1_INTC_s014 + 1.38468 w_k1_KO_s014
                                 + 1.56388 w_k1_MAT_s014
                                 + 1.16939 w_k1_MSFT_s014
                                 + 1.24625 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.23162 w_k1_ADM_s014
                                 + 1.34242 w_k1_AXP_s014
                                 + 1.33349 w_k1_DIS_s014
                                 + 1.24632 w_k1_FDX_s014
                                 + 1.13579 w_k1_IBM_s014
                                 + 1.10812 w_k1_INTC_s014
                                 + 1.21379 w_k1_KO_s014 + 1.15395 w_k1_MAT_s014
                                 + 1.03383 w_k1_MSFT_s014
                                 + 1.13395 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.813663 w_k1_ADM_s014
                                 + 0.96475 w_k1_AXP_s014
                                 + 1.09411 w_k1_DIS_s014
                                 + 0.95853 w_k1_FDX_s014
                                 + 0.960557 w_k1_IBM_s014
                                 + 1.03715 w_k1_INTC_s014
                                 + 1.10437 w_k1_KO_s014
                                 + 0.818301 w_k1_MAT_s014
                                 + 1.04253 w_k1_MSFT_s014
                                 + 1.05071 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.987878 w_k1_ADM_s014
                                 + 1.26326 w_k1_AXP_s014
                                 + 0.843747 w_k1_DIS_s014
                                 + 0.936975 w_k1_FDX_s014
                                 + 1.0096 w_k1_IBM_s014
                                 + 0.850543 w_k1_INTC_s014
                                 + 0.958269 w_k1_KO_s014
                                 + 1.38683 w_k1_MAT_s014
                                 + 0.90068 w_k1_MSFT_s014
                                 + 0.942347 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.862333 w_k1_ADM_s014
                                 + 1.03669 w_k1_AXP_s014
                                 + 0.930503 w_k1_DIS_s014
                                 + 0.902067 w_k1_FDX_s014
                                 + 1.37146 w_k1_IBM_s014
                                 + 1.08093 w_k1_INTC_s014
                                 + 1.15371 w_k1_KO_s014 + 1.07305 w_k1_MAT_s014
                                 + 0.937533 w_k1_MSFT_s014
                                 + 0.99598 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.677787 w_k1_ADM_s014
                                 + 0.749712 w_k1_AXP_s014
                                 + 0.792651 w_k1_DIS_s014
                                 + 0.825884 w_k1_FDX_s014
                                 + 0.75947 w_k1_IBM_s014
                                 + 0.725641 w_k1_INTC_s014
                                 + 0.814346 w_k1_KO_s014
                                 + 1.00727 w_k1_MAT_s014
                                 + 0.827874 w_k1_MSFT_s014
                                 + 0.839019 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.02673 w_k1_ADM_s014
                                 + 1.1373 w_k1_AXP_s014 + 1.13485 w_k1_DIS_s014
                                 + 1.16703 w_k1_FDX_s014
                                 + 0.766278 w_k1_IBM_s014
                                 + 0.858609 w_k1_INTC_s014
                                 + 0.940969 w_k1_KO_s014
                                 + 0.937434 w_k1_MAT_s014
                                 + 0.998067 w_k1_MSFT_s014
                                 + 1.10136 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.0584 w_k1_ADM_s014
                                 + 1.01234 w_k1_AXP_s014 + 1.6311 w_k1_DIS_s014
                                 + 1.16808 w_k1_FDX_s014
                                 + 0.902661 w_k1_IBM_s014
                                 + 0.878935 w_k1_INTC_s014
                                 + 1.24307 w_k1_KO_s014 + 1.15287 w_k1_MAT_s014
                                 + 1.22232 w_k1_MSFT_s014
                                 + 1.05314 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.866032 w_k1_ADM_s015
                                 + 0.999371 w_k1_AXP_s015
                                 + 1.12402 w_k1_DIS_s015
                                 + 0.961543 w_k1_FDX_s015
                                 + 1.31794 w_k1_IBM_s015
                                 + 0.868303 w_k1_INTC_s015
                                 + 1.11706 w_k1_KO_s015 + 1.5565 w_k1_MAT_s015
                                 + 1.10904 w_k1_MSFT_s015 + 1.0044 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.09267 w_k1_ADM_s015
                                 + 1.05879 w_k1_AXP_s015
                                 + 0.952467 w_k1_DIS_s015
                                 + 1.04977 w_k1_FDX_s015
                                 + 1.03356 w_k1_IBM_s015
                                 + 0.900791 w_k1_INTC_s015
                                 + 1.00158 w_k1_KO_s015 + 1.27988 w_k1_MAT_s015
                                 + 1.1115 w_k1_MSFT_s015 + 1.00639 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.11282 w_k1_ADM_s015
                                 + 1.27419 w_k1_AXP_s015
                                 + 1.19665 w_k1_DIS_s015 + 1.2092 w_k1_FDX_s015
                                 + 1.1214 w_k1_IBM_s015
                                 + 1.09603 w_k1_INTC_s015
                                 + 1.28393 w_k1_KO_s015 + 1.6356 w_k1_MAT_s015
                                 + 1.22357 w_k1_MSFT_s015 + 1.1647 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.86994 w_k1_ADM_s015
                                 + 0.822646 w_k1_AXP_s015
                                 + 0.849923 w_k1_DIS_s015
                                 + 0.887901 w_k1_FDX_s015
                                 + 0.843538 w_k1_IBM_s015
                                 + 0.881484 w_k1_INTC_s015
                                 + 0.843156 w_k1_KO_s015
                                 + 0.861268 w_k1_MAT_s015
                                 + 1.02575 w_k1_MSFT_s015
                                 + 0.851585 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.13039 w_k1_ADM_s015
                                 + 1.05409 w_k1_AXP_s015
                                 + 1.36621 w_k1_DIS_s015
                                 + 1.20036 w_k1_FDX_s015
                                 + 0.864393 w_k1_IBM_s015
                                 + 0.904228 w_k1_INTC_s015
                                 + 1.15716 w_k1_KO_s015 + 1.20387 w_k1_MAT_s015
                                 + 1.41875 w_k1_MSFT_s015
                                 + 1.19216 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.33457 w_k1_ADM_s015
                                 + 1.81936 w_k1_AXP_s015
                                 + 1.36847 w_k1_DIS_s015
                                 + 1.32754 w_k1_FDX_s015
                                 + 1.43248 w_k1_IBM_s015
                                 + 1.27684 w_k1_INTC_s015
                                 + 1.43012 w_k1_KO_s015 + 2.16023 w_k1_MAT_s015
                                 + 1.23816 w_k1_MSFT_s015
                                 + 1.22197 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.863997 w_k1_ADM_s015
                                 + 0.708411 w_k1_AXP_s015
                                 + 0.641392 w_k1_DIS_s015
                                 + 0.747128 w_k1_FDX_s015
                                 + 0.657157 w_k1_IBM_s015
                                 + 0.811794 w_k1_INTC_s015
                                 + 0.759761 w_k1_KO_s015
                                 + 0.810695 w_k1_MAT_s015
                                 + 0.865601 w_k1_MSFT_s015
                                 + 0.846086 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.18852 w_k1_ADM_s015
                                 + 1.10756 w_k1_AXP_s015
                                 + 0.995757 w_k1_DIS_s015
                                 + 0.981749 w_k1_FDX_s015
                                 + 1.66506 w_k1_IBM_s015
                                 + 0.770849 w_k1_INTC_s015
                                 + 1.27668 w_k1_KO_s015 + 1.78253 w_k1_MAT_s015
                                 + 0.991189 w_k1_MSFT_s015
                                 + 1.06336 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.16397 w_k1_ADM_s016
                                 + 1.34136 w_k1_AXP_s016
                                 + 1.27521 w_k1_DIS_s016
                                 + 1.12476 w_k1_FDX_s016
                                 + 1.05941 w_k1_IBM_s016
                                 + 1.35268 w_k1_INTC_s016
                                 + 1.19187 w_k1_KO_s016
                                 + 0.750131 w_k1_MAT_s016
                                 + 1.08541 w_k1_MSFT_s016
                                 + 1.08022 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.862131 w_k1_ADM_s016
                                 + 0.887728 w_k1_AXP_s016
                                 + 1.00082 w_k1_DIS_s016
                                 + 0.847129 w_k1_FDX_s016
                                 + 0.959936 w_k1_IBM_s016
                                 + 0.951483 w_k1_INTC_s016
                                 + 0.957411 w_k1_KO_s016
                                 + 0.882848 w_k1_MAT_s016
                                 + 1.00156 w_k1_MSFT_s016
                                 + 0.931001 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.11777 w_k1_ADM_s016
                                 + 1.23548 w_k1_AXP_s016 + 1.5319 w_k1_DIS_s016
                                 + 1.20992 w_k1_FDX_s016
                                 + 1.43039 w_k1_IBM_s016
                                 + 1.13706 w_k1_INTC_s016 + 1.3243 w_k1_KO_s016
                                 + 1.25732 w_k1_MAT_s016
                                 + 1.1514 w_k1_MSFT_s016 + 1.15447 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.758237 w_k1_ADM_s016
                                 + 0.757317 w_k1_AXP_s016
                                 + 0.798491 w_k1_DIS_s016
                                 + 0.736388 w_k1_FDX_s016
                                 + 1.15938 w_k1_IBM_s016
                                 + 0.876977 w_k1_INTC_s016
                                 + 0.993145 w_k1_KO_s016
                                 + 0.992635 w_k1_MAT_s016
                                 + 0.869346 w_k1_MSFT_s016
                                 + 0.950623 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.729228 w_k1_ADM_s016
                                 + 0.63495 w_k1_AXP_s016
                                 + 0.63767 w_k1_DIS_s016
                                 + 0.696512 w_k1_FDX_s016
                                 + 0.715074 w_k1_IBM_s016
                                 + 0.754908 w_k1_INTC_s016
                                 + 0.710408 w_k1_KO_s016
                                 + 0.496178 w_k1_MAT_s016
                                 + 0.693922 w_k1_MSFT_s016
                                 + 0.74007 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.830163 w_k1_ADM_s016
                                 + 0.832237 w_k1_AXP_s016
                                 + 0.915705 w_k1_DIS_s016
                                 + 0.890407 w_k1_FDX_s016
                                 + 0.72928 w_k1_IBM_s016
                                 + 0.962889 w_k1_INTC_s016
                                 + 0.859841 w_k1_KO_s016
                                 + 0.79326 w_k1_MAT_s016
                                 + 0.980371 w_k1_MSFT_s016
                                 + 0.945021 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.778131 w_k1_ADM_s016
                                 + 1.13522 w_k1_AXP_s016
                                 + 1.17645 w_k1_DIS_s016 + 1.2252 w_k1_FDX_s016
                                 + 0.680287 w_k1_IBM_s016
                                 + 1.37989 w_k1_INTC_s016
                                 + 0.777579 w_k1_KO_s016
                                 + 0.605234 w_k1_MAT_s016
                                 + 1.1062 w_k1_MSFT_s016
                                 + 0.995349 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.03869 w_k1_ADM_s016
                                 + 0.965122 w_k1_AXP_s016
                                 + 1.2788 w_k1_DIS_s016 + 1.14843 w_k1_FDX_s016
                                 + 0.928572 w_k1_IBM_s016
                                 + 1.11527 w_k1_INTC_s016
                                 + 1.11763 w_k1_KO_s016
                                 + 0.758684 w_k1_MAT_s016
                                 + 1.19802 w_k1_MSFT_s016
                                 + 1.09281 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.14866 w_k1_ADM_s017
                                 + 1.29044 w_k1_AXP_s017
                                 + 1.63907 w_k1_DIS_s017
                                 + 1.15479 w_k1_FDX_s017
                                 + 1.36604 w_k1_IBM_s017
                                 + 1.20663 w_k1_INTC_s017 + 1.3065 w_k1_KO_s017
                                 + 1.28164 w_k1_MAT_s017
                                 + 1.21448 w_k1_MSFT_s017
                                 + 1.06645 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.901468 w_k1_ADM_s017
                                 + 0.721486 w_k1_AXP_s017
                                 + 0.83503 w_k1_DIS_s017
                                 + 0.867475 w_k1_FDX_s017
                                 + 0.960631 w_k1_IBM_s017
                                 + 1.02693 w_k1_INTC_s017
                                 + 0.898715 w_k1_KO_s017
                                 + 0.761318 w_k1_MAT_s017
                                 + 0.909423 w_k1_MSFT_s017
                                 + 0.817168 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.27486 w_k1_ADM_s017
                                 + 0.96477 w_k1_AXP_s017
                                 + 0.814357 w_k1_DIS_s017
                                 + 1.00764 w_k1_FDX_s017
                                 + 0.566363 w_k1_IBM_s017
                                 + 0.947373 w_k1_INTC_s017
                                 + 0.730233 w_k1_KO_s017
                                 + 0.414378 w_k1_MAT_s017
                                 + 1.02144 w_k1_MSFT_s017
                                 + 0.802961 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.90519 w_k1_ADM_s017
                                 + 1.39743 w_k1_AXP_s017
                                 + 0.970108 w_k1_DIS_s017
                                 + 1.02737 w_k1_FDX_s017
                                 + 1.22766 w_k1_IBM_s017
                                 + 1.17422 w_k1_INTC_s017
                                 + 1.03705 w_k1_KO_s017 + 1.36804 w_k1_MAT_s017
                                 + 1.06909 w_k1_MSFT_s017
                                 + 0.984439 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.12018 w_k1_ADM_s017
                                 + 0.880479 w_k1_AXP_s017
                                 + 1.13268 w_k1_DIS_s017
                                 + 1.04009 w_k1_FDX_s017
                                 + 0.764828 w_k1_IBM_s017
                                 + 1.02165 w_k1_INTC_s017
                                 + 0.848617 w_k1_KO_s017
                                 + 0.806349 w_k1_MAT_s017
                                 + 1.00006 w_k1_MSFT_s017
                                 + 0.826243 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.14652 w_k1_ADM_s017
                                 + 1.05182 w_k1_AXP_s017
                                 + 1.17667 w_k1_DIS_s017
                                 + 0.916747 w_k1_FDX_s017
                                 + 1.47925 w_k1_IBM_s017
                                 + 1.13413 w_k1_INTC_s017
                                 + 1.40753 w_k1_KO_s017 + 1.04424 w_k1_MAT_s017
                                 + 1.09529 w_k1_MSFT_s017
                                 + 0.973715 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.853094 w_k1_ADM_s017
                                 + 0.698772 w_k1_AXP_s017
                                 + 0.700788 w_k1_DIS_s017
                                 + 0.689997 w_k1_FDX_s017
                                 + 0.749211 w_k1_IBM_s017
                                 + 0.797646 w_k1_INTC_s017
                                 + 0.746703 w_k1_KO_s017
                                 + 0.616284 w_k1_MAT_s017
                                 + 0.869858 w_k1_MSFT_s017
                                 + 0.696836 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.27103 w_k1_ADM_s017
                                 + 1.26446 w_k1_AXP_s017
                                 + 1.08029 w_k1_DIS_s017
                                 + 1.31303 w_k1_FDX_s017
                                 + 0.80673 w_k1_IBM_s017
                                 + 1.10034 w_k1_INTC_s017
                                 + 1.00583 w_k1_KO_s017
                                 + 0.812014 w_k1_MAT_s017
                                 + 1.12399 w_k1_MSFT_s017
                                 + 0.967839 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.01325 w_k1_ADM_s100
                                 + 0.982964 w_k1_AXP_s100
                                 + 0.931195 w_k1_DIS_s100
                                 + 0.934569 w_k1_FDX_s100
                                 + 1.04448 w_k1_IBM_s100
                                 + 1.10724 w_k1_INTC_s100
                                 + 0.948478 w_k1_KO_s100
                                 + 0.924366 w_k1_MAT_s100
                                 + 0.998479 w_k1_MSFT_s100
                                 + 1.06442 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.726473 w_k1_ADM_s100
                                 + 0.848374 w_k1_AXP_s100
                                 + 0.773471 w_k1_DIS_s100
                                 + 0.853962 w_k1_FDX_s100
                                 + 0.713456 w_k1_IBM_s100
                                 + 1.05099 w_k1_INTC_s100
                                 + 0.873826 w_k1_KO_s100
                                 + 0.63751 w_k1_MAT_s100
                                 + 0.959488 w_k1_MSFT_s100
                                 + 0.835933 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.860846 w_k1_ADM_s100
                                 + 1.09895 w_k1_AXP_s100
                                 + 0.952989 w_k1_DIS_s100
                                 + 0.895384 w_k1_FDX_s100
                                 + 1.38904 w_k1_IBM_s100
                                 + 0.833459 w_k1_INTC_s100
                                 + 1.08084 w_k1_KO_s100 + 1.35272 w_k1_MAT_s100
                                 + 0.901185 w_k1_MSFT_s100
                                 + 1.04608 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.28121 w_k1_ADM_s100
                                 + 1.10836 w_k1_AXP_s100
                                 + 1.53386 w_k1_DIS_s100
                                 + 1.20684 w_k1_FDX_s100
                                 + 1.08382 w_k1_IBM_s100
                                 + 0.981273 w_k1_INTC_s100
                                 + 1.08813 w_k1_KO_s100
                                 + 0.878839 w_k1_MAT_s100
                                 + 1.19638 w_k1_MSFT_s100
                                 + 1.18441 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.726069 w_k1_ADM_s100
                                 + 0.727951 w_k1_AXP_s100
                                 + 0.577766 w_k1_DIS_s100
                                 + 0.624364 w_k1_FDX_s100
                                 + 1.01619 w_k1_IBM_s100
                                 + 0.770641 w_k1_INTC_s100
                                 + 0.810847 w_k1_KO_s100
                                 + 0.861874 w_k1_MAT_s100
                                 + 0.871744 w_k1_MSFT_s100
                                 + 0.881701 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.25928 w_k1_ADM_s100
                                 + 0.92489 w_k1_AXP_s100
                                 + 1.22064 w_k1_DIS_s100 + 1.206 w_k1_FDX_s100
                                 + 0.711054 w_k1_IBM_s100
                                 + 1.15197 w_k1_INTC_s100
                                 + 1.07724 w_k1_KO_s100
                                 + 0.591598 w_k1_MAT_s100
                                 + 1.327 w_k1_MSFT_s100 + 1.05818 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.956877 w_k1_ADM_s100
                                 + 1.33224 w_k1_AXP_s100
                                 + 1.25268 w_k1_DIS_s100
                                 + 0.986529 w_k1_FDX_s100
                                 + 1.6459 w_k1_IBM_s100
                                 + 1.32552 w_k1_INTC_s100
                                 + 1.55762 w_k1_KO_s100 + 1.53728 w_k1_MAT_s100
                                 + 1.28956 w_k1_MSFT_s100
                                 + 1.20917 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.948248 w_k1_ADM_s100
                                 + 1.6011 w_k1_AXP_s100 + 1.48452 w_k1_DIS_s100
                                 + 1.22386 w_k1_FDX_s100 + 1.2998 w_k1_IBM_s100
                                 + 1.25474 w_k1_INTC_s100
                                 + 1.44657 w_k1_KO_s100 + 1.55243 w_k1_MAT_s100
                                 + 1.14931 w_k1_MSFT_s100
                                 + 1.28644 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.711836 w_k1_ADM_s101
                                 + 0.582051 w_k1_AXP_s101
                                 + 0.54356 w_k1_DIS_s101
                                 + 0.675235 w_k1_FDX_s101
                                 + 0.667542 w_k1_IBM_s101
                                 + 0.676454 w_k1_INTC_s101
                                 + 0.622232 w_k1_KO_s101
                                 + 0.539464 w_k1_MAT_s101
                                 + 0.731831 w_k1_MSFT_s101
                                 + 0.760545 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.876696 w_k1_ADM_s101
                                 + 1.03451 w_k1_AXP_s101
                                 + 1.04533 w_k1_DIS_s101
                                 + 0.952187 w_k1_FDX_s101
                                 + 1.1778 w_k1_IBM_s101
                                 + 0.944104 w_k1_INTC_s101
                                 + 1.1007 w_k1_KO_s101 + 1.3679 w_k1_MAT_s101
                                 + 1.12427 w_k1_MSFT_s101
                                 + 1.01176 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.0777 w_k1_ADM_s101
                                 + 1.68333 w_k1_AXP_s101
                                 + 1.81062 w_k1_DIS_s101
                                 + 1.32826 w_k1_FDX_s101
                                 + 1.40416 w_k1_IBM_s101
                                 + 1.27071 w_k1_INTC_s101
                                 + 1.37347 w_k1_KO_s101 + 1.76158 w_k1_MAT_s101
                                 + 1.32758 w_k1_MSFT_s101
                                 + 1.33885 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.790255 w_k1_ADM_s101
                                 + 0.822923 w_k1_AXP_s101
                                 + 0.600482 w_k1_DIS_s101
                                 + 0.647489 w_k1_FDX_s101
                                 + 1.10206 w_k1_IBM_s101
                                 + 0.580292 w_k1_INTC_s101
                                 + 0.797905 w_k1_KO_s101
                                 + 1.07562 w_k1_MAT_s101
                                 + 0.680896 w_k1_MSFT_s101
                                 + 0.840043 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.949889 w_k1_ADM_s101
                                 + 1.30274 w_k1_AXP_s101
                                 + 1.40531 w_k1_DIS_s101
                                 + 1.14426 w_k1_FDX_s101 + 1.3252 w_k1_IBM_s101
                                 + 0.887451 w_k1_INTC_s101
                                 + 1.12738 w_k1_KO_s101 + 1.31868 w_k1_MAT_s101
                                 + 1.13239 w_k1_MSFT_s101
                                 + 1.09079 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.28748 w_k1_ADM_s101
                                 + 1.06255 w_k1_AXP_s101 + 1.4817 w_k1_DIS_s101
                                 + 1.20549 w_k1_FDX_s101
                                 + 0.880098 w_k1_IBM_s101
                                 + 1.10773 w_k1_INTC_s101
                                 + 1.12761 w_k1_KO_s101
                                 + 0.756125 w_k1_MAT_s101
                                 + 1.25869 w_k1_MSFT_s101
                                 + 1.21775 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.914821 w_k1_ADM_s101
                                 + 0.949453 w_k1_AXP_s101
                                 + 0.869758 w_k1_DIS_s101
                                 + 0.963998 w_k1_FDX_s101
                                 + 0.687645 w_k1_IBM_s101
                                 + 0.83982 w_k1_INTC_s101
                                 + 0.763767 w_k1_KO_s101
                                 + 0.808633 w_k1_MAT_s101
                                 + 0.873309 w_k1_MSFT_s101
                                 + 0.903285 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.789806 w_k1_ADM_s101
                                 + 1.00128 w_k1_AXP_s101
                                 + 0.84036 w_k1_DIS_s101
                                 + 0.898805 w_k1_FDX_s101
                                 + 0.922988 w_k1_IBM_s101
                                 + 1.01323 w_k1_INTC_s101
                                 + 0.915769 w_k1_KO_s101
                                 + 0.881968 w_k1_MAT_s101
                                 + 0.916046 w_k1_MSFT_s101
                                 + 0.961534 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.692466 w_k1_ADM_s102
                                 + 0.608616 w_k1_AXP_s102
                                 + 0.673169 w_k1_DIS_s102
                                 + 0.684603 w_k1_FDX_s102
                                 + 0.924008 w_k1_IBM_s102
                                 + 0.739045 w_k1_INTC_s102
                                 + 0.814537 w_k1_KO_s102
                                 + 0.70564 w_k1_MAT_s102
                                 + 0.590957 w_k1_MSFT_s102
                                 + 0.709944 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.947251 w_k1_ADM_s102
                                 + 0.816777 w_k1_AXP_s102
                                 + 0.950661 w_k1_DIS_s102
                                 + 0.917525 w_k1_FDX_s102
                                 + 0.714768 w_k1_IBM_s102
                                 + 1.05041 w_k1_INTC_s102
                                 + 0.812371 w_k1_KO_s102
                                 + 0.583121 w_k1_MAT_s102
                                 + 0.874922 w_k1_MSFT_s102
                                 + 0.886475 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.18483 w_k1_ADM_s102
                                 + 1.29288 w_k1_AXP_s102
                                 + 1.43886 w_k1_DIS_s102
                                 + 1.15099 w_k1_FDX_s102
                                 + 1.06581 w_k1_IBM_s102
                                 + 1.20771 w_k1_INTC_s102
                                 + 1.20412 w_k1_KO_s102
                                 + 0.925993 w_k1_MAT_s102
                                 + 1.02071 w_k1_MSFT_s102
                                 + 1.10776 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.809205 w_k1_ADM_s102
                                 + 0.643671 w_k1_AXP_s102
                                 + 1.31463 w_k1_DIS_s102
                                 + 0.994452 w_k1_FDX_s102
                                 + 1.02684 w_k1_IBM_s102
                                 + 1.09363 w_k1_INTC_s102
                                 + 1.09762 w_k1_KO_s102
                                 + 0.721267 w_k1_MAT_s102
                                 + 0.997667 w_k1_MSFT_s102
                                 + 0.960604 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.973327 w_k1_ADM_s102
                                 + 0.98305 w_k1_AXP_s102
                                 + 0.737038 w_k1_DIS_s102
                                 + 0.839624 w_k1_FDX_s102
                                 + 0.804234 w_k1_IBM_s102
                                 + 1.00949 w_k1_INTC_s102
                                 + 0.829174 w_k1_KO_s102
                                 + 0.775791 w_k1_MAT_s102
                                 + 0.835067 w_k1_MSFT_s102
                                 + 0.864907 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.27075 w_k1_ADM_s102
                                 + 1.34658 w_k1_AXP_s102
                                 + 1.54017 w_k1_DIS_s102
                                 + 1.28212 w_k1_FDX_s102
                                 + 1.54175 w_k1_IBM_s102
                                 + 1.28004 w_k1_INTC_s102
                                 + 1.42776 w_k1_KO_s102 + 1.23132 w_k1_MAT_s102
                                 + 1.20671 w_k1_MSFT_s102
                                 + 1.17973 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.815031 w_k1_ADM_s102
                                 + 0.943388 w_k1_AXP_s102
                                 + 0.755286 w_k1_DIS_s102
                                 + 0.852268 w_k1_FDX_s102
                                 + 0.977438 w_k1_IBM_s102
                                 + 0.828476 w_k1_INTC_s102
                                 + 0.95673 w_k1_KO_s102 + 1.03158 w_k1_MAT_s102
                                 + 0.789957 w_k1_MSFT_s102
                                 + 0.872174 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.92788 w_k1_ADM_s102
                                 + 0.925449 w_k1_AXP_s102
                                 + 1.01853 w_k1_DIS_s102
                                 + 0.919002 w_k1_FDX_s102
                                 + 1.23181 w_k1_IBM_s102
                                 + 1.0233 w_k1_INTC_s102 + 1.00291 w_k1_KO_s102
                                 + 1.07773 w_k1_MAT_s102
                                 + 0.935621 w_k1_MSFT_s102
                                 + 0.8609 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.04961 w_k1_ADM_s103
                                 + 1.16847 w_k1_AXP_s103
                                 + 0.850995 w_k1_DIS_s103
                                 + 1.05248 w_k1_FDX_s103
                                 + 1.28242 w_k1_IBM_s103
                                 + 1.11798 w_k1_INTC_s103
                                 + 1.07295 w_k1_KO_s103 + 1.54257 w_k1_MAT_s103
                                 + 1.05503 w_k1_MSFT_s103
                                 + 1.03292 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.03711 w_k1_ADM_s103
                                 + 0.842719 w_k1_AXP_s103
                                 + 0.840417 w_k1_DIS_s103
                                 + 0.957607 w_k1_FDX_s103
                                 + 0.742806 w_k1_IBM_s103
                                 + 0.886644 w_k1_INTC_s103
                                 + 0.73262 w_k1_KO_s103 + 1.17657 w_k1_MAT_s103
                                 + 1.01115 w_k1_MSFT_s103
                                 + 0.872229 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.66004 w_k1_ADM_s103
                                 + 0.522869 w_k1_AXP_s103
                                 + 0.767268 w_k1_DIS_s103
                                 + 0.78693 w_k1_FDX_s103
                                 + 0.691445 w_k1_IBM_s103
                                 + 0.928322 w_k1_INTC_s103
                                 + 0.769648 w_k1_KO_s103
                                 + 0.566851 w_k1_MAT_s103
                                 + 0.981239 w_k1_MSFT_s103
                                 + 0.814951 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.17359 w_k1_ADM_s103
                                 + 1.63493 w_k1_AXP_s103
                                 + 1.95911 w_k1_DIS_s103
                                 + 1.38857 w_k1_FDX_s103
                                 + 1.69107 w_k1_IBM_s103
                                 + 1.37836 w_k1_INTC_s103
                                 + 1.74691 w_k1_KO_s103 + 2.35774 w_k1_MAT_s103
                                 + 1.81506 w_k1_MSFT_s103
                                 + 1.59949 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.770659 w_k1_ADM_s103
                                 + 0.844299 w_k1_AXP_s103
                                 + 0.840583 w_k1_DIS_s103
                                 + 0.905212 w_k1_FDX_s103
                                 + 1.08303 w_k1_IBM_s103
                                 + 0.890579 w_k1_INTC_s103
                                 + 1.07352 w_k1_KO_s103 + 1.62485 w_k1_MAT_s103
                                 + 1.06396 w_k1_MSFT_s103
                                 + 0.943809 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.15084 w_k1_ADM_s103
                                 + 0.939515 w_k1_AXP_s103
                                 + 1.11349 w_k1_DIS_s103
                                 + 0.946314 w_k1_FDX_s103
                                 + 1.39346 w_k1_IBM_s103
                                 + 0.949992 w_k1_INTC_s103
                                 + 1.34052 w_k1_KO_s103 + 1.21454 w_k1_MAT_s103
                                 + 1.11359 w_k1_MSFT_s103
                                 + 1.08371 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.49436 w_k1_ADM_s103
                                 + 1.02825 w_k1_AXP_s103
                                 + 1.20404 w_k1_DIS_s103
                                 + 1.26274 w_k1_FDX_s103
                                 + 0.67293 w_k1_IBM_s103
                                 + 1.29057 w_k1_INTC_s103
                                 + 0.973161 w_k1_KO_s103
                                 + 1.01301 w_k1_MAT_s103
                                 + 1.24948 w_k1_MSFT_s103
                                 + 0.967832 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.05439 w_k1_ADM_s103
                                 + 1.35706 w_k1_AXP_s103 + 1.3627 w_k1_DIS_s103
                                 + 1.14087 w_k1_FDX_s103
                                 + 1.73336 w_k1_IBM_s103
                                 + 0.999306 w_k1_INTC_s103
                                 + 1.34383 w_k1_KO_s103 + 2.44358 w_k1_MAT_s103
                                 + 1.04 w_k1_MSFT_s103 + 1.24127 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.2351 w_k1_ADM_s104
                                 + 1.1852 w_k1_AXP_s104
                                 + 0.892933 w_k1_DIS_s104
                                 + 0.938583 w_k1_FDX_s104
                                 + 1.18427 w_k1_IBM_s104
                                 + 0.789664 w_k1_INTC_s104
                                 + 1.14259 w_k1_KO_s104 + 1.51793 w_k1_MAT_s104
                                 + 1.05542 w_k1_MSFT_s104
                                 + 1.17526 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.00433 w_k1_ADM_s104
                                 + 0.774514 w_k1_AXP_s104
                                 + 0.588727 w_k1_DIS_s104
                                 + 0.743367 w_k1_FDX_s104
                                 + 0.881997 w_k1_IBM_s104
                                 + 0.64672 w_k1_INTC_s104
                                 + 0.844247 w_k1_KO_s104
                                 + 0.687494 w_k1_MAT_s104
                                 + 0.768322 w_k1_MSFT_s104
                                 + 0.927824 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.32027 w_k1_ADM_s104
                                 + 1.38892 w_k1_AXP_s104 + 1.7082 w_k1_DIS_s104
                                 + 1.36131 w_k1_FDX_s104
                                 + 1.01476 w_k1_IBM_s104
                                 + 1.09089 w_k1_INTC_s104
                                 + 1.35768 w_k1_KO_s104 + 1.02804 w_k1_MAT_s104
                                 + 1.30548 w_k1_MSFT_s104
                                 + 1.41194 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.830367 w_k1_ADM_s104
                                 + 0.675426 w_k1_AXP_s104
                                 + 0.801563 w_k1_DIS_s104
                                 + 0.921761 w_k1_FDX_s104
                                 + 0.958834 w_k1_IBM_s104
                                 + 0.970684 w_k1_INTC_s104
                                 + 0.989888 w_k1_KO_s104
                                 + 0.742317 w_k1_MAT_s104
                                 + 0.986043 w_k1_MSFT_s104
                                 + 0.906911 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.29827 w_k1_ADM_s104
                                 + 1.12177 w_k1_AXP_s104
                                 + 1.18553 w_k1_DIS_s104
                                 + 1.15202 w_k1_FDX_s104
                                 + 0.93303 w_k1_IBM_s104
                                 + 0.872398 w_k1_INTC_s104
                                 + 1.03801 w_k1_KO_s104 + 1.04748 w_k1_MAT_s104
                                 + 1.23838 w_k1_MSFT_s104
                                 + 1.13748 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.37638 w_k1_ADM_s104
                                 + 1.36166 w_k1_AXP_s104
                                 + 1.48895 w_k1_DIS_s104
                                 + 1.34518 w_k1_FDX_s104
                                 + 1.78904 w_k1_IBM_s104
                                 + 1.09859 w_k1_INTC_s104
                                 + 1.64148 w_k1_KO_s104 + 1.87767 w_k1_MAT_s104
                                 + 1.22559 w_k1_MSFT_s104
                                 + 1.32738 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.758903 w_k1_ADM_s104
                                 + 0.765274 w_k1_AXP_s104
                                 + 0.727754 w_k1_DIS_s104
                                 + 0.86722 w_k1_FDX_s104
                                 + 0.685946 w_k1_IBM_s104
                                 + 0.946144 w_k1_INTC_s104
                                 + 0.665932 w_k1_KO_s104
                                 + 0.55552 w_k1_MAT_s104
                                 + 0.915565 w_k1_MSFT_s104
                                 + 0.860775 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.33289 w_k1_ADM_s104
                                 + 0.938788 w_k1_AXP_s104
                                 + 0.83852 w_k1_DIS_s104
                                 + 0.766136 w_k1_FDX_s104
                                 + 1.30667 w_k1_IBM_s104
                                 + 0.872483 w_k1_INTC_s104
                                 + 1.26275 w_k1_KO_s104 + 1.09604 w_k1_MAT_s104
                                 + 0.950497 w_k1_MSFT_s104
                                 + 1.08524 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.03449 w_k1_ADM_s105
                                 + 1.11342 w_k1_AXP_s105
                                 + 0.758646 w_k1_DIS_s105
                                 + 1.13861 w_k1_FDX_s105
                                 + 0.761836 w_k1_IBM_s105
                                 + 0.952707 w_k1_INTC_s105
                                 + 0.803143 w_k1_KO_s105
                                 + 0.828133 w_k1_MAT_s105
                                 + 1.08332 w_k1_MSFT_s105
                                 + 0.984885 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.41887 w_k1_ADM_s105
                                 + 1.35611 w_k1_AXP_s105
                                 + 1.69138 w_k1_DIS_s105
                                 + 1.45302 w_k1_FDX_s105
                                 + 1.08892 w_k1_IBM_s105
                                 + 1.60987 w_k1_INTC_s105
                                 + 1.42118 w_k1_KO_s105 + 0.92804 w_k1_MAT_s105
                                 + 1.57893 w_k1_MSFT_s105 + 1.5423 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.03786 w_k1_ADM_s105
                                 + 1.11996 w_k1_AXP_s105
                                 + 1.04911 w_k1_DIS_s105
                                 + 0.949036 w_k1_FDX_s105
                                 + 1.40704 w_k1_IBM_s105
                                 + 1.02561 w_k1_INTC_s105
                                 + 1.09197 w_k1_KO_s105 + 1.06289 w_k1_MAT_s105
                                 + 1.02449 w_k1_MSFT_s105
                                 + 1.12459 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.184 w_k1_ADM_s105
                                 + 1.13361 w_k1_AXP_s105
                                 + 1.12426 w_k1_DIS_s105
                                 + 1.09807 w_k1_FDX_s105
                                 + 0.873417 w_k1_IBM_s105
                                 + 1.00162 w_k1_INTC_s105
                                 + 1.04876 w_k1_KO_s105
                                 + 0.737284 w_k1_MAT_s105
                                 + 1.23895 w_k1_MSFT_s105 + 1.1058 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.27861 w_k1_ADM_s105
                                 + 1.20527 w_k1_AXP_s105
                                 + 1.31234 w_k1_DIS_s105
                                 + 1.21829 w_k1_FDX_s105
                                 + 1.33691 w_k1_IBM_s105
                                 + 1.13716 w_k1_INTC_s105
                                 + 1.29367 w_k1_KO_s105 + 1.22849 w_k1_MAT_s105
                                 + 1.11684 w_k1_MSFT_s105
                                 + 1.22142 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.756868 w_k1_ADM_s105
                                 + 0.929656 w_k1_AXP_s105
                                 + 0.770571 w_k1_DIS_s105
                                 + 0.865435 w_k1_FDX_s105
                                 + 1.10036 w_k1_IBM_s105
                                 + 0.840808 w_k1_INTC_s105
                                 + 0.935917 w_k1_KO_s105
                                 + 0.85781 w_k1_MAT_s105
                                 + 0.853559 w_k1_MSFT_s105
                                 + 1.04077 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.805823 w_k1_ADM_s105
                                 + 0.815733 w_k1_AXP_s105
                                 + 0.55719 w_k1_DIS_s105
                                 + 0.745803 w_k1_FDX_s105
                                 + 0.735363 w_k1_IBM_s105
                                 + 0.906789 w_k1_INTC_s105
                                 + 0.850087 w_k1_KO_s105
                                 + 0.616325 w_k1_MAT_s105
                                 + 0.737312 w_k1_MSFT_s105
                                 + 0.928085 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.771616 w_k1_ADM_s105
                                 + 0.731832 w_k1_AXP_s105
                                 + 0.922642 w_k1_DIS_s105
                                 + 0.952413 w_k1_FDX_s105
                                 + 0.689243 w_k1_IBM_s105
                                 + 0.870066 w_k1_INTC_s105
                                 + 0.796031 w_k1_KO_s105
                                 + 0.58445 w_k1_MAT_s105
                                 + 0.995639 w_k1_MSFT_s105
                                 + 1.08927 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.24211 w_k1_ADM_s106
                                 + 1.05936 w_k1_AXP_s106
                                 + 0.999681 w_k1_DIS_s106
                                 + 0.9755 w_k1_FDX_s106 + 1.16723 w_k1_IBM_s106
                                 + 0.864079 w_k1_INTC_s106
                                 + 0.93116 w_k1_KO_s106 + 1.31545 w_k1_MAT_s106
                                 + 0.832153 w_k1_MSFT_s106
                                 + 1.01756 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06739 w_k1_ADM_s106
                                 + 0.703833 w_k1_AXP_s106
                                 + 0.872841 w_k1_DIS_s106
                                 + 0.793405 w_k1_FDX_s106
                                 + 0.905758 w_k1_IBM_s106
                                 + 0.752953 w_k1_INTC_s106
                                 + 1.00204 w_k1_KO_s106
                                 + 0.786177 w_k1_MAT_s106
                                 + 0.812561 w_k1_MSFT_s106
                                 + 0.954833 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.33045 w_k1_ADM_s106
                                 + 1.34414 w_k1_AXP_s106
                                 + 1.46118 w_k1_DIS_s106
                                 + 1.17702 w_k1_FDX_s106
                                 + 1.36564 w_k1_IBM_s106
                                 + 1.01773 w_k1_INTC_s106
                                 + 1.20361 w_k1_KO_s106 + 1.67328 w_k1_MAT_s106
                                 + 1.01222 w_k1_MSFT_s106 + 1.0072 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.09102 w_k1_ADM_s106
                                 + 0.938209 w_k1_AXP_s106
                                 + 0.691516 w_k1_DIS_s106
                                 + 0.847545 w_k1_FDX_s106
                                 + 0.935189 w_k1_IBM_s106
                                 + 0.678089 w_k1_INTC_s106
                                 + 0.835858 w_k1_KO_s106
                                 + 1.24562 w_k1_MAT_s106
                                 + 0.658404 w_k1_MSFT_s106
                                 + 0.956047 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.36792 w_k1_ADM_s106
                                 + 1.57779 w_k1_AXP_s106 + 1.8929 w_k1_DIS_s106
                                 + 1.48771 w_k1_FDX_s106
                                 + 1.29255 w_k1_IBM_s106
                                 + 1.18549 w_k1_INTC_s106
                                 + 1.36772 w_k1_KO_s106 + 1.72874 w_k1_MAT_s106
                                 + 1.13973 w_k1_MSFT_s106
                                 + 1.21317 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.853939 w_k1_ADM_s106
                                 + 0.90209 w_k1_AXP_s106
                                 + 0.860808 w_k1_DIS_s106
                                 + 0.944118 w_k1_FDX_s106
                                 + 0.812922 w_k1_IBM_s106
                                 + 0.967019 w_k1_INTC_s106
                                 + 0.900507 w_k1_KO_s106
                                 + 0.775647 w_k1_MAT_s106
                                 + 0.89508 w_k1_MSFT_s106
                                 + 0.891477 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.786428 w_k1_ADM_s106
                                 + 0.658111 w_k1_AXP_s106
                                 + 0.613291 w_k1_DIS_s106
                                 + 0.678917 w_k1_FDX_s106
                                 + 0.75523 w_k1_IBM_s106
                                 + 0.679542 w_k1_INTC_s106
                                 + 0.66696 w_k1_KO_s106
                                 + 0.936119 w_k1_MAT_s106
                                 + 0.673426 w_k1_MSFT_s106
                                 + 0.761989 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.25775 w_k1_ADM_s106
                                 + 0.898515 w_k1_AXP_s106
                                 + 1.30638 w_k1_DIS_s106
                                 + 1.15752 w_k1_FDX_s106
                                 + 1.00338 w_k1_IBM_s106
                                 + 1.0358 w_k1_INTC_s106 + 1.13371 w_k1_KO_s106
                                 + 0.834543 w_k1_MAT_s106
                                 + 1.09899 w_k1_MSFT_s106
                                 + 1.01208 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08558 w_k1_ADM_s107
                                 + 1.53109 w_k1_AXP_s107
                                 + 1.68569 w_k1_DIS_s107
                                 + 1.39205 w_k1_FDX_s107
                                 + 1.37233 w_k1_IBM_s107
                                 + 1.5696 w_k1_INTC_s107 + 1.3354 w_k1_KO_s107
                                 + 1.24694 w_k1_MAT_s107
                                 + 1.36454 w_k1_MSFT_s107
                                 + 1.10942 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.15111 w_k1_ADM_s107
                                 + 1.37562 w_k1_AXP_s107
                                 + 1.16268 w_k1_DIS_s107
                                 + 1.17419 w_k1_FDX_s107
                                 + 0.991068 w_k1_IBM_s107
                                 + 1.30292 w_k1_INTC_s107
                                 + 1.04434 w_k1_KO_s107
                                 + 0.909337 w_k1_MAT_s107
                                 + 1.18436 w_k1_MSFT_s107
                                 + 1.06347 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.906088 w_k1_ADM_s107
                                 + 0.864582 w_k1_AXP_s107
                                 + 0.919228 w_k1_DIS_s107
                                 + 0.880839 w_k1_FDX_s107
                                 + 1.35818 w_k1_IBM_s107
                                 + 0.758863 w_k1_INTC_s107
                                 + 1.10522 w_k1_KO_s107 + 1.31691 w_k1_MAT_s107
                                 + 0.815616 w_k1_MSFT_s107
                                 + 1.04965 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.25665 w_k1_ADM_s107
                                 + 1.07767 w_k1_AXP_s107
                                 + 1.20046 w_k1_DIS_s107
                                 + 1.26007 w_k1_FDX_s107
                                 + 0.794549 w_k1_IBM_s107
                                 + 1.02414 w_k1_INTC_s107
                                 + 0.965522 w_k1_KO_s107
                                 + 0.763976 w_k1_MAT_s107
                                 + 1.10806 w_k1_MSFT_s107
                                 + 1.04966 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.86929 w_k1_ADM_s107
                                 + 0.781503 w_k1_AXP_s107
                                 + 0.655352 w_k1_DIS_s107
                                 + 0.861036 w_k1_FDX_s107
                                 + 0.709654 w_k1_IBM_s107
                                 + 0.885497 w_k1_INTC_s107
                                 + 0.775199 w_k1_KO_s107
                                 + 0.745271 w_k1_MAT_s107
                                 + 0.901394 w_k1_MSFT_s107
                                 + 0.893464 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.869476 w_k1_ADM_s107
                                 + 0.999665 w_k1_AXP_s107
                                 + 1.0347 w_k1_DIS_s107 + 1.0205 w_k1_FDX_s107
                                 + 1.03228 w_k1_IBM_s107
                                 + 1.23333 w_k1_INTC_s107
                                 + 0.972121 w_k1_KO_s107
                                 + 0.686262 w_k1_MAT_s107
                                 + 1.03097 w_k1_MSFT_s107
                                 + 0.928923 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.732984 w_k1_ADM_s107
                                 + 0.652697 w_k1_AXP_s107
                                 + 0.820367 w_k1_DIS_s107
                                 + 0.806688 w_k1_FDX_s107
                                 + 0.80183 w_k1_IBM_s107
                                 + 1.01344 w_k1_INTC_s107
                                 + 0.946523 w_k1_KO_s107
                                 + 0.743045 w_k1_MAT_s107
                                 + 0.784506 w_k1_MSFT_s107
                                 + 0.79587 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.00721 w_k1_ADM_s107
                                 + 0.986256 w_k1_AXP_s107
                                 + 0.857178 w_k1_DIS_s107
                                 + 0.918399 w_k1_FDX_s107
                                 + 0.928227 w_k1_IBM_s107
                                 + 1.0213 w_k1_INTC_s107
                                 + 0.974004 w_k1_KO_s107
                                 + 0.947929 w_k1_MAT_s107
                                 + 0.896725 w_k1_MSFT_s107
                                 + 0.918389 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.558889 w_k1_ADM_s110
                                 + 0.626163 w_k1_AXP_s110
                                 + 0.573521 w_k1_DIS_s110
                                 + 0.73286 w_k1_FDX_s110
                                 + 0.708132 w_k1_IBM_s110
                                 + 0.681623 w_k1_INTC_s110
                                 + 0.652258 w_k1_KO_s110
                                 + 0.704583 w_k1_MAT_s110
                                 + 0.672437 w_k1_MSFT_s110
                                 + 0.738199 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01935 w_k1_ADM_s110
                                 + 1.31318 w_k1_AXP_s110
                                 + 1.21826 w_k1_DIS_s110
                                 + 1.19741 w_k1_FDX_s110
                                 + 0.842244 w_k1_IBM_s110
                                 + 1.1109 w_k1_INTC_s110 + 1.06561 w_k1_KO_s110
                                 + 0.990915 w_k1_MAT_s110
                                 + 1.07635 w_k1_MSFT_s110 + 1.1283 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.925343 w_k1_ADM_s110
                                 + 0.884247 w_k1_AXP_s110
                                 + 0.862875 w_k1_DIS_s110
                                 + 0.91697 w_k1_FDX_s110
                                 + 0.683068 w_k1_IBM_s110
                                 + 0.908458 w_k1_INTC_s110
                                 + 0.877276 w_k1_KO_s110
                                 + 0.643237 w_k1_MAT_s110
                                 + 0.983625 w_k1_MSFT_s110
                                 + 0.949246 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.755905 w_k1_ADM_s110
                                 + 0.979173 w_k1_AXP_s110
                                 + 1.10338 w_k1_DIS_s110
                                 + 0.966848 w_k1_FDX_s110
                                 + 1.31017 w_k1_IBM_s110
                                 + 1.23732 w_k1_INTC_s110 + 1.356 w_k1_KO_s110
                                 + 1.09888 w_k1_MAT_s110
                                 + 1.18342 w_k1_MSFT_s110
                                 + 1.03544 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.901179 w_k1_ADM_s110
                                 + 0.907851 w_k1_AXP_s110
                                 + 0.798248 w_k1_DIS_s110
                                 + 0.820363 w_k1_FDX_s110
                                 + 0.932266 w_k1_IBM_s110
                                 + 0.704101 w_k1_INTC_s110
                                 + 0.880688 w_k1_KO_s110
                                 + 1.05589 w_k1_MAT_s110
                                 + 0.758817 w_k1_MSFT_s110
                                 + 0.986371 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.924609 w_k1_ADM_s110
                                 + 1.12649 w_k1_AXP_s110
                                 + 1.25631 w_k1_DIS_s110
                                 + 1.05016 w_k1_FDX_s110
                                 + 1.00073 w_k1_IBM_s110
                                 + 1.15723 w_k1_INTC_s110
                                 + 1.11076 w_k1_KO_s110 + 1.23916 w_k1_MAT_s110
                                 + 1.17686 w_k1_MSFT_s110 + 1.1304 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.06675 w_k1_ADM_s110
                                 + 1.45375 w_k1_AXP_s110
                                 + 0.910974 w_k1_DIS_s110
                                 + 1.07326 w_k1_FDX_s110
                                 + 1.15106 w_k1_IBM_s110
                                 + 0.968509 w_k1_INTC_s110
                                 + 1.0571 w_k1_KO_s110 + 1.65225 w_k1_MAT_s110
                                 + 1.00957 w_k1_MSFT_s110
                                 + 1.18842 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.23299 w_k1_ADM_s110
                                 + 1.39382 w_k1_AXP_s110
                                 + 1.60561 w_k1_DIS_s110
                                 + 1.33172 w_k1_FDX_s110
                                 + 1.52369 w_k1_IBM_s110
                                 + 0.993267 w_k1_INTC_s110
                                 + 1.5084 w_k1_KO_s110 + 1.40938 w_k1_MAT_s110
                                 + 1.14674 w_k1_MSFT_s110
                                 + 1.24579 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.1323 w_k1_ADM_s111
                                 + 1.28515 w_k1_AXP_s111
                                 + 1.02438 w_k1_DIS_s111
                                 + 1.08996 w_k1_FDX_s111
                                 + 0.999075 w_k1_IBM_s111
                                 + 0.975591 w_k1_INTC_s111
                                 + 0.860733 w_k1_KO_s111
                                 + 0.764441 w_k1_MAT_s111
                                 + 0.954232 w_k1_MSFT_s111
                                 + 0.957358 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.56076 w_k1_ADM_s111
                                 + 1.14246 w_k1_AXP_s111
                                 + 1.59748 w_k1_DIS_s111
                                 + 1.23502 w_k1_FDX_s111 + 1.3096 w_k1_IBM_s111
                                 + 0.937806 w_k1_INTC_s111
                                 + 1.33887 w_k1_KO_s111
                                 + 0.987762 w_k1_MAT_s111
                                 + 1.11743 w_k1_MSFT_s111
                                 + 1.17859 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.05233 w_k1_ADM_s111
                                 + 0.795608 w_k1_AXP_s111
                                 + 0.757022 w_k1_DIS_s111
                                 + 0.909985 w_k1_FDX_s111
                                 + 0.857708 w_k1_IBM_s111
                                 + 1.01789 w_k1_INTC_s111
                                 + 0.878295 w_k1_KO_s111
                                 + 0.817351 w_k1_MAT_s111
                                 + 0.907129 w_k1_MSFT_s111
                                 + 0.869418 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.930419 w_k1_ADM_s111
                                 + 0.511633 w_k1_AXP_s111
                                 + 0.594046 w_k1_DIS_s111
                                 + 0.691571 w_k1_FDX_s111
                                 + 0.867111 w_k1_IBM_s111
                                 + 0.777842 w_k1_INTC_s111
                                 + 0.807291 w_k1_KO_s111
                                 + 0.563238 w_k1_MAT_s111
                                 + 0.864491 w_k1_MSFT_s111
                                 + 0.843516 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.1052 w_k1_ADM_s111
                                 + 0.690398 w_k1_AXP_s111
                                 + 0.980607 w_k1_DIS_s111
                                 + 1.04672 w_k1_FDX_s111
                                 + 0.679811 w_k1_IBM_s111
                                 + 0.923336 w_k1_INTC_s111
                                 + 0.794092 w_k1_KO_s111
                                 + 0.443585 w_k1_MAT_s111
                                 + 0.905936 w_k1_MSFT_s111
                                 + 0.964257 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.07189 w_k1_ADM_s111
                                 + 1.02676 w_k1_AXP_s111
                                 + 1.04495 w_k1_DIS_s111
                                 + 1.02222 w_k1_FDX_s111
                                 + 1.25717 w_k1_IBM_s111
                                 + 0.895456 w_k1_INTC_s111
                                 + 1.19654 w_k1_KO_s111 + 1.16123 w_k1_MAT_s111
                                 + 1.07376 w_k1_MSFT_s111
                                 + 1.19039 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.44786 w_k1_ADM_s111
                                 + 1.53012 w_k1_AXP_s111
                                 + 1.77469 w_k1_DIS_s111 + 1.5194 w_k1_FDX_s111
                                 + 1.10903 w_k1_IBM_s111
                                 + 1.30994 w_k1_INTC_s111
                                 + 1.36735 w_k1_KO_s111 + 1.42676 w_k1_MAT_s111
                                 + 1.50325 w_k1_MSFT_s111
                                 + 1.25035 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.20351 w_k1_ADM_s111
                                 + 0.957338 w_k1_AXP_s111
                                 + 1.20565 w_k1_DIS_s111
                                 + 1.19622 w_k1_FDX_s111
                                 + 0.778328 w_k1_IBM_s111
                                 + 1.18553 w_k1_INTC_s111
                                 + 0.969473 w_k1_KO_s111
                                 + 0.534288 w_k1_MAT_s111
                                 + 1.23115 w_k1_MSFT_s111
                                 + 1.11952 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.953873 w_k1_ADM_s112
                                 + 0.671959 w_k1_AXP_s112
                                 + 0.656029 w_k1_DIS_s112
                                 + 0.780539 w_k1_FDX_s112
                                 + 0.776973 w_k1_IBM_s112
                                 + 0.896152 w_k1_INTC_s112
                                 + 0.827158 w_k1_KO_s112
                                 + 0.513274 w_k1_MAT_s112
                                 + 0.788194 w_k1_MSFT_s112
                                 + 0.810581 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.09783 w_k1_ADM_s112
                                 + 1.39914 w_k1_AXP_s112
                                 + 1.32575 w_k1_DIS_s112
                                 + 1.22456 w_k1_FDX_s112
                                 + 0.923981 w_k1_IBM_s112
                                 + 1.01048 w_k1_INTC_s112
                                 + 1.00505 w_k1_KO_s112
                                 + 0.874412 w_k1_MAT_s112
                                 + 1.02157 w_k1_MSFT_s112
                                 + 1.01369 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.16402 w_k1_ADM_s112
                                 + 1.25761 w_k1_AXP_s112
                                 + 1.38825 w_k1_DIS_s112
                                 + 1.12823 w_k1_FDX_s112
                                 + 1.76824 w_k1_IBM_s112
                                 + 1.09452 w_k1_INTC_s112
                                 + 1.30029 w_k1_KO_s112 + 1.4918 w_k1_MAT_s112
                                 + 0.919287 w_k1_MSFT_s112
                                 + 1.11675 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.823166 w_k1_ADM_s112
                                 + 1.05021 w_k1_AXP_s112
                                 + 0.98903 w_k1_DIS_s112
                                 + 0.941759 w_k1_FDX_s112
                                 + 0.842425 w_k1_IBM_s112
                                 + 0.895062 w_k1_INTC_s112
                                 + 0.947704 w_k1_KO_s112
                                 + 0.91768 w_k1_MAT_s112
                                 + 0.987359 w_k1_MSFT_s112
                                 + 0.932491 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.1475 w_k1_ADM_s112
                                 + 1.14114 w_k1_AXP_s112
                                 + 1.03676 w_k1_DIS_s112
                                 + 0.944772 w_k1_FDX_s112
                                 + 1.15462 w_k1_IBM_s112
                                 + 0.905097 w_k1_INTC_s112
                                 + 1.14618 w_k1_KO_s112 + 1.21239 w_k1_MAT_s112
                                 + 1.02762 w_k1_MSFT_s112
                                 + 0.898571 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10834 w_k1_ADM_s112
                                 + 0.690337 w_k1_AXP_s112
                                 + 1.06413 w_k1_DIS_s112
                                 + 0.932972 w_k1_FDX_s112
                                 + 0.774555 w_k1_IBM_s112
                                 + 0.943592 w_k1_INTC_s112 + 1.034 w_k1_KO_s112
                                 + 0.489912 w_k1_MAT_s112
                                 + 1.07872 w_k1_MSFT_s112
                                 + 0.980199 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.24669 w_k1_ADM_s112
                                 + 1.3047 w_k1_AXP_s112 + 1.60995 w_k1_DIS_s112
                                 + 1.29071 w_k1_FDX_s112
                                 + 1.30013 w_k1_IBM_s112
                                 + 1.25131 w_k1_INTC_s112
                                 + 1.24088 w_k1_KO_s112 + 1.10334 w_k1_MAT_s112
                                 + 1.03649 w_k1_MSFT_s112
                                 + 1.08597 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.902075 w_k1_ADM_s112
                                 + 0.681326 w_k1_AXP_s112
                                 + 0.806925 w_k1_DIS_s112
                                 + 0.703205 w_k1_FDX_s112
                                 + 1.11163 w_k1_IBM_s112
                                 + 0.844307 w_k1_INTC_s112
                                 + 1.00894 w_k1_KO_s112
                                 + 0.920772 w_k1_MAT_s112
                                 + 0.809449 w_k1_MSFT_s112
                                 + 0.887359 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.04933 w_k1_ADM_s113
                                 + 1.15309 w_k1_AXP_s113
                                 + 1.35276 w_k1_DIS_s113
                                 + 1.18911 w_k1_FDX_s113
                                 + 0.952029 w_k1_IBM_s113
                                 + 1.04908 w_k1_INTC_s113 + 1.0547 w_k1_KO_s113
                                 + 0.90078 w_k1_MAT_s113
                                 + 1.11345 w_k1_MSFT_s113 + 1.0772 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.85594 w_k1_ADM_s113
                                 + 0.80368 w_k1_AXP_s113
                                 + 0.750048 w_k1_DIS_s113
                                 + 0.935549 w_k1_FDX_s113
                                 + 0.586182 w_k1_IBM_s113
                                 + 0.886469 w_k1_INTC_s113
                                 + 0.698756 w_k1_KO_s113
                                 + 0.605196 w_k1_MAT_s113
                                 + 0.835679 w_k1_MSFT_s113 + 0.862 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.04904 w_k1_ADM_s113
                                 + 1.06206 w_k1_AXP_s113
                                 + 1.43169 w_k1_DIS_s113
                                 + 1.14069 w_k1_FDX_s113 + 1.3411 w_k1_IBM_s113
                                 + 0.879786 w_k1_INTC_s113
                                 + 1.18984 w_k1_KO_s113 + 1.29366 w_k1_MAT_s113
                                 + 0.979212 w_k1_MSFT_s113
                                 + 1.12694 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.853419 w_k1_ADM_s113
                                 + 0.949032 w_k1_AXP_s113
                                 + 0.987304 w_k1_DIS_s113
                                 + 0.927337 w_k1_FDX_s113
                                 + 0.970629 w_k1_IBM_s113
                                 + 0.954432 w_k1_INTC_s113
                                 + 0.939798 w_k1_KO_s113
                                 + 1.05345 w_k1_MAT_s113
                                 + 0.961917 w_k1_MSFT_s113
                                 + 0.912852 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.802696 w_k1_ADM_s113
                                 + 1.11507 w_k1_AXP_s113
                                 + 0.998842 w_k1_DIS_s113
                                 + 0.882928 w_k1_FDX_s113
                                 + 1.42331 w_k1_IBM_s113
                                 + 0.734301 w_k1_INTC_s113
                                 + 1.11478 w_k1_KO_s113 + 1.82869 w_k1_MAT_s113
                                 + 0.811682 w_k1_MSFT_s113
                                 + 1.03661 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.592853 w_k1_ADM_s113
                                 + 0.639788 w_k1_AXP_s113
                                 + 0.579298 w_k1_DIS_s113
                                 + 0.765495 w_k1_FDX_s113 + 0.751 w_k1_IBM_s113
                                 + 0.759594 w_k1_INTC_s113
                                 + 0.770311 w_k1_KO_s113
                                 + 1.05729 w_k1_MAT_s113
                                 + 0.790096 w_k1_MSFT_s113
                                 + 0.84511 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.03165 w_k1_ADM_s113
                                 + 1.32108 w_k1_AXP_s113
                                 + 0.884416 w_k1_DIS_s113
                                 + 1.09985 w_k1_FDX_s113
                                 + 0.967221 w_k1_IBM_s113
                                 + 0.998268 w_k1_INTC_s113
                                 + 1.01113 w_k1_KO_s113 + 1.31522 w_k1_MAT_s113
                                 + 0.961866 w_k1_MSFT_s113
                                 + 0.994247 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.01508 w_k1_ADM_s113
                                 + 0.766155 w_k1_AXP_s113
                                 + 0.873951 w_k1_DIS_s113
                                 + 0.871711 w_k1_FDX_s113
                                 + 0.88457 w_k1_IBM_s113
                                 + 0.810495 w_k1_INTC_s113
                                 + 0.834645 w_k1_KO_s113
                                 + 0.837049 w_k1_MAT_s113
                                 + 0.799667 w_k1_MSFT_s113
                                 + 0.912865 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.8933 w_k1_ADM_s114
                                 + 0.690979 w_k1_AXP_s114
                                 + 0.844389 w_k1_DIS_s114
                                 + 0.866244 w_k1_FDX_s114
                                 + 0.768513 w_k1_IBM_s114
                                 + 0.809313 w_k1_INTC_s114
                                 + 0.79474 w_k1_KO_s114
                                 + 0.746136 w_k1_MAT_s114
                                 + 0.891711 w_k1_MSFT_s114
                                 + 0.866643 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.972802 w_k1_ADM_s114
                                 + 0.676708 w_k1_AXP_s114
                                 + 0.613221 w_k1_DIS_s114
                                 + 0.685864 w_k1_FDX_s114
                                 + 0.77685 w_k1_IBM_s114
                                 + 0.773507 w_k1_INTC_s114
                                 + 0.720924 w_k1_KO_s114
                                 + 0.662585 w_k1_MAT_s114
                                 + 0.710251 w_k1_MSFT_s114
                                 + 0.790146 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.58226 w_k1_ADM_s114
                                 + 1.51091 w_k1_AXP_s114
                                 + 1.66903 w_k1_DIS_s114 + 1.5099 w_k1_FDX_s114
                                 + 0.980715 w_k1_IBM_s114
                                 + 1.47985 w_k1_INTC_s114
                                 + 1.21243 w_k1_KO_s114
                                 + 0.778104 w_k1_MAT_s114
                                 + 1.20773 w_k1_MSFT_s114
                                 + 1.15657 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.21824 w_k1_ADM_s114
                                 + 1.17694 w_k1_AXP_s114
                                 + 0.939067 w_k1_DIS_s114
                                 + 0.970852 w_k1_FDX_s114
                                 + 0.961898 w_k1_IBM_s114
                                 + 1.13184 w_k1_INTC_s114
                                 + 0.960538 w_k1_KO_s114 + 1.013 w_k1_MAT_s114
                                 + 0.902628 w_k1_MSFT_s114
                                 + 0.956105 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.10205 w_k1_ADM_s114
                                 + 0.889432 w_k1_AXP_s114
                                 + 0.95122 w_k1_DIS_s114
                                 + 0.908298 w_k1_FDX_s114
                                 + 1.24465 w_k1_IBM_s114
                                 + 0.988911 w_k1_INTC_s114
                                 + 1.09214 w_k1_KO_s114 + 1.11181 w_k1_MAT_s114
                                 + 0.958978 w_k1_MSFT_s114
                                 + 1.03507 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.05997 w_k1_ADM_s114
                                 + 1.23404 w_k1_AXP_s114
                                 + 1.45268 w_k1_DIS_s114
                                 + 1.07021 w_k1_FDX_s114 + 1.3755 w_k1_IBM_s114
                                 + 1.27746 w_k1_INTC_s114
                                 + 1.41612 w_k1_KO_s114 + 1.34236 w_k1_MAT_s114
                                 + 1.15669 w_k1_MSFT_s114 + 1.0954 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.2893 w_k1_ADM_s114
                                 + 1.34141 w_k1_AXP_s114
                                 + 1.77097 w_k1_DIS_s114
                                 + 1.30932 w_k1_FDX_s114
                                 + 1.27719 w_k1_IBM_s114
                                 + 1.20415 w_k1_INTC_s114
                                 + 1.39053 w_k1_KO_s114 + 1.28816 w_k1_MAT_s114
                                 + 1.27705 w_k1_MSFT_s114
                                 + 1.27326 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.05235 w_k1_ADM_s114
                                 + 0.915793 w_k1_AXP_s114
                                 + 1.13912 w_k1_DIS_s114
                                 + 1.03492 w_k1_FDX_s114
                                 + 0.813025 w_k1_IBM_s114
                                 + 0.924935 w_k1_INTC_s114
                                 + 1.00921 w_k1_KO_s114
                                 + 0.545652 w_k1_MAT_s114
                                 + 1.05262 w_k1_MSFT_s114
                                 + 1.08122 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.33744 w_k1_ADM_s115
                                 + 1.41632 w_k1_AXP_s115
                                 + 1.02978 w_k1_DIS_s115
                                 + 1.03101 w_k1_FDX_s115
                                 + 1.25369 w_k1_IBM_s115
                                 + 1.07396 w_k1_INTC_s115
                                 + 1.12699 w_k1_KO_s115 + 1.33916 w_k1_MAT_s115
                                 + 1.10399 w_k1_MSFT_s115
                                 + 1.13101 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.13144 w_k1_ADM_s115
                                 + 1.02397 w_k1_AXP_s115
                                 + 0.953841 w_k1_DIS_s115
                                 + 0.914014 w_k1_FDX_s115
                                 + 0.959361 w_k1_IBM_s115
                                 + 0.863662 w_k1_INTC_s115
                                 + 0.900865 w_k1_KO_s115
                                 + 0.967446 w_k1_MAT_s115
                                 + 0.878901 w_k1_MSFT_s115
                                 + 1.01521 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.39662 w_k1_ADM_s115
                                 + 1.06295 w_k1_AXP_s115 + 1.1577 w_k1_DIS_s115
                                 + 1.15206 w_k1_FDX_s115
                                 + 0.941791 w_k1_IBM_s115
                                 + 1.10356 w_k1_INTC_s115
                                 + 1.16528 w_k1_KO_s115 + 1.0717 w_k1_MAT_s115
                                 + 1.21258 w_k1_MSFT_s115
                                 + 1.13919 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.963849 w_k1_ADM_s115
                                 + 0.542068 w_k1_AXP_s115
                                 + 0.81297 w_k1_DIS_s115
                                 + 0.775051 w_k1_FDX_s115
                                 + 0.662904 w_k1_IBM_s115
                                 + 0.763449 w_k1_INTC_s115
                                 + 0.752485 w_k1_KO_s115
                                 + 0.589934 w_k1_MAT_s115
                                 + 0.883208 w_k1_MSFT_s115
                                 + 0.802097 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.859443 w_k1_ADM_s115
                                 + 0.74328 w_k1_AXP_s115
                                 + 0.602725 w_k1_DIS_s115
                                 + 0.703896 w_k1_FDX_s115
                                 + 0.875849 w_k1_IBM_s115
                                 + 0.884839 w_k1_INTC_s115
                                 + 0.72728 w_k1_KO_s115
                                 + 0.824422 w_k1_MAT_s115
                                 + 0.750518 w_k1_MSFT_s115
                                 + 0.804986 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.934154 w_k1_ADM_s115
                                 + 0.716587 w_k1_AXP_s115
                                 + 1.05237 w_k1_DIS_s115
                                 + 0.85493 w_k1_FDX_s115
                                 + 0.975763 w_k1_IBM_s115
                                 + 0.794675 w_k1_INTC_s115
                                 + 0.955467 w_k1_KO_s115
                                 + 0.943989 w_k1_MAT_s115
                                 + 0.978091 w_k1_MSFT_s115
                                 + 0.978442 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.971053 w_k1_ADM_s115
                                 + 0.825327 w_k1_AXP_s115
                                 + 0.784363 w_k1_DIS_s115
                                 + 0.902275 w_k1_FDX_s115
                                 + 0.807716 w_k1_IBM_s115
                                 + 0.881557 w_k1_INTC_s115
                                 + 0.79147 w_k1_KO_s115
                                 + 0.776754 w_k1_MAT_s115
                                 + 0.938328 w_k1_MSFT_s115
                                 + 0.856781 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.33149 w_k1_ADM_s115
                                 + 1.07584 w_k1_AXP_s115
                                 + 1.24691 w_k1_DIS_s115
                                 + 1.02167 w_k1_FDX_s115
                                 + 1.55286 w_k1_IBM_s115
                                 + 0.709857 w_k1_INTC_s115
                                 + 1.13971 w_k1_KO_s115 + 1.74263 w_k1_MAT_s115
                                 + 0.960492 w_k1_MSFT_s115
                                 + 1.04015 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.05751 w_k1_ADM_s116
                                 + 0.899782 w_k1_AXP_s116
                                 + 0.799927 w_k1_DIS_s116
                                 + 0.895332 w_k1_FDX_s116
                                 + 0.8672 w_k1_IBM_s116
                                 + 0.852814 w_k1_INTC_s116
                                 + 0.730863 w_k1_KO_s116
                                 + 0.58165 w_k1_MAT_s116
                                 + 0.944095 w_k1_MSFT_s116
                                 + 0.884663 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.957212 w_k1_ADM_s116
                                 + 0.850439 w_k1_AXP_s116
                                 + 1.09111 w_k1_DIS_s116
                                 + 0.955675 w_k1_FDX_s116
                                 + 1.1459 w_k1_IBM_s116
                                 + 0.747863 w_k1_INTC_s116
                                 + 1.17961 w_k1_KO_s116 + 1.10221 w_k1_MAT_s116
                                 + 1.03397 w_k1_MSFT_s116 + 1.0069 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.852104 w_k1_ADM_s116
                                 + 0.568522 w_k1_AXP_s116
                                 + 0.634239 w_k1_DIS_s116
                                 + 0.816441 w_k1_FDX_s116
                                 + 0.712073 w_k1_IBM_s116
                                 + 0.672159 w_k1_INTC_s116
                                 + 0.671669 w_k1_KO_s116
                                 + 0.669147 w_k1_MAT_s116
                                 + 0.855858 w_k1_MSFT_s116
                                 + 0.810357 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.12283 w_k1_ADM_s116
                                 + 1.08631 w_k1_AXP_s116
                                 + 1.19869 w_k1_DIS_s116
                                 + 1.21072 w_k1_FDX_s116
                                 + 0.976822 w_k1_IBM_s116
                                 + 0.951253 w_k1_INTC_s116
                                 + 1.12381 w_k1_KO_s116 + 1.16162 w_k1_MAT_s116
                                 + 1.25493 w_k1_MSFT_s116
                                 + 1.12068 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.21049 w_k1_ADM_s116
                                 + 1.09023 w_k1_AXP_s116
                                 + 0.866925 w_k1_DIS_s116
                                 + 1.01212 w_k1_FDX_s116
                                 + 1.03391 w_k1_IBM_s116
                                 + 0.863743 w_k1_INTC_s116
                                 + 0.92591 w_k1_KO_s116 + 1.17063 w_k1_MAT_s116
                                 + 0.962702 w_k1_MSFT_s116
                                 + 0.939311 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.30407 w_k1_ADM_s116
                                 + 1.41391 w_k1_AXP_s116
                                 + 1.60593 w_k1_DIS_s116
                                 + 1.31685 w_k1_FDX_s116
                                 + 1.21403 w_k1_IBM_s116
                                 + 1.09615 w_k1_INTC_s116
                                 + 1.19481 w_k1_KO_s116 + 1.32576 w_k1_MAT_s116
                                 + 1.2689 w_k1_MSFT_s116 + 1.07287 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.13523 w_k1_ADM_s116
                                 + 0.923932 w_k1_AXP_s116
                                 + 0.994893 w_k1_DIS_s116
                                 + 1.10076 w_k1_FDX_s116
                                 + 0.626414 w_k1_IBM_s116
                                 + 1.00137 w_k1_INTC_s116
                                 + 0.768575 w_k1_KO_s116
                                 + 0.590867 w_k1_MAT_s116
                                 + 1.09054 w_k1_MSFT_s116
                                 + 0.854204 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.05163 w_k1_ADM_s116
                                 + 1.2532 w_k1_AXP_s116 + 1.01447 w_k1_DIS_s116
                                 + 1.0147 w_k1_FDX_s116 + 1.17203 w_k1_IBM_s116
                                 + 0.781331 w_k1_INTC_s116
                                 + 1.15181 w_k1_KO_s116 + 1.51617 w_k1_MAT_s116
                                 + 0.982513 w_k1_MSFT_s116
                                 + 1.07727 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.29718 w_k1_ADM_s117
                                 + 1.36042 w_k1_AXP_s117
                                 + 1.44666 w_k1_DIS_s117
                                 + 1.37178 w_k1_FDX_s117
                                 + 0.942855 w_k1_IBM_s117
                                 + 1.4639 w_k1_INTC_s117 + 1.26329 w_k1_KO_s117
                                 + 0.890579 w_k1_MAT_s117
                                 + 1.23702 w_k1_MSFT_s117
                                 + 1.00276 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.16271 w_k1_ADM_s117
                                 + 1.04584 w_k1_AXP_s117
                                 + 0.721145 w_k1_DIS_s117
                                 + 0.879982 w_k1_FDX_s117
                                 + 0.921105 w_k1_IBM_s117
                                 + 0.916184 w_k1_INTC_s117
                                 + 0.850247 w_k1_KO_s117
                                 + 0.848712 w_k1_MAT_s117
                                 + 0.783864 w_k1_MSFT_s117
                                 + 0.793181 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.29731 w_k1_ADM_s117
                                 + 0.961782 w_k1_AXP_s117
                                 + 1.00605 w_k1_DIS_s117
                                 + 0.983414 w_k1_FDX_s117
                                 + 1.04962 w_k1_IBM_s117
                                 + 1.01134 w_k1_INTC_s117 + 1.2754 w_k1_KO_s117
                                 + 0.999354 w_k1_MAT_s117
                                 + 1.02423 w_k1_MSFT_s117
                                 + 0.95083 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.07335 w_k1_ADM_s117
                                 + 0.785521 w_k1_AXP_s117
                                 + 0.578023 w_k1_DIS_s117
                                 + 0.829785 w_k1_FDX_s117
                                 + 0.664598 w_k1_IBM_s117
                                 + 0.82539 w_k1_INTC_s117
                                 + 0.77065 w_k1_KO_s117
                                 + 0.714211 w_k1_MAT_s117
                                 + 0.785882 w_k1_MSFT_s117
                                 + 0.876196 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.15015 w_k1_ADM_s117
                                 + 1.04293 w_k1_AXP_s117
                                 + 1.14848 w_k1_DIS_s117
                                 + 0.974858 w_k1_FDX_s117
                                 + 1.51113 w_k1_IBM_s117
                                 + 1.17189 w_k1_INTC_s117
                                 + 1.43534 w_k1_KO_s117 + 1.59684 w_k1_MAT_s117
                                 + 1.03588 w_k1_MSFT_s117
                                 + 1.11675 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.13938 w_k1_ADM_s117
                                 + 1.02301 w_k1_AXP_s117
                                 + 1.01031 w_k1_DIS_s117
                                 + 1.08612 w_k1_FDX_s117
                                 + 0.682304 w_k1_IBM_s117
                                 + 1.27009 w_k1_INTC_s117
                                 + 1.00766 w_k1_KO_s117
                                 + 0.903683 w_k1_MAT_s117
                                 + 1.23441 w_k1_MSFT_s117
                                 + 1.00916 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.929558 w_k1_ADM_s117
                                 + 0.658934 w_k1_AXP_s117
                                 + 0.782847 w_k1_DIS_s117
                                 + 0.906308 w_k1_FDX_s117
                                 + 0.733812 w_k1_IBM_s117
                                 + 1.19605 w_k1_INTC_s117
                                 + 0.894244 w_k1_KO_s117
                                 + 0.511194 w_k1_MAT_s117
                                 + 0.961492 w_k1_MSFT_s117
                                 + 0.811978 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.14266 w_k1_ADM_s117
                                 + 1.85723 w_k1_AXP_s117
                                 + 1.46451 w_k1_DIS_s117 + 1.4346 w_k1_FDX_s117
                                 + 1.14051 w_k1_IBM_s117
                                 + 1.39385 w_k1_INTC_s117
                                 + 1.35676 w_k1_KO_s117 + 2.22662 w_k1_MAT_s117
                                 + 1.41359 w_k1_MSFT_s117
                                 + 1.25724 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0539649139260744 w_k1_ADM_s000
                                 + 0.0625020766313483 w_k1_ADM_s001
                                 + 0.0428058643749196 w_k1_ADM_s002
                                 + 0.0338197439489952 w_k1_ADM_s003
                                 + 0.0346933396550373 w_k1_ADM_s004
                                 + 0.014036533344982 w_k1_ADM_s005
                                 + 0.0174119544480697 w_k1_ADM_s006
                                 + 0.036522988621749 w_k1_ADM_s007
                                 + 0.0364269580710136 w_k1_ADM_s010
                                 + 0.0352269933355617 w_k1_ADM_s011
                                 + 0.0259894069756459 w_k1_ADM_s012
                                 + 0.0340452119023938 w_k1_ADM_s013
                                 + 0.0629446035817736 w_k1_ADM_s014
                                 + 0.0206894955369371 w_k1_ADM_s015
                                 + 0.0248432445261123 w_k1_ADM_s016
                                 + 0.0152651114505809 w_k1_ADM_s017
                                 + 0.0341565911872209 w_k1_ADM_s100
                                 + 0.0536658271240406 w_k1_ADM_s101
                                 + 0.0147274627429666 w_k1_ADM_s102
                                 + 0.0164407761820461 w_k1_ADM_s103
                                 + 0.0147213131192759 w_k1_ADM_s104
                                 + 0.0515512238379678 w_k1_ADM_s105
                                 + 0.0142137626095278 w_k1_ADM_s106
                                 + 0.0355229052048657 w_k1_ADM_s107
                                 + 0.0226198538431162 w_k1_ADM_s110
                                 + 0.0219076119801906 w_k1_ADM_s111
                                 + 0.0295843816357365 w_k1_ADM_s112
                                 + 0.00939611245436301 w_k1_ADM_s113
                                 + 0.0396432054183562 w_k1_ADM_s114
                                 + 0.0365913292377213 w_k1_ADM_s115
                                 + 0.0496520723919822 w_k1_ADM_s116
                                 + 0.0155648258909965 w_k1_ADM_s117
                                 + 0.0531613622557294 w_k1_AXP_s000
                                 + 0.0743321153915628 w_k1_AXP_s001
                                 + 0.0423414444703434 w_k1_AXP_s002
                                 + 0.0289832762820449 w_k1_AXP_s003
                                 + 0.0388034604769038 w_k1_AXP_s004
                                 + 0.0152975850843225 w_k1_AXP_s005
                                 + 0.0174950018555578 w_k1_AXP_s006
                                 + 0.0350998082879068 w_k1_AXP_s007
                                 + 0.0372275823901522 w_k1_AXP_s010
                                 + 0.0325050889210141 w_k1_AXP_s011
                                 + 0.0285486543778982 w_k1_AXP_s012
                                 + 0.0332279837848829 w_k1_AXP_s013
                                 + 0.0726760792457267 w_k1_AXP_s014
                                 + 0.0205054008716323 w_k1_AXP_s015
                                 + 0.0265383942378453 w_k1_AXP_s016
                                 + 0.0143924747806178 w_k1_AXP_s017
                                 + 0.0372993721657927 w_k1_AXP_s100
                                 + 0.0591383766246121 w_k1_AXP_s101
                                 + 0.014919946879417 w_k1_AXP_s102
                                 + 0.0152350854765701 w_k1_AXP_s103
                                 + 0.0131038883372379 w_k1_AXP_s104
                                 + 0.052212929100958 w_k1_AXP_s105
                                 + 0.0127672841408124 w_k1_AXP_s106
                                 + 0.0375576211228811 w_k1_AXP_s107
                                 + 0.0260951421653133 w_k1_AXP_s110
                                 + 0.0177645617221614 w_k1_AXP_s111
                                 + 0.0297859839590461 w_k1_AXP_s112
                                 + 0.0101645740299581 w_k1_AXP_s113
                                 + 0.0353309720002203 w_k1_AXP_s114
                                 + 0.0311873601161905 w_k1_AXP_s115
                                 + 0.0462399966615385 w_k1_AXP_s116
                                 + 0.0142141531671151 w_k1_AXP_s117
                                 + 0.0588588408392987 w_k1_DIS_s000
                                 + 0.0713334951265122 w_k1_DIS_s001
                                 + 0.0447546363025306 w_k1_DIS_s002
                                 + 0.0297741136339612 w_k1_DIS_s003
                                 + 0.0347486567798596 w_k1_DIS_s004
                                 + 0.0160848480352561 w_k1_DIS_s005
                                 + 0.020951975671823 w_k1_DIS_s006
                                 + 0.0368269914823257 w_k1_DIS_s007
                                 + 0.0383823274566751 w_k1_DIS_s010
                                 + 0.0332267296338662 w_k1_DIS_s011
                                 + 0.02712150819498 w_k1_DIS_s012
                                 + 0.0338543436026639 w_k1_DIS_s013
                                 + 0.0737761984777596 w_k1_DIS_s014
                                 + 0.0199740648876512 w_k1_DIS_s015
                                 + 0.029880958761962 w_k1_DIS_s016
                                 + 0.0149244717914707 w_k1_DIS_s017
                                 + 0.0371159129627076 w_k1_DIS_s100
                                 + 0.0609993104498906 w_k1_DIS_s101
                                 + 0.0165921685599286 w_k1_DIS_s102
                                 + 0.0169383311718164 w_k1_DIS_s103
                                 + 0.0131458258299115 w_k1_DIS_s104
                                 + 0.0509411256643488 w_k1_DIS_s105
                                 + 0.0136493894074323 w_k1_DIS_s106
                                 + 0.0371201848367028 w_k1_DIS_s107
                                 + 0.0260434402474828 w_k1_DIS_s110
                                 + 0.0196921561899013 w_k1_DIS_s111
                                 + 0.0314056605254205 w_k1_DIS_s112
                                 + 0.0103817645555164 w_k1_DIS_s113
                                 + 0.0393882865304384 w_k1_DIS_s114
                                 + 0.0312242138414143 w_k1_DIS_s115
                                 + 0.049662046493428 w_k1_DIS_s116
                                 + 0.0133941678297414 w_k1_DIS_s117
                                 + 0.055507293729519 w_k1_FDX_s000
                                 + 0.0673064440215519 w_k1_FDX_s001
                                 + 0.0431267119988486 w_k1_FDX_s002
                                 + 0.0300226959677843 w_k1_FDX_s003
                                 + 0.0343667392773686 w_k1_FDX_s004
                                 + 0.014956737447973 w_k1_FDX_s005
                                 + 0.0182922944854515 w_k1_FDX_s006
                                 + 0.0358860790128676 w_k1_FDX_s007
                                 + 0.036636687595341 w_k1_FDX_s010
                                 + 0.0337590081529686 w_k1_FDX_s011
                                 + 0.0278103783112527 w_k1_FDX_s012
                                 + 0.0338417831864017 w_k1_FDX_s013
                                 + 0.0693043248464612 w_k1_FDX_s014
                                 + 0.0203315330272174 w_k1_FDX_s015
                                 + 0.0263129400521833 w_k1_FDX_s016
                                 + 0.0142436877476983 w_k1_FDX_s017
                                 + 0.0347391805862432 w_k1_FDX_s100
                                 + 0.0565661602785212 w_k1_FDX_s101
                                 + 0.0146327661237628 w_k1_FDX_s102
                                 + 0.0161946670422881 w_k1_FDX_s103
                                 + 0.0130616692897909 w_k1_FDX_s104
                                 + 0.0520155760313294 w_k1_FDX_s105
                                 + 0.012737501711212 w_k1_FDX_s106
                                 + 0.0372840473258038 w_k1_FDX_s107
                                 + 0.0247863135397601 w_k1_FDX_s110
                                 + 0.0196042232057826 w_k1_FDX_s111
                                 + 0.02833354903658 w_k1_FDX_s112
                                 + 0.0100303740471685 w_k1_FDX_s113
                                 + 0.0359875874979508 w_k1_FDX_s114
                                 + 0.0296584403401242 w_k1_FDX_s115
                                 + 0.047981692842782 w_k1_FDX_s116
                                 + 0.0137417738039701 w_k1_FDX_s117
                                 + 0.0569414246310919 w_k1_IBM_s000
                                 + 0.0699417111427394 w_k1_IBM_s001
                                 + 0.0468290670806096 w_k1_IBM_s002
                                 + 0.0331823444124218 w_k1_IBM_s003
                                 + 0.035527364964342 w_k1_IBM_s004
                                 + 0.0150731240025767 w_k1_IBM_s005
                                 + 0.0194450363544381 w_k1_IBM_s006
                                 + 0.0366495446833163 w_k1_IBM_s007
                                 + 0.0367503646471954 w_k1_IBM_s010
                                 + 0.037267842128725 w_k1_IBM_s011
                                 + 0.0268094222646305 w_k1_IBM_s012
                                 + 0.033774541466017 w_k1_IBM_s013
                                 + 0.0683791257186098 w_k1_IBM_s014
                                 + 0.0202285883595523 w_k1_IBM_s015
                                 + 0.0262667542736024 w_k1_IBM_s016
                                 + 0.0141829582868046 w_k1_IBM_s017
                                 + 0.0375886604819603 w_k1_IBM_s100
                                 + 0.0554182649510088 w_k1_IBM_s101
                                 + 0.0156157247519154 w_k1_IBM_s102
                                 + 0.016967544683297 w_k1_IBM_s103
                                 + 0.0134092611321046 w_k1_IBM_s104
                                 + 0.0512878500273909 w_k1_IBM_s105
                                 + 0.0130444955564211 w_k1_IBM_s106
                                 + 0.037745504873118 w_k1_IBM_s107
                                 + 0.0251589321498938 w_k1_IBM_s110
                                 + 0.019565011685604 w_k1_IBM_s111
                                 + 0.0293906034040839 w_k1_IBM_s112
                                 + 0.0105579390665123 w_k1_IBM_s113
                                 + 0.0360635165572705 w_k1_IBM_s114
                                 + 0.0330173491496293 w_k1_IBM_s115
                                 + 0.0460020041386033 w_k1_IBM_s116
                                 + 0.0132259532847886 w_k1_IBM_s117
                                 + 0.0610154236153772 w_k1_INTC_s000
                                 + 0.0747782349897273 w_k1_INTC_s001
                                 + 0.0471052058408069 w_k1_INTC_s002
                                 + 0.029658539704225 w_k1_INTC_s003
                                 + 0.0370988685868047 w_k1_INTC_s004
                                 + 0.0148165307231777 w_k1_INTC_s005
                                 + 0.0192860624215008 w_k1_INTC_s006
                                 + 0.0393799894308483 w_k1_INTC_s007
                                 + 0.0389414850955379 w_k1_INTC_s010
                                 + 0.0340997268503715 w_k1_INTC_s011
                                 + 0.0280964440753567 w_k1_INTC_s012
                                 + 0.0303442796693517 w_k1_INTC_s013
                                 + 0.0632421242755276 w_k1_INTC_s014
                                 + 0.018682456531525 w_k1_INTC_s015
                                 + 0.0279586720059449 w_k1_INTC_s016
                                 + 0.0150607240820581 w_k1_INTC_s017
                                 + 0.0383493344123136 w_k1_INTC_s100
                                 + 0.0534546546193635 w_k1_INTC_s101
                                 + 0.0156479932022943 w_k1_INTC_s102
                                 + 0.0163323662949095 w_k1_INTC_s103
                                 + 0.0116424075468648 w_k1_INTC_s104
                                 + 0.0514569397532879 w_k1_INTC_s105
                                 + 0.0114233318722557 w_k1_INTC_s106
                                 + 0.0394062415567038 w_k1_INTC_s107
                                 + 0.0242981807597189 w_k1_INTC_s110
                                 + 0.0185558932363965 w_k1_INTC_s111
                                 + 0.0277290336622068 w_k1_INTC_s112
                                 + 0.00901050145369961 w_k1_INTC_s113
                                 + 0.0366094113107052 w_k1_INTC_s114
                                 + 0.0279815932259068 w_k1_INTC_s115
                                 + 0.0401445966989728 w_k1_INTC_s116
                                 + 0.0149397488602871 w_k1_INTC_s117
                                 + 0.0542141616504859 w_k1_KO_s000
                                 + 0.0719589626422877 w_k1_KO_s001
                                 + 0.0424209551566238 w_k1_KO_s002
                                 + 0.0327366779327901 w_k1_KO_s003
                                 + 0.0344036901635879 w_k1_KO_s004
                                 + 0.0152599313155305 w_k1_KO_s005
                                 + 0.0205677570669396 w_k1_KO_s006
                                 + 0.0321404412120377 w_k1_KO_s007
                                 + 0.0373994194430052 w_k1_KO_s010
                                 + 0.0358169982004832 w_k1_KO_s011
                                 + 0.0266272219539247 w_k1_KO_s012
                                 + 0.0325509903974292 w_k1_KO_s013
                                 + 0.0717540511796191 w_k1_KO_s014
                                 + 0.020700758800903 w_k1_KO_s015
                                 + 0.0272366639496763 w_k1_KO_s016
                                 + 0.0142232773866656 w_k1_KO_s017
                                 + 0.0376857435437503 w_k1_KO_s100
                                 + 0.0554258910357423 w_k1_KO_s101
                                 + 0.0155475517464596 w_k1_KO_s102
                                 + 0.0171502472026272 w_k1_KO_s103
                                 + 0.0138433770578291 w_k1_KO_s104
                                 + 0.0521535886441765 w_k1_KO_s105
                                 + 0.0126152000212769 w_k1_KO_s106
                                 + 0.0370774878834353 w_k1_KO_s107
                                 + 0.0266386776392041 w_k1_KO_s110
                                 + 0.0198109038381853 w_k1_KO_s111
                                 + 0.0295471085419656 w_k1_KO_s112
                                 + 0.00998111774283928 w_k1_KO_s113
                                 + 0.0370885827389176 w_k1_KO_s114
                                 + 0.0309685555984468 w_k1_KO_s115
                                 + 0.0457512498338291 w_k1_KO_s116
                                 + 0.0148883738289458 w_k1_KO_s117
                                 + 0.050047044352503 w_k1_MAT_s000
                                 + 0.06852026154593 w_k1_MAT_s001
                                 + 0.0320816287707528 w_k1_MAT_s002
                                 + 0.027672814045606 w_k1_MAT_s003
                                 + 0.0309217172331216 w_k1_MAT_s004
                                 + 0.0141781726671934 w_k1_MAT_s005
                                 + 0.0200822562946574 w_k1_MAT_s006
                                 + 0.0313070637020936 w_k1_MAT_s007
                                 + 0.0352733252181744 w_k1_MAT_s010
                                 + 0.0308835970811023 w_k1_MAT_s011
                                 + 0.0302682117079267 w_k1_MAT_s012
                                 + 0.0397155924780883 w_k1_MAT_s013
                                 + 0.0756792279223015 w_k1_MAT_s014
                                 + 0.0249298287422099 w_k1_MAT_s015
                                 + 0.0225117304979153 w_k1_MAT_s016
                                 + 0.0125967498331517 w_k1_MAT_s017
                                 + 0.0348519161261 w_k1_MAT_s100
                                 + 0.0579115867299567 w_k1_MAT_s101
                                 + 0.0132557204712005 w_k1_MAT_s102
                                 + 0.0212492341680221 w_k1_MAT_s103
                                 + 0.0130072609828418 w_k1_MAT_s104
                                 + 0.043787562501261 w_k1_MAT_s105
                                 + 0.0143961390094673 w_k1_MAT_s106
                                 + 0.0347432224615962 w_k1_MAT_s107
                                 + 0.0261325234723303 w_k1_MAT_s110
                                 + 0.0165861148728595 w_k1_MAT_s111
                                 + 0.0264588347787112 w_k1_MAT_s112
                                 + 0.011853353577977 w_k1_MAT_s113
                                 + 0.033008306710383 w_k1_MAT_s114
                                 + 0.0349009737952998 w_k1_MAT_s115
                                 + 0.0462876872840931 w_k1_MAT_s116
                                 + 0.0140545470653365 w_k1_MAT_s117
                                 + 0.0533376711862342 w_k1_MSFT_s000
                                 + 0.0696828121627299 w_k1_MSFT_s001
                                 + 0.0421735213054121 w_k1_MSFT_s002
                                 + 0.0290160625455181 w_k1_MSFT_s003
                                 + 0.0349144867492223 w_k1_MSFT_s004
                                 + 0.0149169059954333 w_k1_MSFT_s005
                                 + 0.0208921240532071 w_k1_MSFT_s006
                                 + 0.0357089570080567 w_k1_MSFT_s007
                                 + 0.039123549618147 w_k1_MSFT_s010
                                 + 0.034061831162236 w_k1_MSFT_s011
                                 + 0.0244058265740423 w_k1_MSFT_s012
                                 + 0.0332867006163311 w_k1_MSFT_s013
                                 + 0.0667690033942792 w_k1_MSFT_s014
                                 + 0.0223291291037009 w_k1_MSFT_s015
                                 + 0.0268471057216605 w_k1_MSFT_s016
                                 + 0.0147805311074323 w_k1_MSFT_s017
                                 + 0.0377223676876638 w_k1_MSFT_s100
                                 + 0.0582283200399953 w_k1_MSFT_s101
                                 + 0.0137485882815816 w_k1_MSFT_s102
                                 + 0.0180490767579735 w_k1_MSFT_s103
                                 + 0.0136878175354329 w_k1_MSFT_s104
                                 + 0.0527494898209084 w_k1_MSFT_s105
                                 + 0.0112777117459767 w_k1_MSFT_s106
                                 + 0.0365517926954503 w_k1_MSFT_s107
                                 + 0.0249522375641105 w_k1_MSFT_s110
                                 + 0.0199399824681803 w_k1_MSFT_s111
                                 + 0.0270087275415312 w_k1_MSFT_s112
                                 + 0.00931020984461538 w_k1_MSFT_s113
                                 + 0.035728634417255 w_k1_MSFT_s114
                                 + 0.0307687563894871 w_k1_MSFT_s115
                                 + 0.0488190469368221 w_k1_MSFT_s116
                                 + 0.0135953697134575 w_k1_MSFT_s117
                                 + 0.0552180303972865 w_k1_PG_s000
                                 + 0.0665520844804551 w_k1_PG_s001
                                 + 0.0447005753640819 w_k1_PG_s002
                                 + 0.0287872870349948 w_k1_PG_s003
                                 + 0.0369397235754422 w_k1_PG_s004
                                 + 0.0157980073851006 w_k1_PG_s005
                                 + 0.0192592790696234 w_k1_PG_s006
                                 + 0.0360419386264173 w_k1_PG_s007
                                 + 0.0336634707534658 w_k1_PG_s010
                                 + 0.0363549936919296 w_k1_PG_s011
                                 + 0.0264238162304316 w_k1_PG_s012
                                 + 0.0316520878067446 w_k1_PG_s013
                                 + 0.0691681444639655 w_k1_PG_s014
                                 + 0.0203157546186927 w_k1_PG_s015
                                 + 0.0261583833088094 w_k1_PG_s016
                                 + 0.0127292808063113 w_k1_PG_s017
                                 + 0.0374586609213472 w_k1_PG_s100
                                 + 0.058011771601957 w_k1_PG_s101
                                 + 0.0141394020727043 w_k1_PG_s102
                                 + 0.0161888136182733 w_k1_PG_s103
                                 + 0.0140977358476453 w_k1_PG_s104
                                 + 0.0562641040803081 w_k1_PG_s105
                                 + 0.0122323807692483 w_k1_PG_s106
                                 + 0.0359808263334941 w_k1_PG_s107
                                 + 0.0258709758329011 w_k1_PG_s110
                                 + 0.0200134458843399 w_k1_PG_s111
                                 + 0.0266807765730666 w_k1_PG_s112
                                 + 0.010091937193848 w_k1_PG_s113
                                 + 0.0363631555985585 w_k1_PG_s114
                                 + 0.0312621928240713 w_k1_PG_s115
                                 + 0.0447588200626162 w_k1_PG_s116
                                 + 0.0127751457198102 w_k1_PG_s117 >= 1.08
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
