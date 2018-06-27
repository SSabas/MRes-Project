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
 w_k1_ADM_s0_return_constraint:  1.10284 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.87766 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.19213 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.887551 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.949143 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.20445 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.26594 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.804898 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.35499 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.11978 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.967583 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.87025 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.07583 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.40722 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.4464 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.42676 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.16218 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.770605 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.849504 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.04995 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.07694 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.905568 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.97835 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.19717 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.03768 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.969228 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.33979 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.83757 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.641323 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.06137 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.875647 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.889431 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.772197 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.02211 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.22574 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.19223 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.19313 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.17227 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.14994 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.869808 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.29662 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.946392 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.751228 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.24815 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.19716 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.683535 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.3385 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.855568 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.869772 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.815572 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.16812 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.5525 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.30152 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.54538 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.20122 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.687188 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.621554 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.32313 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.823503 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.871863 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.18697 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.34638 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.889893 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.08054 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.5665 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.825481 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.713057 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.24109 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.730533 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.922039 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.77947 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.991502 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.15313 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.08193 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.293 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.61337 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27516 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.82184 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.38825 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.82557 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.763058 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.22035 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.26239 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.56667 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.42212 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.16913 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.871128 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.766093 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.02144 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.6139 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.28847 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.66986 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.08994 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.875953 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.571048 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.33284 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.995252 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.930107 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.05106 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.31303 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.93836 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.970819 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.62417 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.785537 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.578397 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.35652 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.7499 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.979806 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.756031 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.05609 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.20527 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.34628 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.56979 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.77959 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15251 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.899486 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.22414 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.874491 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.83544 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.13797 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.30324 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.767565 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.07472 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.00819 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05752 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.858111 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.03549 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.41154 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.35352 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.37143 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.06721 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.78748 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.742339 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.09817 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.0742 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.889026 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.06957 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.04581 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.92848 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.964961 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.48588 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.912826 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.667069 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.217 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.835066 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.996127 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.781406 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.966642 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.14874 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.03127 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.43361 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.34563 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18118 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.887446 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.15227 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.99227 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.832586 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.19902 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.809678 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.504026 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.82073 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.04271 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.525647 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.855439 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.17694 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.0897 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.670426 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.36441 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.0728 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.94333 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.671501 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 2.10761 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.669836 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.20307 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.750029 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.72316 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.878318 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.21367 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.924549 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.626449 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.855649 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.10103 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.676663 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.878233 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.31753 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.0255 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.02661 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.72169 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.839482 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.32996 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.05238 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.888424 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.11471 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.980775 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.928507 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.06945 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.18294 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.829881 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.12493 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.18039 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.11261 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.988252 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.09155 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.3989 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.1649 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.27134 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.954646 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.880084 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.827248 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.909253 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.03936 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.900226 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01264 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.97064 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.940297 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.05189 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.37491 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.886078 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.778728 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.29707 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.924451 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.16808 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.939701 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.998407 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.20429 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.11864 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.48849 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.21227 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13277 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.83175 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.34263 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.911397 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.853526 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.16745 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.899306 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.590546 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.46174 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.17851 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.726934 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.77572 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.07999 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.22712 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.98266 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.42547 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.08194 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.87178 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.74975 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.57433 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.953621 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.03243 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.904827 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.39946 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.965519 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.10098 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.17089 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.840153 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.806543 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.23735 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.768862 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.0855 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.04884 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.16445 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.15923 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.49614 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.22309 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.43793 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2982 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.839977 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.1524 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.968082 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.713722 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.28338 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.757485 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.462419 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.73693 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.883884 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.462516 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.88753 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.13864 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.28538 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.746899 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.18308 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.18639 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.848054 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.571632 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.90982 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.558482 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.957381 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.965021 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.86956 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.840475 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.5613 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.16245 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.686833 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.941659 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.96343 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.763949 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.846647 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.20862 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.17301 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.07653 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.5791 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.792945 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.61593 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12663 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.885223 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.21768 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.896004 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.891548 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.12848 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.12173 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.711272 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.08161 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.06374 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.00191 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.879592 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.93282 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.36826 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.28655 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.19138 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.03972 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.825548 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.7731 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.993684 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.966545 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.918907 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.10801 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.12017 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.973869 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.04128 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.3134 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.98541 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.755365 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.24823 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.863455 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.01694 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.812046 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.00887 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.14194 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.08693 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.27273 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.29022 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06832 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.91409 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.15263 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.999682 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.90934 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05672 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12085 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.764133 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.23572 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08004 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.873952 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.928268 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04466 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.3362 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.11785 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.1352 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.04556 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.89984 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.743076 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.10713 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.981252 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.880271 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.933687 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.16532 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.962784 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08396 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.27585 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.860962 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.807782 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.12393 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.885046 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.02274 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.928134 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.13906 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.18928 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.2072 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.23464 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.35955 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.242999328375202 z_k1_s0000
                                 - 0.136999872476727 z_k1_s0001
                                 - 0.103999916996189 z_k1_s0002
                                 - 0.223000501355606 z_k1_s0003
                                 - 0.414000129082468 z_k1_s0004
                                 - 0.163999836051713 z_k1_s0005
                                 - 0.204999150440254 z_k1_s0006
                                 - 0.228999805661811 z_k1_s0007
                                 - 0.0969999332576419 z_k1_s0010
                                 - 0.201999413703991 z_k1_s0011
                                 - 0.123999824618107 z_k1_s0012
                                 - 0.291999051572207 z_k1_s0013
                                 - 0.147000524783635 z_k1_s0014
                                 - 0.283999795583208 z_k1_s0015
                                 - 0.112999901234181 z_k1_s0016
                                 - 0.196999514711231 z_k1_s0017
                                 - 0.174999737683275 z_k1_s0020
                                 - 0.110999817913412 z_k1_s0021
                                 - 0.471999522802587 z_k1_s0022
                                 - 0.395999102826553 z_k1_s0023
                                 - 0.253999137712358 z_k1_s0024
                                 - 0.110999817913412 z_k1_s0025
                                 - 0.160999795308565 z_k1_s0026
                                 - 0.151999740753514 z_k1_s0027
                                 - 0.164999268028561 z_k1_s0030
                                 - 0.209999617490084 z_k1_s0031
                                 - 0.210999840011134 z_k1_s0032
                                 - 0.118999783244381 z_k1_s0033
                                 - 0.143999427079374 z_k1_s0034
                                 - 0.0989998024167796 z_k1_s0035
                                 - 0.212000062532184 z_k1_s0036
                                 - 0.0479999298262795 z_k1_s0037
                                 - 0.256000086008524 z_k1_s0040
                                 - 0.269999833212351 z_k1_s0041
                                 - 0.114999971686703 z_k1_s0042
                                 - 0.114999971686703 z_k1_s0043
                                 - 0.183999031819568 z_k1_s0044
                                 - 0.271000168298016 z_k1_s0045
                                 - 0.276000195727846 z_k1_s0046
                                 - 0.160999894441444 z_k1_s0047
                                 - 0.923999498276196 z_k1_s0050
                                 - 0.544999739017878 z_k1_s0051
                                 - 0.445999664631171 z_k1_s0052
                                 - 0.696000088052098 z_k1_s0053
                                 - 1.81699868332025 z_k1_s0054
                                 - 0.497000083496968 z_k1_s0055
                                 - 0.497000083496968 z_k1_s0056
                                 - 0.557999658179789 z_k1_s0057
                                 - 0.162999640398256 z_k1_s0060
                                 - 0.252999312517856 z_k1_s0061
                                 - 0.309999296499798 z_k1_s0062
                                 - 0.636997019243679 z_k1_s0063
                                 - 0.179999663638415 z_k1_s0064
                                 - 0.404998205138985 z_k1_s0065
                                 - 0.750996987207565 z_k1_s0066
                                 - 0.494998516057016 z_k1_s0067
                                 - 0.184999463292045 z_k1_s0070
                                 - 0.179999661148802 z_k1_s0071
                                 - 0.106999836260633 z_k1_s0072
                                 - 0.328999870610292 z_k1_s0073
                                 - 0.202000147377482 z_k1_s0074
                                 - 0.287999118638481 z_k1_s0075
                                 - 0.151999751547833 z_k1_s0076
                                 - 0.25300050363578 z_k1_s0077
                                 - 0.228000385435383 z_k1_s0100
                                 - 0.15899998142186 z_k1_s0101
                                 - 0.128000109647866 z_k1_s0102
                                 - 0.191000281834001 z_k1_s0103
                                 - 0.205999691401725 z_k1_s0104
                                 - 0.161999863335405 z_k1_s0105
                                 - 0.200999888212484 z_k1_s0106
                                 - 0.284999701793731 z_k1_s0107
                                 - 0.221000136415182 z_k1_s0110
                                 - 0.633999661581903 z_k1_s0111
                                 - 0.13099991556282 z_k1_s0112
                                 - 0.418001296336567 z_k1_s0113
                                 - 0.177000028442916 z_k1_s0114
                                 - 0.349000991716402 z_k1_s0115
                                 - 0.418001296336567 z_k1_s0116
                                 - 0.358001149101659 z_k1_s0117
                                 - 0.849000418388174 z_k1_s0120
                                 - 1.42100366403944 z_k1_s0121
                                 - 0.664998729449891 z_k1_s0122
                                 - 0.526001506013615 z_k1_s0123
                                 - 1.15200312803094 z_k1_s0124
                                 - 0.524001029428429 z_k1_s0125
                                 - 0.727002305664399 z_k1_s0126
                                 - 0.7230039873002 z_k1_s0127
                                 - 0.253999588996824 z_k1_s0130
                                 - 0.263000365772019 z_k1_s0131
                                 - 0.0629999434283299 z_k1_s0132
                                 - 0.253999588996824 z_k1_s0133
                                 - 0.275999701114606 z_k1_s0134
                                 - 0.0599999525031691 z_k1_s0135
                                 - 0.0710000085620602 z_k1_s0136
                                 - 0.0989999685302121 z_k1_s0137
                                 - 0.548001274429216 z_k1_s0140
                                 - 0.712000327152216 z_k1_s0141
                                 - 1.06600251072554 z_k1_s0142
                                 - 0.267000772784034 z_k1_s0143
                                 - 0.275000927419906 z_k1_s0144
                                 - 0.28200079185048 z_k1_s0145
                                 - 0.542003000407846 z_k1_s0146
                                 - 0.642001682846477 z_k1_s0147
                                 - 0.0890000907333758 z_k1_s0150
                                 - 0.195000232932699 z_k1_s0151
                                 - 0.135000149576477 z_k1_s0152
                                 - 0.21499939995088 z_k1_s0153
                                 - 0.158999909498788 z_k1_s0154
                                 - 0.144000078534856 z_k1_s0155
                                 - 0.167999990357266 z_k1_s0156
                                 - 0.414000681488007 z_k1_s0157
                                 - 0.636997564596516 z_k1_s0160
                                 - 0.374000041332142 z_k1_s0161
                                 - 0.602999138565441 z_k1_s0162
                                 - 1.09999978127119 z_k1_s0163
                                 - 0.646999824015902 z_k1_s0164
                                 - 0.861999229060125 z_k1_s0165
                                 - 0.811999501956742 z_k1_s0166
                                 - 0.750001691547515 z_k1_s0167
                                 - 0.791997873998306 z_k1_s0170
                                 - 1.04199723555583 z_k1_s0171
                                 - 0.433997823032712 z_k1_s0172
                                 - 0.37099986264271 z_k1_s0173
                                 - 0.287999971189697 z_k1_s0174
                                 - 0.317999636036806 z_k1_s0175
                                 - 0.796000931788767 z_k1_s0176
                                 - 0.267999763725302 z_k1_s0177
                                 - 0.373999253630825 z_k1_s1000
                                 - 0.604000420081369 z_k1_s1001
                                 - 0.498997571435966 z_k1_s1002
                                 - 0.388999380794908 z_k1_s1003
                                 - 0.654000843961646 z_k1_s1004
                                 - 0.486999819900886 z_k1_s1005
                                 - 0.6729982631406 z_k1_s1006
                                 - 0.236999658996325 z_k1_s1007
                                 - 0.312999260432825 z_k1_s1010
                                 - 0.418000973543758 z_k1_s1011
                                 - 0.334000515855421 z_k1_s1012
                                 - 0.254000405961913 z_k1_s1013
                                 - 0.456000854593098 z_k1_s1014
                                 - 0.116000045245535 z_k1_s1015
                                 - 0.200000046533668 z_k1_s1016
                                 - 0.191000062695661 z_k1_s1017
                                 - 0.23600012170243 z_k1_s1020
                                 - 0.266999861803492 z_k1_s1021
                                 - 0.144999745384501 z_k1_s1022
                                 - 0.402998594311526 z_k1_s1023
                                 - 0.123999946903107 z_k1_s1024
                                 - 0.320000499426636 z_k1_s1025
                                 - 0.212000577494861 z_k1_s1026
                                 - 0.265999551961891 z_k1_s1027
                                 - 0.548998876345698 z_k1_s1030
                                 - 0.65499942406299 z_k1_s1031
                                 - 0.325000067842067 z_k1_s1032
                                 - 0.778002039829986 z_k1_s1033
                                 - 0.27499994605094 z_k1_s1034
                                 - 0.186999948838634 z_k1_s1035
                                 - 0.632000670879535 z_k1_s1036
                                 - 0.217999951969107 z_k1_s1037
                                 - 0.154999705707013 z_k1_s1040
                                 - 0.215999763581754 z_k1_s1041
                                 - 0.471999623474721 z_k1_s1042
                                 - 0.225999565159912 z_k1_s1043
                                 - 0.137999842774464 z_k1_s1044
                                 - 0.572998887662093 z_k1_s1045
                                 - 0.137999842774464 z_k1_s1046
                                 - 0.307000561188842 z_k1_s1047
                                 - 0.389999469236644 z_k1_s1050
                                 - 0.620000021992062 z_k1_s1051
                                 - 0.164000071168904 z_k1_s1052
                                 - 0.401000112251371 z_k1_s1053
                                 - 0.128000033585474 z_k1_s1054
                                 - 0.221000149434345 z_k1_s1055
                                 - 0.161000161828669 z_k1_s1056
                                 - 0.166000085762434 z_k1_s1057
                                 - 0.280998773417178 z_k1_s1060
                                 - 0.440997907050312 z_k1_s1061
                                 - 0.290999047018121 z_k1_s1062
                                 - 0.261999040172679 z_k1_s1063
                                 - 0.30299989973745 z_k1_s1064
                                 - 0.201999233960705 z_k1_s1065
                                 - 0.153999493870765 z_k1_s1066
                                 - 0.68799732341887 z_k1_s1067
                                 - 0.197000498280776 z_k1_s1070
                                 - 0.599001260683562 z_k1_s1071
                                 - 0.283000547230345 z_k1_s1072
                                 - 0.262000633570808 z_k1_s1073
                                 - 0.454001353675997 z_k1_s1074
                                 - 0.49100230852171 z_k1_s1075
                                 - 0.211000541354048 z_k1_s1076
                                 - 0.522001476631678 z_k1_s1077
                                 - 0.524000996966913 z_k1_s1100
                                 - 2.03200016966456 z_k1_s1101
                                 - 1.17600095232685 z_k1_s1102
                                 - 0.948003512777845 z_k1_s1103
                                 - 1.17600095232685 z_k1_s1104
                                 - 1.05100285088187 z_k1_s1105
                                 - 1.02299890243957 z_k1_s1106
                                 - 0.748001343657992 z_k1_s1107
                                 - 0.574002156971361 z_k1_s1110
                                 - 0.266000414417955 z_k1_s1111
                                 - 0.757000864441439 z_k1_s1112
                                 - 0.176000321515678 z_k1_s1113
                                 - 0.806002151732584 z_k1_s1114
                                 - 0.498999785300911 z_k1_s1115
                                 - 0.501001683946966 z_k1_s1116
                                 - 0.35400018187547 z_k1_s1117
                                 - 0.53000300043814 z_k1_s1120
                                 - 0.622000408962676 z_k1_s1121
                                 - 0.573000200507171 z_k1_s1122
                                 - 0.839001332122772 z_k1_s1123
                                 - 0.420000742760015 z_k1_s1124
                                 - 0.511000607890939 z_k1_s1125
                                 - 0.609999612197021 z_k1_s1126
                                 - 1.11500399771097 z_k1_s1127
                                 - 0.497999987669348 z_k1_s1130
                                 - 0.197000872583043 z_k1_s1131
                                 - 0.138000133420576 z_k1_s1132
                                 - 0.25800034150379 z_k1_s1133
                                 - 0.229000183737726 z_k1_s1134
                                 - 0.131000055511481 z_k1_s1135
                                 - 0.121000054212898 z_k1_s1136
                                 - 0.353001123841199 z_k1_s1137
                                 - 0.32500052596138 z_k1_s1140
                                 - 0.339999895466875 z_k1_s1141
                                 - 0.268000185838765 z_k1_s1142
                                 - 0.646000409387447 z_k1_s1143
                                 - 0.732999184520859 z_k1_s1144
                                 - 0.212000006349979 z_k1_s1145
                                 - 0.348001180537499 z_k1_s1146
                                 - 0.168000234466166 z_k1_s1147
                                 - 0.439999937817314 z_k1_s1150
                                 - 0.254000072067286 z_k1_s1151
                                 - 0.470999272158875 z_k1_s1152
                                 - 0.215000172047132 z_k1_s1153
                                 - 0.383998650167615 z_k1_s1154
                                 - 0.453998775229483 z_k1_s1155
                                 - 0.537000153531625 z_k1_s1156
                                 - 0.213999915774775 z_k1_s1157
                                 - 0.294998839634609 z_k1_s1160
                                 - 0.16999929883364 z_k1_s1161
                                 - 0.783995954737363 z_k1_s1162
                                 - 0.164999432185119 z_k1_s1163
                                 - 0.229998976210533 z_k1_s1164
                                 - 0.615998646714566 z_k1_s1165
                                 - 0.405997220706145 z_k1_s1166
                                 - 0.527997927473452 z_k1_s1167
                                 - 0.33299984445234 z_k1_s1170
                                 - 0.175000011913433 z_k1_s1171
                                 - 0.261999653653095 z_k1_s1172
                                 - 0.180000038082399 z_k1_s1173
                                 - 0.230000299772063 z_k1_s1174
                                 - 0.209000144662385 z_k1_s1175
                                 - 0.711999748859093 z_k1_s1176
                                 - 0.159000108972818 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.928802 w_k1_ADM_s000
                                 + 0.884499 w_k1_AXP_s000
                                 + 0.883872 w_k1_DIS_s000
                                 + 0.830979 w_k1_FDX_s000
                                 + 1.06068 w_k1_IBM_s000
                                 + 0.890929 w_k1_INTC_s000
                                 + 1.10679 w_k1_KO_s000 + 1.07416 w_k1_MAT_s000
                                 + 0.929061 w_k1_MSFT_s000
                                 + 0.928366 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.608494 w_k1_ADM_s000
                                 + 0.676058 w_k1_AXP_s000
                                 + 0.630834 w_k1_DIS_s000
                                 + 0.679823 w_k1_FDX_s000
                                 + 0.739045 w_k1_IBM_s000
                                 + 0.7841 w_k1_INTC_s000 + 0.81357 w_k1_KO_s000
                                 + 1.15073 w_k1_MAT_s000
                                 + 0.733454 w_k1_MSFT_s000
                                 + 0.7847 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.814496 w_k1_ADM_s000
                                 + 1.51628 w_k1_AXP_s000
                                 + 1.35817 w_k1_DIS_s000
                                 + 1.10637 w_k1_FDX_s000
                                 + 1.58509 w_k1_IBM_s000
                                 + 1.32928 w_k1_INTC_s000
                                 + 1.46161 w_k1_KO_s000 + 2.36111 w_k1_MAT_s000
                                 + 1.15751 w_k1_MSFT_s000
                                 + 1.18263 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.879807 w_k1_ADM_s000
                                 + 1.19397 w_k1_AXP_s000
                                 + 0.846241 w_k1_DIS_s000
                                 + 0.970432 w_k1_FDX_s000
                                 + 0.820153 w_k1_IBM_s000
                                 + 1.00162 w_k1_INTC_s000
                                 + 0.886629 w_k1_KO_s000
                                 + 1.26935 w_k1_MAT_s000
                                 + 0.975409 w_k1_MSFT_s000
                                 + 0.884315 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.04201 w_k1_ADM_s000
                                 + 0.990116 w_k1_AXP_s000
                                 + 1.1734 w_k1_DIS_s000 + 1.03326 w_k1_FDX_s000
                                 + 0.881381 w_k1_IBM_s000
                                 + 1.07334 w_k1_INTC_s000
                                 + 1.15612 w_k1_KO_s000 + 0.84885 w_k1_MAT_s000
                                 + 1.1649 w_k1_MSFT_s000 + 1.05533 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.962365 w_k1_ADM_s000
                                 + 1.1744 w_k1_AXP_s000 + 1.06465 w_k1_DIS_s000
                                 + 0.976761 w_k1_FDX_s000
                                 + 1.36456 w_k1_IBM_s000
                                 + 1.06285 w_k1_INTC_s000
                                 + 1.09696 w_k1_KO_s000 + 1.35189 w_k1_MAT_s000
                                 + 0.926759 w_k1_MSFT_s000
                                 + 1.04228 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.00566 w_k1_ADM_s000
                                 + 0.816695 w_k1_AXP_s000
                                 + 0.772632 w_k1_DIS_s000
                                 + 0.868373 w_k1_FDX_s000
                                 + 0.679635 w_k1_IBM_s000
                                 + 0.931328 w_k1_INTC_s000
                                 + 0.885259 w_k1_KO_s000
                                 + 0.71805 w_k1_MAT_s000
                                 + 1.02691 w_k1_MSFT_s000
                                 + 0.871897 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.34712 w_k1_ADM_s000
                                 + 1.14381 w_k1_AXP_s000
                                 + 1.55679 w_k1_DIS_s000
                                 + 1.36213 w_k1_FDX_s000
                                 + 0.977846 w_k1_IBM_s000
                                 + 1.60959 w_k1_INTC_s000
                                 + 1.18976 w_k1_KO_s000
                                 + 0.841203 w_k1_MAT_s000
                                 + 1.33188 w_k1_MSFT_s000
                                 + 1.01225 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.745081 w_k1_ADM_s001
                                 + 0.673311 w_k1_AXP_s001
                                 + 0.774257 w_k1_DIS_s001
                                 + 0.720362 w_k1_FDX_s001
                                 + 1.27548 w_k1_IBM_s001
                                 + 0.966496 w_k1_INTC_s001
                                 + 0.968015 w_k1_KO_s001
                                 + 0.897846 w_k1_MAT_s001
                                 + 0.925054 w_k1_MSFT_s001
                                 + 0.97478 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.10105 w_k1_ADM_s001
                                 + 1.23781 w_k1_AXP_s001
                                 + 1.07939 w_k1_DIS_s001
                                 + 1.11955 w_k1_FDX_s001
                                 + 1.20544 w_k1_IBM_s001
                                 + 1.25343 w_k1_INTC_s001
                                 + 1.16036 w_k1_KO_s001 + 1.57069 w_k1_MAT_s001
                                 + 1.10381 w_k1_MSFT_s001
                                 + 1.18014 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.32097 w_k1_ADM_s001
                                 + 1.11904 w_k1_AXP_s001
                                 + 0.799852 w_k1_DIS_s001
                                 + 1.06186 w_k1_FDX_s001
                                 + 0.810185 w_k1_IBM_s001
                                 + 1.04702 w_k1_INTC_s001
                                 + 0.763263 w_k1_KO_s001
                                 + 0.748033 w_k1_MAT_s001
                                 + 0.993516 w_k1_MSFT_s001
                                 + 1.03881 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.913615 w_k1_ADM_s001
                                 + 0.683836 w_k1_AXP_s001
                                 + 0.800791 w_k1_DIS_s001
                                 + 0.859125 w_k1_FDX_s001
                                 + 0.731076 w_k1_IBM_s001
                                 + 0.916754 w_k1_INTC_s001
                                 + 0.729288 w_k1_KO_s001
                                 + 0.724979 w_k1_MAT_s001
                                 + 0.886009 w_k1_MSFT_s001
                                 + 0.914922 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.06501 w_k1_ADM_s001
                                 + 0.996145 w_k1_AXP_s001
                                 + 1.59849 w_k1_DIS_s001
                                 + 1.28239 w_k1_FDX_s001
                                 + 1.29519 w_k1_IBM_s001
                                 + 1.19871 w_k1_INTC_s001 + 1.1005 w_k1_KO_s001
                                 + 1.21475 w_k1_MAT_s001
                                 + 1.16523 w_k1_MSFT_s001
                                 + 1.10616 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.906644 w_k1_ADM_s001
                                 + 0.928811 w_k1_AXP_s001
                                 + 0.90794 w_k1_DIS_s001
                                 + 0.894135 w_k1_FDX_s001
                                 + 1.15088 w_k1_IBM_s001
                                 + 0.9168 w_k1_INTC_s001
                                 + 0.968843 w_k1_KO_s001
                                 + 1.27207 w_k1_MAT_s001
                                 + 0.968256 w_k1_MSFT_s001
                                 + 1.05921 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.44226 w_k1_ADM_s001
                                 + 1.39558 w_k1_AXP_s001
                                 + 1.54496 w_k1_DIS_s001
                                 + 1.45924 w_k1_FDX_s001
                                 + 1.32947 w_k1_IBM_s001
                                 + 1.36093 w_k1_INTC_s001
                                 + 1.33994 w_k1_KO_s001 + 1.60483 w_k1_MAT_s001
                                 + 1.65592 w_k1_MSFT_s001
                                 + 1.41205 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.08108 w_k1_ADM_s001
                                 + 1.10709 w_k1_AXP_s001
                                 + 1.16491 w_k1_DIS_s001
                                 + 1.24017 w_k1_FDX_s001
                                 + 0.811217 w_k1_IBM_s001
                                 + 1.07258 w_k1_INTC_s001
                                 + 0.961245 w_k1_KO_s001
                                 + 1.12193 w_k1_MAT_s001
                                 + 1.19273 w_k1_MSFT_s001
                                 + 1.05916 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.58134 w_k1_ADM_s002
                                 + 0.597786 w_k1_AXP_s002
                                 + 0.66734 w_k1_DIS_s002
                                 + 0.696849 w_k1_FDX_s002
                                 + 0.978999 w_k1_IBM_s002
                                 + 0.855033 w_k1_INTC_s002
                                 + 0.814098 w_k1_KO_s002
                                 + 0.694145 w_k1_MAT_s002
                                 + 0.75638 w_k1_MSFT_s002
                                 + 0.785937 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.927683 w_k1_ADM_s002
                                 + 0.99778 w_k1_AXP_s002
                                 + 0.769214 w_k1_DIS_s002
                                 + 0.987374 w_k1_FDX_s002
                                 + 0.794707 w_k1_IBM_s002
                                 + 1.17804 w_k1_INTC_s002
                                 + 0.780262 w_k1_KO_s002
                                 + 0.728579 w_k1_MAT_s002
                                 + 1.03588 w_k1_MSFT_s002
                                 + 0.910002 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.974354 w_k1_ADM_s002
                                 + 1.29631 w_k1_AXP_s002
                                 + 1.58374 w_k1_DIS_s002
                                 + 1.36344 w_k1_FDX_s002
                                 + 1.37899 w_k1_IBM_s002
                                 + 1.31991 w_k1_INTC_s002 + 1.0632 w_k1_KO_s002
                                 + 1.19203 w_k1_MAT_s002
                                 + 1.15847 w_k1_MSFT_s002
                                 + 1.22573 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.884682 w_k1_ADM_s002
                                 + 1.12012 w_k1_AXP_s002
                                 + 1.00815 w_k1_DIS_s002
                                 + 1.02706 w_k1_FDX_s002
                                 + 1.08404 w_k1_IBM_s002
                                 + 0.946702 w_k1_INTC_s002
                                 + 0.979774 w_k1_KO_s002
                                 + 1.40846 w_k1_MAT_s002
                                 + 1.01472 w_k1_MSFT_s002
                                 + 1.12614 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.90108 w_k1_ADM_s002
                                 + 0.971305 w_k1_AXP_s002
                                 + 1.17149 w_k1_DIS_s002
                                 + 1.22854 w_k1_FDX_s002
                                 + 0.812339 w_k1_IBM_s002
                                 + 1.1505 w_k1_INTC_s002 + 1.00246 w_k1_KO_s002
                                 + 0.913795 w_k1_MAT_s002
                                 + 1.19679 w_k1_MSFT_s002
                                 + 0.983571 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.845961 w_k1_ADM_s002
                                 + 1.08454 w_k1_AXP_s002
                                 + 0.692829 w_k1_DIS_s002
                                 + 0.924323 w_k1_FDX_s002
                                 + 1.23426 w_k1_IBM_s002
                                 + 0.944175 w_k1_INTC_s002
                                 + 0.932177 w_k1_KO_s002
                                 + 1.07729 w_k1_MAT_s002
                                 + 0.777774 w_k1_MSFT_s002
                                 + 0.926367 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.768433 w_k1_ADM_s002
                                 + 0.806213 w_k1_AXP_s002
                                 + 0.678187 w_k1_DIS_s002
                                 + 0.780393 w_k1_FDX_s002
                                 + 0.864805 w_k1_IBM_s002
                                 + 0.751787 w_k1_INTC_s002
                                 + 0.699505 w_k1_KO_s002
                                 + 0.907068 w_k1_MAT_s002
                                 + 0.730602 w_k1_MSFT_s002
                                 + 0.87761 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.776936 w_k1_ADM_s002
                                 + 0.66787 w_k1_AXP_s002
                                 + 0.663625 w_k1_DIS_s002
                                 + 0.865015 w_k1_FDX_s002
                                 + 0.622763 w_k1_IBM_s002
                                 + 0.762591 w_k1_INTC_s002
                                 + 0.699053 w_k1_KO_s002
                                 + 0.569312 w_k1_MAT_s002
                                 + 0.84365 w_k1_MSFT_s002
                                 + 0.88526 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.675081 w_k1_ADM_s003
                                 + 0.813714 w_k1_AXP_s003
                                 + 0.786351 w_k1_DIS_s003
                                 + 0.916379 w_k1_FDX_s003
                                 + 0.689779 w_k1_IBM_s003
                                 + 0.940971 w_k1_INTC_s003
                                 + 0.669683 w_k1_KO_s003
                                 + 0.561773 w_k1_MAT_s003
                                 + 0.8836 w_k1_MSFT_s003
                                 + 0.817391 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.848723 w_k1_ADM_s003
                                 + 0.932905 w_k1_AXP_s003
                                 + 0.755146 w_k1_DIS_s003
                                 + 0.870044 w_k1_FDX_s003
                                 + 0.953021 w_k1_IBM_s003
                                 + 0.783582 w_k1_INTC_s003
                                 + 0.766853 w_k1_KO_s003
                                 + 1.09411 w_k1_MAT_s003
                                 + 0.786306 w_k1_MSFT_s003
                                 + 0.959886 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.843332 w_k1_ADM_s003
                                 + 1.23049 w_k1_AXP_s003
                                 + 1.33472 w_k1_DIS_s003
                                 + 1.20377 w_k1_FDX_s003
                                 + 0.848224 w_k1_IBM_s003
                                 + 1.29418 w_k1_INTC_s003
                                 + 0.883015 w_k1_KO_s003
                                 + 0.795064 w_k1_MAT_s003
                                 + 1.30634 w_k1_MSFT_s003
                                 + 1.02288 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.32434 w_k1_ADM_s003
                                 + 1.14802 w_k1_AXP_s003
                                 + 1.22585 w_k1_DIS_s003
                                 + 1.12505 w_k1_FDX_s003
                                 + 1.36573 w_k1_IBM_s003
                                 + 0.974798 w_k1_INTC_s003
                                 + 1.10458 w_k1_KO_s003 + 1.33139 w_k1_MAT_s003
                                 + 1.04864 w_k1_MSFT_s003
                                 + 1.11821 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.09742 w_k1_ADM_s003
                                 + 1.24735 w_k1_AXP_s003
                                 + 1.70451 w_k1_DIS_s003
                                 + 1.24924 w_k1_FDX_s003
                                 + 1.31768 w_k1_IBM_s003
                                 + 1.08628 w_k1_INTC_s003
                                 + 1.38499 w_k1_KO_s003 + 1.47961 w_k1_MAT_s003
                                 + 1.20589 w_k1_MSFT_s003
                                 + 1.04604 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.668409 w_k1_ADM_s003
                                 + 0.879503 w_k1_AXP_s003
                                 + 0.834134 w_k1_DIS_s003
                                 + 0.863406 w_k1_FDX_s003
                                 + 1.49527 w_k1_IBM_s003
                                 + 0.936233 w_k1_INTC_s003
                                 + 0.939481 w_k1_KO_s003
                                 + 1.53689 w_k1_MAT_s003
                                 + 0.881292 w_k1_MSFT_s003
                                 + 1.06473 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.778345 w_k1_ADM_s003
                                 + 0.947061 w_k1_AXP_s003
                                 + 0.996934 w_k1_DIS_s003
                                 + 0.999197 w_k1_FDX_s003
                                 + 0.910929 w_k1_IBM_s003
                                 + 0.948467 w_k1_INTC_s003
                                 + 0.840991 w_k1_KO_s003
                                 + 1.26206 w_k1_MAT_s003
                                 + 1.03746 w_k1_MSFT_s003
                                 + 0.933877 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.935877 w_k1_ADM_s003
                                 + 0.929671 w_k1_AXP_s003
                                 + 1.1818 w_k1_DIS_s003
                                 + 0.916065 w_k1_FDX_s003
                                 + 1.50677 w_k1_IBM_s003
                                 + 1.11756 w_k1_INTC_s003
                                 + 1.21503 w_k1_KO_s003 + 1.22304 w_k1_MAT_s003
                                 + 1.23439 w_k1_MSFT_s003
                                 + 1.05411 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.02819 w_k1_ADM_s004
                                 + 1.31154 w_k1_AXP_s004
                                 + 1.33395 w_k1_DIS_s004 + 1.3373 w_k1_FDX_s004
                                 + 1.11475 w_k1_IBM_s004
                                 + 1.57365 w_k1_INTC_s004
                                 + 1.21839 w_k1_KO_s004 + 1.1711 w_k1_MAT_s004
                                 + 1.36238 w_k1_MSFT_s004
                                 + 1.27385 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.02717 w_k1_ADM_s004
                                 + 1.08917 w_k1_AXP_s004
                                 + 1.02855 w_k1_DIS_s004
                                 + 1.07966 w_k1_FDX_s004
                                 + 1.11503 w_k1_IBM_s004
                                 + 1.02976 w_k1_INTC_s004
                                 + 1.12463 w_k1_KO_s004 + 1.86117 w_k1_MAT_s004
                                 + 0.998675 w_k1_MSFT_s004
                                 + 1.1567 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.04476 w_k1_ADM_s004
                                 + 0.7999 w_k1_AXP_s004
                                 + 0.899313 w_k1_DIS_s004
                                 + 0.837629 w_k1_FDX_s004
                                 + 1.09167 w_k1_IBM_s004
                                 + 0.871803 w_k1_INTC_s004
                                 + 1.01468 w_k1_KO_s004 + 1.103 w_k1_MAT_s004
                                 + 0.959883 w_k1_MSFT_s004
                                 + 1.12988 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.982301 w_k1_ADM_s004
                                 + 1.35301 w_k1_AXP_s004
                                 + 0.878692 w_k1_DIS_s004
                                 + 0.925273 w_k1_FDX_s004
                                 + 1.2531 w_k1_IBM_s004
                                 + 0.89992 w_k1_INTC_s004
                                 + 0.909493 w_k1_KO_s004
                                 + 2.10404 w_k1_MAT_s004
                                 + 0.882935 w_k1_MSFT_s004
                                 + 0.948476 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.783731 w_k1_ADM_s004
                                 + 0.864293 w_k1_AXP_s004
                                 + 0.70066 w_k1_DIS_s004
                                 + 0.738271 w_k1_FDX_s004
                                 + 1.16505 w_k1_IBM_s004
                                 + 0.820688 w_k1_INTC_s004
                                 + 0.853013 w_k1_KO_s004
                                 + 1.29212 w_k1_MAT_s004
                                 + 0.789652 w_k1_MSFT_s004
                                 + 0.872076 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.2568 w_k1_ADM_s004
                                 + 0.95853 w_k1_AXP_s004
                                 + 1.24463 w_k1_DIS_s004
                                 + 1.01338 w_k1_FDX_s004
                                 + 1.08265 w_k1_IBM_s004
                                 + 0.944226 w_k1_INTC_s004
                                 + 1.10786 w_k1_KO_s004 + 1.00476 w_k1_MAT_s004
                                 + 1.09646 w_k1_MSFT_s004
                                 + 1.18029 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.832473 w_k1_ADM_s004
                                 + 0.893493 w_k1_AXP_s004
                                 + 0.664819 w_k1_DIS_s004
                                 + 0.879108 w_k1_FDX_s004
                                 + 0.742131 w_k1_IBM_s004
                                 + 0.836244 w_k1_INTC_s004
                                 + 0.705034 w_k1_KO_s004
                                 + 1.11605 w_k1_MAT_s004
                                 + 0.813229 w_k1_MSFT_s004
                                 + 0.968397 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.13394 w_k1_ADM_s004
                                 + 1.36501 w_k1_AXP_s004
                                 + 1.27529 w_k1_DIS_s004
                                 + 1.10815 w_k1_FDX_s004
                                 + 1.68664 w_k1_IBM_s004
                                 + 1.18947 w_k1_INTC_s004 + 1.3219 w_k1_KO_s004
                                 + 2.58698 w_k1_MAT_s004
                                 + 1.2171 w_k1_MSFT_s004 + 1.20725 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.935727 w_k1_ADM_s005
                                 + 0.886439 w_k1_AXP_s005
                                 + 0.737605 w_k1_DIS_s005
                                 + 0.831634 w_k1_FDX_s005
                                 + 1.15851 w_k1_IBM_s005
                                 + 0.914725 w_k1_INTC_s005
                                 + 0.987432 w_k1_KO_s005
                                 + 0.901317 w_k1_MAT_s005
                                 + 0.824254 w_k1_MSFT_s005
                                 + 0.963016 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.39726 w_k1_ADM_s005
                                 + 1.58587 w_k1_AXP_s005
                                 + 1.18733 w_k1_DIS_s005
                                 + 1.28346 w_k1_FDX_s005
                                 + 0.909909 w_k1_IBM_s005
                                 + 1.09563 w_k1_INTC_s005
                                 + 0.997214 w_k1_KO_s005
                                 + 0.784489 w_k1_MAT_s005
                                 + 1.00869 w_k1_MSFT_s005 + 1.0803 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.03644 w_k1_ADM_s005
                                 + 1.3638 w_k1_AXP_s005 + 1.1932 w_k1_DIS_s005
                                 + 1.01136 w_k1_FDX_s005
                                 + 1.90688 w_k1_IBM_s005
                                 + 0.963603 w_k1_INTC_s005
                                 + 1.44252 w_k1_KO_s005 + 1.81142 w_k1_MAT_s005
                                 + 0.912035 w_k1_MSFT_s005
                                 + 1.10735 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.04475 w_k1_ADM_s005
                                 + 0.980613 w_k1_AXP_s005
                                 + 1.36542 w_k1_DIS_s005
                                 + 1.16038 w_k1_FDX_s005
                                 + 0.903462 w_k1_IBM_s005
                                 + 1.21889 w_k1_INTC_s005
                                 + 1.04267 w_k1_KO_s005
                                 + 0.844137 w_k1_MAT_s005
                                 + 1.29189 w_k1_MSFT_s005
                                 + 1.05809 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.812415 w_k1_ADM_s005
                                 + 0.823231 w_k1_AXP_s005
                                 + 0.792722 w_k1_DIS_s005
                                 + 0.919712 w_k1_FDX_s005
                                 + 0.761601 w_k1_IBM_s005
                                 + 0.954884 w_k1_INTC_s005
                                 + 0.903451 w_k1_KO_s005
                                 + 0.739264 w_k1_MAT_s005
                                 + 0.848787 w_k1_MSFT_s005
                                 + 0.94912 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.16861 w_k1_ADM_s005
                                 + 1.36207 w_k1_AXP_s005
                                 + 1.64675 w_k1_DIS_s005
                                 + 1.23336 w_k1_FDX_s005
                                 + 1.23572 w_k1_IBM_s005
                                 + 1.18186 w_k1_INTC_s005
                                 + 1.35858 w_k1_KO_s005 + 1.36995 w_k1_MAT_s005
                                 + 1.25554 w_k1_MSFT_s005
                                 + 1.20746 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.944505 w_k1_ADM_s005
                                 + 1.28725 w_k1_AXP_s005
                                 + 0.912757 w_k1_DIS_s005
                                 + 1.02781 w_k1_FDX_s005
                                 + 1.17364 w_k1_IBM_s005
                                 + 0.813823 w_k1_INTC_s005
                                 + 1.04989 w_k1_KO_s005 + 1.34014 w_k1_MAT_s005
                                 + 0.953789 w_k1_MSFT_s005
                                 + 0.98702 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.941922 w_k1_ADM_s005
                                 + 1.00888 w_k1_AXP_s005
                                 + 1.03379 w_k1_DIS_s005
                                 + 0.909632 w_k1_FDX_s005
                                 + 1.42705 w_k1_IBM_s005
                                 + 0.97061 w_k1_INTC_s005
                                 + 1.19054 w_k1_KO_s005 + 1.02071 w_k1_MAT_s005
                                 + 0.881844 w_k1_MSFT_s005
                                 + 1.10759 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.770096 w_k1_ADM_s006
                                 + 0.662793 w_k1_AXP_s006
                                 + 0.579153 w_k1_DIS_s006
                                 + 0.681154 w_k1_FDX_s006
                                 + 0.807722 w_k1_IBM_s006
                                 + 0.787899 w_k1_INTC_s006
                                 + 0.753751 w_k1_KO_s006
                                 + 0.800813 w_k1_MAT_s006
                                 + 0.779948 w_k1_MSFT_s006
                                 + 0.803415 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.776354 w_k1_ADM_s006
                                 + 0.780327 w_k1_AXP_s006
                                 + 0.65821 w_k1_DIS_s006
                                 + 0.840466 w_k1_FDX_s006
                                 + 0.646712 w_k1_IBM_s006
                                 + 0.962931 w_k1_INTC_s006
                                 + 0.640653 w_k1_KO_s006
                                 + 0.629116 w_k1_MAT_s006
                                 + 0.884526 w_k1_MSFT_s006
                                 + 0.838826 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.863671 w_k1_ADM_s006
                                 + 0.746041 w_k1_AXP_s006
                                 + 0.708496 w_k1_DIS_s006
                                 + 0.766359 w_k1_FDX_s006
                                 + 1.09599 w_k1_IBM_s006
                                 + 0.84307 w_k1_INTC_s006
                                 + 0.868548 w_k1_KO_s006
                                 + 1.12559 w_k1_MAT_s006
                                 + 0.873418 w_k1_MSFT_s006
                                 + 0.937736 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.2515 w_k1_ADM_s006
                                 + 1.28609 w_k1_AXP_s006
                                 + 1.53853 w_k1_DIS_s006
                                 + 1.25747 w_k1_FDX_s006
                                 + 1.21666 w_k1_IBM_s006
                                 + 1.09771 w_k1_INTC_s006
                                 + 1.22718 w_k1_KO_s006 + 1.17616 w_k1_MAT_s006
                                 + 1.25538 w_k1_MSFT_s006
                                 + 1.30081 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.08732 w_k1_ADM_s006
                                 + 1.05348 w_k1_AXP_s006 + 1.1644 w_k1_DIS_s006
                                 + 1.04631 w_k1_FDX_s006
                                 + 1.21437 w_k1_IBM_s006
                                 + 1.50285 w_k1_INTC_s006
                                 + 1.03624 w_k1_KO_s006
                                 + 0.876502 w_k1_MAT_s006
                                 + 1.11495 w_k1_MSFT_s006
                                 + 1.12876 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.986912 w_k1_ADM_s006
                                 + 1.22926 w_k1_AXP_s006
                                 + 1.01055 w_k1_DIS_s006
                                 + 0.869564 w_k1_FDX_s006
                                 + 1.49325 w_k1_IBM_s006
                                 + 1.07389 w_k1_INTC_s006
                                 + 1.15225 w_k1_KO_s006 + 1.52358 w_k1_MAT_s006
                                 + 0.94364 w_k1_MSFT_s006
                                 + 1.05512 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.976103 w_k1_ADM_s006
                                 + 0.954405 w_k1_AXP_s006
                                 + 0.988439 w_k1_DIS_s006
                                 + 1.01353 w_k1_FDX_s006
                                 + 0.919428 w_k1_IBM_s006
                                 + 1.03652 w_k1_INTC_s006
                                 + 0.884629 w_k1_KO_s006
                                 + 1.11036 w_k1_MAT_s006
                                 + 1.10029 w_k1_MSFT_s006
                                 + 1.01815 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.908731 w_k1_ADM_s006
                                 + 0.915453 w_k1_AXP_s006
                                 + 1.01544 w_k1_DIS_s006
                                 + 1.02449 w_k1_FDX_s006
                                 + 0.668944 w_k1_IBM_s006
                                 + 1.20577 w_k1_INTC_s006
                                 + 0.820792 w_k1_KO_s006
                                 + 0.704691 w_k1_MAT_s006
                                 + 1.20366 w_k1_MSFT_s006
                                 + 0.871912 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.44156 w_k1_ADM_s007
                                 + 1.4077 w_k1_AXP_s007 + 1.65297 w_k1_DIS_s007
                                 + 1.45893 w_k1_FDX_s007
                                 + 1.03762 w_k1_IBM_s007
                                 + 1.40676 w_k1_INTC_s007 + 1.1089 w_k1_KO_s007
                                 + 0.954329 w_k1_MAT_s007
                                 + 1.23054 w_k1_MSFT_s007
                                 + 1.12764 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.736445 w_k1_ADM_s007
                                 + 0.887842 w_k1_AXP_s007
                                 + 0.710563 w_k1_DIS_s007
                                 + 0.836026 w_k1_FDX_s007
                                 + 1.13665 w_k1_IBM_s007
                                 + 0.953832 w_k1_INTC_s007
                                 + 0.928174 w_k1_KO_s007
                                 + 1.32528 w_k1_MAT_s007
                                 + 0.798236 w_k1_MSFT_s007
                                 + 0.974809 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.92433 w_k1_ADM_s007
                                 + 1.28659 w_k1_AXP_s007
                                 + 0.913116 w_k1_DIS_s007
                                 + 1.10743 w_k1_FDX_s007
                                 + 0.863885 w_k1_IBM_s007
                                 + 0.934251 w_k1_INTC_s007
                                 + 0.874345 w_k1_KO_s007
                                 + 0.883197 w_k1_MAT_s007
                                 + 0.987586 w_k1_MSFT_s007
                                 + 0.961945 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.907976 w_k1_ADM_s007
                                 + 1.12903 w_k1_AXP_s007
                                 + 1.08635 w_k1_DIS_s007
                                 + 1.05657 w_k1_FDX_s007
                                 + 1.47768 w_k1_IBM_s007
                                 + 1.1061 w_k1_INTC_s007 + 1.14028 w_k1_KO_s007
                                 + 1.29541 w_k1_MAT_s007
                                 + 0.957547 w_k1_MSFT_s007
                                 + 1.02014 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.866163 w_k1_ADM_s007
                                 + 0.86045 w_k1_AXP_s007
                                 + 1.06034 w_k1_DIS_s007
                                 + 0.892057 w_k1_FDX_s007
                                 + 0.990778 w_k1_IBM_s007
                                 + 1.10973 w_k1_INTC_s007
                                 + 0.966971 w_k1_KO_s007
                                 + 0.868954 w_k1_MAT_s007
                                 + 1.03206 w_k1_MSFT_s007
                                 + 1.04295 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.869751 w_k1_ADM_s007
                                 + 0.824907 w_k1_AXP_s007
                                 + 0.634982 w_k1_DIS_s007
                                 + 0.821592 w_k1_FDX_s007
                                 + 0.838243 w_k1_IBM_s007
                                 + 0.729885 w_k1_INTC_s007
                                 + 0.710178 w_k1_KO_s007
                                 + 0.967622 w_k1_MAT_s007
                                 + 0.722782 w_k1_MSFT_s007
                                 + 0.842516 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.18693 w_k1_ADM_s007
                                 + 1.4045 w_k1_AXP_s007 + 1.08446 w_k1_DIS_s007
                                 + 1.13199 w_k1_FDX_s007
                                 + 1.33509 w_k1_IBM_s007
                                 + 1.1893 w_k1_INTC_s007 + 1.04058 w_k1_KO_s007
                                 + 1.61951 w_k1_MAT_s007
                                 + 0.966939 w_k1_MSFT_s007
                                 + 1.01244 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.947967 w_k1_ADM_s007
                                 + 1.07061 w_k1_AXP_s007 + 1.4814 w_k1_DIS_s007
                                 + 1.19626 w_k1_FDX_s007
                                 + 0.974311 w_k1_IBM_s007
                                 + 1.06823 w_k1_INTC_s007
                                 + 1.09823 w_k1_KO_s007 + 1.15349 w_k1_MAT_s007
                                 + 1.17802 w_k1_MSFT_s007
                                 + 0.92441 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.801036 w_k1_ADM_s010
                                 + 1.15029 w_k1_AXP_s010
                                 + 1.05599 w_k1_DIS_s010
                                 + 1.06216 w_k1_FDX_s010
                                 + 1.30587 w_k1_IBM_s010
                                 + 1.12851 w_k1_INTC_s010
                                 + 1.24417 w_k1_KO_s010 + 1.52572 w_k1_MAT_s010
                                 + 1.02701 w_k1_MSFT_s010
                                 + 0.930256 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.01973 w_k1_ADM_s010
                                 + 1.23741 w_k1_AXP_s010
                                 + 1.03921 w_k1_DIS_s010
                                 + 1.16936 w_k1_FDX_s010
                                 + 0.89495 w_k1_IBM_s010
                                 + 0.961346 w_k1_INTC_s010
                                 + 0.971975 w_k1_KO_s010
                                 + 1.20173 w_k1_MAT_s010
                                 + 1.00002 w_k1_MSFT_s010
                                 + 0.934374 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.674409 w_k1_ADM_s010
                                 + 0.756695 w_k1_AXP_s010
                                 + 0.629873 w_k1_DIS_s010
                                 + 0.808177 w_k1_FDX_s010
                                 + 0.637579 w_k1_IBM_s010
                                 + 0.709722 w_k1_INTC_s010
                                 + 0.718745 w_k1_KO_s010
                                 + 0.784961 w_k1_MAT_s010
                                 + 0.720351 w_k1_MSFT_s010
                                 + 0.72383 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.724004 w_k1_ADM_s010
                                 + 0.775143 w_k1_AXP_s010
                                 + 0.698694 w_k1_DIS_s010
                                 + 0.853232 w_k1_FDX_s010
                                 + 0.949168 w_k1_IBM_s010
                                 + 0.914861 w_k1_INTC_s010
                                 + 0.82354 w_k1_KO_s010
                                 + 0.672489 w_k1_MAT_s010
                                 + 0.747185 w_k1_MSFT_s010
                                 + 0.792249 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.766466 w_k1_ADM_s010
                                 + 0.750399 w_k1_AXP_s010
                                 + 0.843833 w_k1_DIS_s010
                                 + 0.831179 w_k1_FDX_s010
                                 + 1.30892 w_k1_IBM_s010
                                 + 0.898282 w_k1_INTC_s010
                                 + 1.11147 w_k1_KO_s010 + 1.34861 w_k1_MAT_s010
                                 + 0.903914 w_k1_MSFT_s010
                                 + 0.943882 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.961289 w_k1_ADM_s010
                                 + 1.58053 w_k1_AXP_s010
                                 + 1.26961 w_k1_DIS_s010
                                 + 1.21448 w_k1_FDX_s010
                                 + 1.36383 w_k1_IBM_s010
                                 + 1.44658 w_k1_INTC_s010
                                 + 1.15727 w_k1_KO_s010 + 1.30276 w_k1_MAT_s010
                                 + 1.10106 w_k1_MSFT_s010
                                 + 1.06822 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.899905 w_k1_ADM_s010
                                 + 0.986175 w_k1_AXP_s010
                                 + 0.872958 w_k1_DIS_s010
                                 + 1.06894 w_k1_FDX_s010
                                 + 0.755126 w_k1_IBM_s010
                                 + 0.807581 w_k1_INTC_s010
                                 + 0.789227 w_k1_KO_s010
                                 + 0.922539 w_k1_MAT_s010
                                 + 0.896751 w_k1_MSFT_s010
                                 + 0.831378 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.14659 w_k1_ADM_s010
                                 + 1.00656 w_k1_AXP_s010 + 1.5449 w_k1_DIS_s010
                                 + 1.26211 w_k1_FDX_s010
                                 + 1.09166 w_k1_IBM_s010
                                 + 1.16628 w_k1_INTC_s010
                                 + 1.26155 w_k1_KO_s010 + 1.08022 w_k1_MAT_s010
                                 + 1.27903 w_k1_MSFT_s010
                                 + 0.987636 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.14652 w_k1_ADM_s011
                                 + 1.3373 w_k1_AXP_s011 + 1.50124 w_k1_DIS_s011
                                 + 1.37313 w_k1_FDX_s011
                                 + 1.14145 w_k1_IBM_s011
                                 + 1.20285 w_k1_INTC_s011
                                 + 1.12569 w_k1_KO_s011 + 1.72267 w_k1_MAT_s011
                                 + 1.39931 w_k1_MSFT_s011
                                 + 1.19596 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.927728 w_k1_ADM_s011
                                 + 0.904226 w_k1_AXP_s011
                                 + 0.857083 w_k1_DIS_s011
                                 + 0.903745 w_k1_FDX_s011
                                 + 0.941507 w_k1_IBM_s011
                                 + 0.956259 w_k1_INTC_s011
                                 + 0.992942 w_k1_KO_s011
                                 + 1.21959 w_k1_MAT_s011
                                 + 1.07741 w_k1_MSFT_s011
                                 + 0.943678 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.32241 w_k1_ADM_s011
                                 + 1.12425 w_k1_AXP_s011
                                 + 1.91074 w_k1_DIS_s011
                                 + 1.43751 w_k1_FDX_s011
                                 + 1.15048 w_k1_IBM_s011
                                 + 1.33171 w_k1_INTC_s011
                                 + 1.46258 w_k1_KO_s011 + 1.46178 w_k1_MAT_s011
                                 + 1.88911 w_k1_MSFT_s011
                                 + 1.39062 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.01593 w_k1_ADM_s011
                                 + 0.939905 w_k1_AXP_s011
                                 + 1.22717 w_k1_DIS_s011
                                 + 1.13492 w_k1_FDX_s011
                                 + 0.856079 w_k1_IBM_s011
                                 + 1.24065 w_k1_INTC_s011
                                 + 1.04129 w_k1_KO_s011
                                 + 0.690819 w_k1_MAT_s011
                                 + 1.42958 w_k1_MSFT_s011
                                 + 1.09919 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.907643 w_k1_ADM_s011
                                 + 1.40179 w_k1_AXP_s011
                                 + 0.92657 w_k1_DIS_s011
                                 + 1.02835 w_k1_FDX_s011
                                 + 1.25733 w_k1_IBM_s011
                                 + 1.18924 w_k1_INTC_s011
                                 + 0.967287 w_k1_KO_s011
                                 + 1.94545 w_k1_MAT_s011
                                 + 1.11029 w_k1_MSFT_s011
                                 + 1.14989 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.00671 w_k1_ADM_s011
                                 + 1.01987 w_k1_AXP_s011
                                 + 1.03823 w_k1_DIS_s011
                                 + 0.932375 w_k1_FDX_s011
                                 + 1.57474 w_k1_IBM_s011
                                 + 0.996381 w_k1_INTC_s011
                                 + 1.29546 w_k1_KO_s011 + 1.66311 w_k1_MAT_s011
                                 + 1.15649 w_k1_MSFT_s011
                                 + 1.22504 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.90151 w_k1_ADM_s011
                                 + 0.747949 w_k1_AXP_s011
                                 + 0.745146 w_k1_DIS_s011
                                 + 0.787464 w_k1_FDX_s011
                                 + 0.743783 w_k1_IBM_s011
                                 + 0.762684 w_k1_INTC_s011
                                 + 0.762682 w_k1_KO_s011
                                 + 0.815862 w_k1_MAT_s011
                                 + 0.980231 w_k1_MSFT_s011
                                 + 0.827691 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.06823 w_k1_ADM_s011
                                 + 1.24767 w_k1_AXP_s011
                                 + 1.09754 w_k1_DIS_s011
                                 + 1.11214 w_k1_FDX_s011
                                 + 1.05199 w_k1_IBM_s011
                                 + 1.05279 w_k1_INTC_s011 + 1.1268 w_k1_KO_s011
                                 + 1.567 w_k1_MAT_s011 + 1.03368 w_k1_MSFT_s011
                                 + 1.14297 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.785247 w_k1_ADM_s012
                                 + 0.694437 w_k1_AXP_s012
                                 + 0.753985 w_k1_DIS_s012
                                 + 0.799103 w_k1_FDX_s012
                                 + 0.871139 w_k1_IBM_s012
                                 + 0.96226 w_k1_INTC_s012
                                 + 0.88636 w_k1_KO_s012
                                 + 0.895606 w_k1_MAT_s012
                                 + 0.861978 w_k1_MSFT_s012
                                 + 0.861039 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.05496 w_k1_ADM_s012
                                 + 1.12805 w_k1_AXP_s012
                                 + 1.12819 w_k1_DIS_s012
                                 + 1.05218 w_k1_FDX_s012
                                 + 1.25554 w_k1_IBM_s012
                                 + 1.38352 w_k1_INTC_s012
                                 + 1.16087 w_k1_KO_s012 + 1.07757 w_k1_MAT_s012
                                 + 1.07766 w_k1_MSFT_s012
                                 + 0.994381 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.965555 w_k1_ADM_s012
                                 + 1.12036 w_k1_AXP_s012
                                 + 1.01263 w_k1_DIS_s012
                                 + 1.11143 w_k1_FDX_s012
                                 + 0.864974 w_k1_IBM_s012
                                 + 1.12074 w_k1_INTC_s012
                                 + 1.01954 w_k1_KO_s012
                                 + 0.791554 w_k1_MAT_s012
                                 + 1.01303 w_k1_MSFT_s012
                                 + 0.907657 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.06754 w_k1_ADM_s012
                                 + 1.21176 w_k1_AXP_s012
                                 + 1.19771 w_k1_DIS_s012
                                 + 1.36412 w_k1_FDX_s012
                                 + 0.80603 w_k1_IBM_s012
                                 + 1.18204 w_k1_INTC_s012
                                 + 1.00784 w_k1_KO_s012 + 0.95978 w_k1_MAT_s012
                                 + 1.12704 w_k1_MSFT_s012
                                 + 0.967941 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07897 w_k1_ADM_s012
                                 + 1.28694 w_k1_AXP_s012
                                 + 1.13794 w_k1_DIS_s012
                                 + 1.04562 w_k1_FDX_s012
                                 + 1.57165 w_k1_IBM_s012
                                 + 1.25549 w_k1_INTC_s012
                                 + 1.39783 w_k1_KO_s012 + 1.67781 w_k1_MAT_s012
                                 + 1.05721 w_k1_MSFT_s012
                                 + 1.12779 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.02564 w_k1_ADM_s012
                                 + 0.654997 w_k1_AXP_s012
                                 + 1.21587 w_k1_DIS_s012
                                 + 1.07856 w_k1_FDX_s012
                                 + 1.03201 w_k1_IBM_s012
                                 + 1.02975 w_k1_INTC_s012
                                 + 1.18813 w_k1_KO_s012
                                 + 0.745614 w_k1_MAT_s012
                                 + 1.21127 w_k1_MSFT_s012
                                 + 0.967303 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.888704 w_k1_ADM_s012
                                 + 0.908727 w_k1_AXP_s012
                                 + 1.17893 w_k1_DIS_s012
                                 + 1.07304 w_k1_FDX_s012
                                 + 0.905134 w_k1_IBM_s012
                                 + 1.08612 w_k1_INTC_s012
                                 + 1.18326 w_k1_KO_s012
                                 + 0.859927 w_k1_MAT_s012
                                 + 1.13613 w_k1_MSFT_s012
                                 + 0.956608 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.765839 w_k1_ADM_s012
                                 + 0.780611 w_k1_AXP_s012
                                 + 0.639576 w_k1_DIS_s012
                                 + 0.801948 w_k1_FDX_s012
                                 + 0.703264 w_k1_IBM_s012
                                 + 0.961972 w_k1_INTC_s012
                                 + 0.694203 w_k1_KO_s012
                                 + 0.577201 w_k1_MAT_s012
                                 + 0.830644 w_k1_MSFT_s012
                                 + 0.81639 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.06067 w_k1_ADM_s013
                                 + 1.01383 w_k1_AXP_s013
                                 + 0.993983 w_k1_DIS_s013
                                 + 1.02065 w_k1_FDX_s013
                                 + 0.967502 w_k1_IBM_s013
                                 + 0.932614 w_k1_INTC_s013
                                 + 1.18287 w_k1_KO_s013
                                 + 0.868969 w_k1_MAT_s013
                                 + 1.12403 w_k1_MSFT_s013
                                 + 0.979948 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.99274 w_k1_ADM_s013
                                 + 0.99914 w_k1_AXP_s013
                                 + 0.890131 w_k1_DIS_s013
                                 + 1.01525 w_k1_FDX_s013 + 0.7367 w_k1_IBM_s013
                                 + 0.93525 w_k1_INTC_s013
                                 + 0.991936 w_k1_KO_s013
                                 + 0.692796 w_k1_MAT_s013
                                 + 1.11725 w_k1_MSFT_s013
                                 + 0.986686 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.63978 w_k1_ADM_s013
                                 + 1.49436 w_k1_AXP_s013
                                 + 1.58929 w_k1_DIS_s013
                                 + 1.48651 w_k1_FDX_s013
                                 + 1.04969 w_k1_IBM_s013
                                 + 1.29757 w_k1_INTC_s013
                                 + 1.24007 w_k1_KO_s013
                                 + 0.939051 w_k1_MAT_s013
                                 + 1.37889 w_k1_MSFT_s013
                                 + 1.15571 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.31035 w_k1_ADM_s013
                                 + 1.58802 w_k1_AXP_s013
                                 + 1.13742 w_k1_DIS_s013
                                 + 1.11967 w_k1_FDX_s013
                                 + 1.11073 w_k1_IBM_s013
                                 + 1.3151 w_k1_INTC_s013 + 1.23329 w_k1_KO_s013
                                 + 1.29375 w_k1_MAT_s013
                                 + 1.20881 w_k1_MSFT_s013
                                 + 1.10828 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06728 w_k1_ADM_s013
                                 + 0.965785 w_k1_AXP_s013
                                 + 0.838945 w_k1_DIS_s013
                                 + 0.912225 w_k1_FDX_s013
                                 + 1.25549 w_k1_IBM_s013
                                 + 0.847179 w_k1_INTC_s013
                                 + 1.12931 w_k1_KO_s013 + 1.51959 w_k1_MAT_s013
                                 + 1.00397 w_k1_MSFT_s013
                                 + 0.958461 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.30891 w_k1_ADM_s013
                                 + 1.80923 w_k1_AXP_s013
                                 + 1.18382 w_k1_DIS_s013 + 1.2619 w_k1_FDX_s013
                                 + 1.577 w_k1_IBM_s013
                                 + 0.984204 w_k1_INTC_s013
                                 + 1.38995 w_k1_KO_s013 + 2.5202 w_k1_MAT_s013
                                 + 1.2186 w_k1_MSFT_s013 + 1.16192 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.23431 w_k1_ADM_s013
                                 + 1.02936 w_k1_AXP_s013
                                 + 1.61785 w_k1_DIS_s013
                                 + 1.20819 w_k1_FDX_s013
                                 + 1.07287 w_k1_IBM_s013
                                 + 1.08576 w_k1_INTC_s013
                                 + 1.46852 w_k1_KO_s013
                                 + 0.718096 w_k1_MAT_s013
                                 + 1.33157 w_k1_MSFT_s013
                                 + 1.09207 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.804206 w_k1_ADM_s013
                                 + 0.656108 w_k1_AXP_s013
                                 + 0.621406 w_k1_DIS_s013
                                 + 0.75806 w_k1_FDX_s013
                                 + 0.802545 w_k1_IBM_s013
                                 + 0.710056 w_k1_INTC_s013
                                 + 0.807764 w_k1_KO_s013
                                 + 0.77886 w_k1_MAT_s013
                                 + 0.800337 w_k1_MSFT_s013
                                 + 0.808416 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.903429 w_k1_ADM_s014
                                 + 1.23711 w_k1_AXP_s014
                                 + 1.05761 w_k1_DIS_s014
                                 + 0.945364 w_k1_FDX_s014
                                 + 1.75443 w_k1_IBM_s014
                                 + 1.2021 w_k1_INTC_s014 + 1.33109 w_k1_KO_s014
                                 + 1.72581 w_k1_MAT_s014
                                 + 1.04677 w_k1_MSFT_s014
                                 + 1.11003 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.920933 w_k1_ADM_s014
                                 + 1.07039 w_k1_AXP_s014
                                 + 1.14968 w_k1_DIS_s014
                                 + 0.963304 w_k1_FDX_s014
                                 + 1.09188 w_k1_IBM_s014
                                 + 0.885269 w_k1_INTC_s014
                                 + 1.19801 w_k1_KO_s014
                                 + 0.995991 w_k1_MAT_s014
                                 + 0.999739 w_k1_MSFT_s014
                                 + 1.08544 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.00082 w_k1_ADM_s014
                                 + 1.06667 w_k1_AXP_s014
                                 + 0.986465 w_k1_DIS_s014
                                 + 1.00718 w_k1_FDX_s014
                                 + 0.707506 w_k1_IBM_s014
                                 + 1.0483 w_k1_INTC_s014
                                 + 0.773598 w_k1_KO_s014
                                 + 0.81158 w_k1_MAT_s014
                                 + 1.09188 w_k1_MSFT_s014
                                 + 0.955671 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.15352 w_k1_ADM_s014
                                 + 1.62757 w_k1_AXP_s014
                                 + 1.79576 w_k1_DIS_s014
                                 + 1.39558 w_k1_FDX_s014
                                 + 1.69709 w_k1_IBM_s014
                                 + 1.23465 w_k1_INTC_s014
                                 + 1.32338 w_k1_KO_s014 + 1.7766 w_k1_MAT_s014
                                 + 1.25962 w_k1_MSFT_s014
                                 + 1.29649 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.982464 w_k1_ADM_s014
                                 + 0.79751 w_k1_AXP_s014
                                 + 0.835376 w_k1_DIS_s014
                                 + 0.748879 w_k1_FDX_s014
                                 + 1.36554 w_k1_IBM_s014
                                 + 0.81528 w_k1_INTC_s014 + 1.03 w_k1_KO_s014
                                 + 1.01189 w_k1_MAT_s014
                                 + 0.838974 w_k1_MSFT_s014
                                 + 0.95731 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.910419 w_k1_ADM_s014
                                 + 1.0584 w_k1_AXP_s014
                                 + 0.757898 w_k1_DIS_s014
                                 + 0.833475 w_k1_FDX_s014
                                 + 1.1066 w_k1_IBM_s014
                                 + 0.718425 w_k1_INTC_s014
                                 + 0.910816 w_k1_KO_s014
                                 + 1.09379 w_k1_MAT_s014
                                 + 0.761269 w_k1_MSFT_s014
                                 + 0.85245 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.758725 w_k1_ADM_s014
                                 + 0.702376 w_k1_AXP_s014
                                 + 0.653908 w_k1_DIS_s014
                                 + 0.729057 w_k1_FDX_s014
                                 + 0.762489 w_k1_IBM_s014
                                 + 0.797167 w_k1_INTC_s014
                                 + 0.746828 w_k1_KO_s014
                                 + 0.674104 w_k1_MAT_s014
                                 + 0.766118 w_k1_MSFT_s014
                                 + 0.865923 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.18935 w_k1_ADM_s014
                                 + 1.44964 w_k1_AXP_s014
                                 + 1.45422 w_k1_DIS_s014
                                 + 1.38325 w_k1_FDX_s014
                                 + 1.03567 w_k1_IBM_s014
                                 + 1.06441 w_k1_INTC_s014
                                 + 1.05306 w_k1_KO_s014 + 1.26106 w_k1_MAT_s014
                                 + 1.26417 w_k1_MSFT_s014
                                 + 1.21516 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.54367 w_k1_ADM_s015
                                 + 0.452429 w_k1_AXP_s015
                                 + 0.57079 w_k1_DIS_s015
                                 + 0.647569 w_k1_FDX_s015
                                 + 0.609597 w_k1_IBM_s015
                                 + 0.85315 w_k1_INTC_s015
                                 + 0.679273 w_k1_KO_s015
                                 + 0.551313 w_k1_MAT_s015
                                 + 0.889449 w_k1_MSFT_s015
                                 + 0.827075 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.92407 w_k1_ADM_s015
                                 + 0.74386 w_k1_AXP_s015
                                 + 0.733857 w_k1_DIS_s015
                                 + 0.715072 w_k1_FDX_s015
                                 + 1.03127 w_k1_IBM_s015
                                 + 0.903166 w_k1_INTC_s015
                                 + 0.90872 w_k1_KO_s015 + 0.97238 w_k1_MAT_s015
                                 + 0.790195 w_k1_MSFT_s015
                                 + 0.90113 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.854874 w_k1_ADM_s015
                                 + 0.660326 w_k1_AXP_s015
                                 + 0.71387 w_k1_DIS_s015
                                 + 0.775406 w_k1_FDX_s015
                                 + 0.856956 w_k1_IBM_s015
                                 + 1.05148 w_k1_INTC_s015
                                 + 0.809641 w_k1_KO_s015
                                 + 0.667647 w_k1_MAT_s015
                                 + 0.842204 w_k1_MSFT_s015
                                 + 0.846911 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.61642 w_k1_ADM_s015
                                 + 1.3738 w_k1_AXP_s015 + 1.61808 w_k1_DIS_s015
                                 + 1.21096 w_k1_FDX_s015
                                 + 1.57973 w_k1_IBM_s015
                                 + 1.3795 w_k1_INTC_s015 + 1.52774 w_k1_KO_s015
                                 + 1.67313 w_k1_MAT_s015
                                 + 1.54284 w_k1_MSFT_s015
                                 + 1.31196 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.21341 w_k1_ADM_s015
                                 + 1.0144 w_k1_AXP_s015 + 1.05839 w_k1_DIS_s015
                                 + 0.937665 w_k1_FDX_s015
                                 + 1.28553 w_k1_IBM_s015
                                 + 1.04362 w_k1_INTC_s015
                                 + 1.08599 w_k1_KO_s015 + 1.10915 w_k1_MAT_s015
                                 + 1.03297 w_k1_MSFT_s015
                                 + 1.06191 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.3163 w_k1_ADM_s015
                                 + 1.00576 w_k1_AXP_s015
                                 + 0.891841 w_k1_DIS_s015
                                 + 1.02554 w_k1_FDX_s015
                                 + 0.92118 w_k1_IBM_s015
                                 + 1.27783 w_k1_INTC_s015
                                 + 0.890321 w_k1_KO_s015
                                 + 0.938133 w_k1_MAT_s015
                                 + 1.03218 w_k1_MSFT_s015
                                 + 0.918352 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.00721 w_k1_ADM_s015
                                 + 0.922653 w_k1_AXP_s015
                                 + 0.809319 w_k1_DIS_s015
                                 + 0.893359 w_k1_FDX_s015
                                 + 0.641479 w_k1_IBM_s015
                                 + 1.16085 w_k1_INTC_s015
                                 + 0.812719 w_k1_KO_s015
                                 + 0.724869 w_k1_MAT_s015
                                 + 0.927049 w_k1_MSFT_s015
                                 + 1.02506 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.37711 w_k1_ADM_s015
                                 + 1.144 w_k1_AXP_s015 + 1.29001 w_k1_DIS_s015
                                 + 1.0132 w_k1_FDX_s015 + 0.96508 w_k1_IBM_s015
                                 + 1.12956 w_k1_INTC_s015
                                 + 1.03834 w_k1_KO_s015 + 1.19323 w_k1_MAT_s015
                                 + 1.22223 w_k1_MSFT_s015
                                 + 1.10397 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.946227 w_k1_ADM_s016
                                 + 1.15949 w_k1_AXP_s016
                                 + 1.03223 w_k1_DIS_s016
                                 + 0.924249 w_k1_FDX_s016
                                 + 1.40869 w_k1_IBM_s016
                                 + 0.928083 w_k1_INTC_s016
                                 + 1.02927 w_k1_KO_s016 + 1.41906 w_k1_MAT_s016
                                 + 0.963025 w_k1_MSFT_s016
                                 + 0.934069 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.966413 w_k1_ADM_s016
                                 + 0.925397 w_k1_AXP_s016
                                 + 0.824901 w_k1_DIS_s016
                                 + 1.02506 w_k1_FDX_s016
                                 + 0.453812 w_k1_IBM_s016
                                 + 0.833945 w_k1_INTC_s016
                                 + 0.58384 w_k1_KO_s016
                                 + 0.492759 w_k1_MAT_s016
                                 + 1.02447 w_k1_MSFT_s016
                                 + 0.790485 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.02006 w_k1_ADM_s016
                                 + 0.904306 w_k1_AXP_s016
                                 + 1.04516 w_k1_DIS_s016
                                 + 1.03813 w_k1_FDX_s016
                                 + 0.816295 w_k1_IBM_s016
                                 + 0.894144 w_k1_INTC_s016
                                 + 0.983008 w_k1_KO_s016
                                 + 0.695485 w_k1_MAT_s016
                                 + 1.1021 w_k1_MSFT_s016
                                 + 0.965426 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.828667 w_k1_ADM_s016
                                 + 0.864516 w_k1_AXP_s016
                                 + 0.674847 w_k1_DIS_s016
                                 + 0.801281 w_k1_FDX_s016
                                 + 0.747719 w_k1_IBM_s016
                                 + 0.756221 w_k1_INTC_s016
                                 + 0.772283 w_k1_KO_s016
                                 + 0.80609 w_k1_MAT_s016
                                 + 0.841482 w_k1_MSFT_s016
                                 + 0.788841 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.704957 w_k1_ADM_s016
                                 + 0.859007 w_k1_AXP_s016
                                 + 0.756166 w_k1_DIS_s016
                                 + 0.807266 w_k1_FDX_s016
                                 + 1.03173 w_k1_IBM_s016
                                 + 0.850676 w_k1_INTC_s016
                                 + 0.824177 w_k1_KO_s016 + 1.2242 w_k1_MAT_s016
                                 + 0.82674 w_k1_MSFT_s016
                                 + 0.819264 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.21049 w_k1_ADM_s016
                                 + 1.29094 w_k1_AXP_s016
                                 + 1.52781 w_k1_DIS_s016
                                 + 1.28812 w_k1_FDX_s016
                                 + 0.953786 w_k1_IBM_s016
                                 + 1.12698 w_k1_INTC_s016
                                 + 1.08985 w_k1_KO_s016
                                 + 0.936431 w_k1_MAT_s016
                                 + 1.32755 w_k1_MSFT_s016
                                 + 0.993381 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.0339 w_k1_ADM_s016
                                 + 1.4871 w_k1_AXP_s016 + 1.56731 w_k1_DIS_s016
                                 + 1.23305 w_k1_FDX_s016
                                 + 1.46138 w_k1_IBM_s016
                                 + 1.38453 w_k1_INTC_s016
                                 + 1.35536 w_k1_KO_s016 + 1.65603 w_k1_MAT_s016
                                 + 1.45898 w_k1_MSFT_s016
                                 + 1.19228 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01752 w_k1_ADM_s016
                                 + 1.18643 w_k1_AXP_s016
                                 + 1.05824 w_k1_DIS_s016
                                 + 0.989837 w_k1_FDX_s016
                                 + 1.05593 w_k1_IBM_s016
                                 + 0.880619 w_k1_INTC_s016
                                 + 0.892987 w_k1_KO_s016
                                 + 1.22016 w_k1_MAT_s016
                                 + 1.01644 w_k1_MSFT_s016
                                 + 0.921638 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.882484 w_k1_ADM_s017
                                 + 0.923681 w_k1_AXP_s017
                                 + 0.864914 w_k1_DIS_s017
                                 + 0.806912 w_k1_FDX_s017
                                 + 1.26678 w_k1_IBM_s017
                                 + 0.86492 w_k1_INTC_s017
                                 + 1.02639 w_k1_KO_s017 + 1.04838 w_k1_MAT_s017
                                 + 0.92053 w_k1_MSFT_s017 + 1.0153 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.27509 w_k1_ADM_s017
                                 + 1.46641 w_k1_AXP_s017 + 1.3723 w_k1_DIS_s017
                                 + 1.21809 w_k1_FDX_s017
                                 + 1.44629 w_k1_IBM_s017
                                 + 1.38114 w_k1_INTC_s017
                                 + 1.42291 w_k1_KO_s017 + 1.63377 w_k1_MAT_s017
                                 + 1.30851 w_k1_MSFT_s017
                                 + 1.19297 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.867814 w_k1_ADM_s017
                                 + 0.778712 w_k1_AXP_s017
                                 + 0.992475 w_k1_DIS_s017
                                 + 0.859448 w_k1_FDX_s017
                                 + 0.825906 w_k1_IBM_s017
                                 + 1.08467 w_k1_INTC_s017
                                 + 0.918199 w_k1_KO_s017
                                 + 0.754674 w_k1_MAT_s017
                                 + 0.965389 w_k1_MSFT_s017
                                 + 0.897356 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.985316 w_k1_ADM_s017
                                 + 0.786493 w_k1_AXP_s017
                                 + 0.577023 w_k1_DIS_s017
                                 + 0.805708 w_k1_FDX_s017
                                 + 0.690815 w_k1_IBM_s017
                                 + 0.721797 w_k1_INTC_s017
                                 + 0.685016 w_k1_KO_s017
                                 + 0.701769 w_k1_MAT_s017
                                 + 0.834194 w_k1_MSFT_s017
                                 + 0.874533 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.769328 w_k1_ADM_s017
                                 + 0.920366 w_k1_AXP_s017
                                 + 0.646354 w_k1_DIS_s017
                                 + 0.79702 w_k1_FDX_s017
                                 + 0.926397 w_k1_IBM_s017
                                 + 0.999898 w_k1_INTC_s017
                                 + 0.853284 w_k1_KO_s017
                                 + 0.741052 w_k1_MAT_s017
                                 + 0.797929 w_k1_MSFT_s017
                                 + 0.948552 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.2055 w_k1_ADM_s017
                                 + 0.864779 w_k1_AXP_s017
                                 + 1.19707 w_k1_DIS_s017
                                 + 1.19192 w_k1_FDX_s017
                                 + 0.80332 w_k1_IBM_s017
                                 + 1.31036 w_k1_INTC_s017
                                 + 1.14103 w_k1_KO_s017
                                 + 0.948533 w_k1_MAT_s017
                                 + 1.16567 w_k1_MSFT_s017
                                 + 1.19303 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.11345 w_k1_ADM_s017
                                 + 1.07639 w_k1_AXP_s017
                                 + 1.03285 w_k1_DIS_s017
                                 + 1.07018 w_k1_FDX_s017
                                 + 1.07033 w_k1_IBM_s017
                                 + 0.919948 w_k1_INTC_s017
                                 + 0.976515 w_k1_KO_s017
                                 + 1.45322 w_k1_MAT_s017
                                 + 1.02905 w_k1_MSFT_s017
                                 + 1.08742 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.895829 w_k1_ADM_s017
                                 + 0.914202 w_k1_AXP_s017
                                 + 0.82163 w_k1_DIS_s017
                                 + 1.00275 w_k1_FDX_s017
                                 + 0.60978 w_k1_IBM_s017
                                 + 1.11052 w_k1_INTC_s017
                                 + 0.938676 w_k1_KO_s017
                                 + 0.801636 w_k1_MAT_s017
                                 + 1.05529 w_k1_MSFT_s017
                                 + 1.05122 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.829373 w_k1_ADM_s100
                                 + 0.73246 w_k1_AXP_s100
                                 + 0.770419 w_k1_DIS_s100
                                 + 0.864503 w_k1_FDX_s100
                                 + 1.05121 w_k1_IBM_s100
                                 + 0.842134 w_k1_INTC_s100
                                 + 0.859232 w_k1_KO_s100
                                 + 0.886789 w_k1_MAT_s100
                                 + 0.788299 w_k1_MSFT_s100
                                 + 0.931368 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.889789 w_k1_ADM_s100
                                 + 1.03548 w_k1_AXP_s100
                                 + 0.905211 w_k1_DIS_s100
                                 + 0.991122 w_k1_FDX_s100
                                 + 1.16868 w_k1_IBM_s100
                                 + 1.04868 w_k1_INTC_s100
                                 + 1.11448 w_k1_KO_s100 + 1.18505 w_k1_MAT_s100
                                 + 0.890536 w_k1_MSFT_s100
                                 + 1.0973 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.01872 w_k1_ADM_s100
                                 + 1.0355 w_k1_AXP_s100 + 1.34384 w_k1_DIS_s100
                                 + 1.1632 w_k1_FDX_s100 + 1.27382 w_k1_IBM_s100
                                 + 0.998983 w_k1_INTC_s100
                                 + 1.15882 w_k1_KO_s100 + 1.36862 w_k1_MAT_s100
                                 + 1.08137 w_k1_MSFT_s100
                                 + 1.09738 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.73857 w_k1_ADM_s100
                                 + 0.632662 w_k1_AXP_s100
                                 + 0.789116 w_k1_DIS_s100
                                 + 0.729606 w_k1_FDX_s100
                                 + 1.24235 w_k1_IBM_s100
                                 + 0.733024 w_k1_INTC_s100
                                 + 0.94874 w_k1_KO_s100
                                 + 0.951913 w_k1_MAT_s100
                                 + 0.762154 w_k1_MSFT_s100
                                 + 0.959433 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.834706 w_k1_ADM_s100
                                 + 0.775928 w_k1_AXP_s100
                                 + 0.86479 w_k1_DIS_s100
                                 + 0.928076 w_k1_FDX_s100
                                 + 0.700399 w_k1_IBM_s100
                                 + 0.852553 w_k1_INTC_s100
                                 + 0.874435 w_k1_KO_s100
                                 + 0.646525 w_k1_MAT_s100
                                 + 0.801511 w_k1_MSFT_s100
                                 + 0.845363 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.26035 w_k1_ADM_s100
                                 + 0.834633 w_k1_AXP_s100
                                 + 1.10487 w_k1_DIS_s100
                                 + 1.14859 w_k1_FDX_s100
                                 + 0.732791 w_k1_IBM_s100
                                 + 1.105 w_k1_INTC_s100 + 1.00025 w_k1_KO_s100
                                 + 0.740021 w_k1_MAT_s100
                                 + 1.12316 w_k1_MSFT_s100
                                 + 1.06489 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.09032 w_k1_ADM_s100
                                 + 1.37111 w_k1_AXP_s100
                                 + 1.44166 w_k1_DIS_s100
                                 + 1.36851 w_k1_FDX_s100
                                 + 1.27897 w_k1_IBM_s100
                                 + 1.21994 w_k1_INTC_s100
                                 + 1.48435 w_k1_KO_s100 + 1.4038 w_k1_MAT_s100
                                 + 1.11935 w_k1_MSFT_s100
                                 + 1.23207 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.96096 w_k1_ADM_s100
                                 + 1.04886 w_k1_AXP_s100
                                 + 0.794938 w_k1_DIS_s100
                                 + 0.890469 w_k1_FDX_s100
                                 + 1.31492 w_k1_IBM_s100
                                 + 0.915746 w_k1_INTC_s100
                                 + 1.01331 w_k1_KO_s100 + 1.76989 w_k1_MAT_s100
                                 + 0.80506 w_k1_MSFT_s100
                                 + 0.983683 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.945496 w_k1_ADM_s101
                                 + 1.05266 w_k1_AXP_s101
                                 + 1.37555 w_k1_DIS_s101
                                 + 1.15522 w_k1_FDX_s101
                                 + 1.59831 w_k1_IBM_s101
                                 + 1.25358 w_k1_INTC_s101
                                 + 1.36475 w_k1_KO_s101 + 1.67688 w_k1_MAT_s101
                                 + 1.11157 w_k1_MSFT_s101
                                 + 1.12976 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.641877 w_k1_ADM_s101
                                 + 0.74761 w_k1_AXP_s101
                                 + 0.576938 w_k1_DIS_s101
                                 + 0.682808 w_k1_FDX_s101
                                 + 0.832463 w_k1_IBM_s101
                                 + 0.725204 w_k1_INTC_s101
                                 + 0.708901 w_k1_KO_s101
                                 + 0.767705 w_k1_MAT_s101
                                 + 0.65991 w_k1_MSFT_s101
                                 + 0.740307 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.835457 w_k1_ADM_s101
                                 + 0.724956 w_k1_AXP_s101
                                 + 0.886497 w_k1_DIS_s101
                                 + 0.927397 w_k1_FDX_s101
                                 + 0.8614 w_k1_IBM_s101
                                 + 0.992651 w_k1_INTC_s101
                                 + 0.975592 w_k1_KO_s101
                                 + 0.795511 w_k1_MAT_s101
                                 + 0.922284 w_k1_MSFT_s101
                                 + 0.865938 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.874003 w_k1_ADM_s101
                                 + 1.18866 w_k1_AXP_s101
                                 + 0.773467 w_k1_DIS_s101
                                 + 1.01428 w_k1_FDX_s101
                                 + 0.810369 w_k1_IBM_s101
                                 + 0.916052 w_k1_INTC_s101
                                 + 0.778543 w_k1_KO_s101
                                 + 0.882572 w_k1_MAT_s101
                                 + 0.909121 w_k1_MSFT_s101
                                 + 0.954557 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.820011 w_k1_ADM_s101
                                 + 0.927342 w_k1_AXP_s101
                                 + 0.899601 w_k1_DIS_s101
                                 + 0.988438 w_k1_FDX_s101
                                 + 1.15868 w_k1_IBM_s101
                                 + 1.15077 w_k1_INTC_s101
                                 + 1.15602 w_k1_KO_s101 + 1.29739 w_k1_MAT_s101
                                 + 1.01357 w_k1_MSFT_s101
                                 + 1.00022 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.12094 w_k1_ADM_s101
                                 + 1.64534 w_k1_AXP_s101 + 1.766 w_k1_DIS_s101
                                 + 1.54355 w_k1_FDX_s101
                                 + 1.02358 w_k1_IBM_s101
                                 + 1.50537 w_k1_INTC_s101
                                 + 1.35996 w_k1_KO_s101 + 1.33983 w_k1_MAT_s101
                                 + 1.14257 w_k1_MSFT_s101
                                 + 1.13391 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.18457 w_k1_ADM_s101
                                 + 1.30383 w_k1_AXP_s101
                                 + 1.36909 w_k1_DIS_s101
                                 + 1.43079 w_k1_FDX_s101
                                 + 0.849199 w_k1_IBM_s101
                                 + 1.50172 w_k1_INTC_s101
                                 + 1.01619 w_k1_KO_s101 + 0.69713 w_k1_MAT_s101
                                 + 1.31224 w_k1_MSFT_s101
                                 + 1.13905 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.10173 w_k1_ADM_s101
                                 + 0.997949 w_k1_AXP_s101
                                 + 1.32055 w_k1_DIS_s101
                                 + 1.22271 w_k1_FDX_s101
                                 + 1.05828 w_k1_IBM_s101
                                 + 1.20113 w_k1_INTC_s101
                                 + 1.05987 w_k1_KO_s101 + 1.14911 w_k1_MAT_s101
                                 + 1.08513 w_k1_MSFT_s101
                                 + 1.00068 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.885141 w_k1_ADM_s102
                                 + 0.823444 w_k1_AXP_s102
                                 + 0.603054 w_k1_DIS_s102
                                 + 0.785479 w_k1_FDX_s102
                                 + 1.07612 w_k1_IBM_s102
                                 + 0.846527 w_k1_INTC_s102
                                 + 0.765646 w_k1_KO_s102
                                 + 0.699949 w_k1_MAT_s102
                                 + 0.710196 w_k1_MSFT_s102
                                 + 0.881804 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.975388 w_k1_ADM_s102
                                 + 1.05489 w_k1_AXP_s102
                                 + 0.990832 w_k1_DIS_s102
                                 + 0.939406 w_k1_FDX_s102
                                 + 1.60337 w_k1_IBM_s102
                                 + 1.07859 w_k1_INTC_s102
                                 + 1.13339 w_k1_KO_s102 + 1.41375 w_k1_MAT_s102
                                 + 1.12294 w_k1_MSFT_s102
                                 + 0.978351 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.11777 w_k1_ADM_s102
                                 + 1.28312 w_k1_AXP_s102
                                 + 0.856452 w_k1_DIS_s102
                                 + 1.06818 w_k1_FDX_s102
                                 + 1.05547 w_k1_IBM_s102
                                 + 0.976177 w_k1_INTC_s102
                                 + 0.900335 w_k1_KO_s102
                                 + 1.59034 w_k1_MAT_s102
                                 + 0.939585 w_k1_MSFT_s102
                                 + 1.03259 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.39351 w_k1_ADM_s102
                                 + 1.15776 w_k1_AXP_s102
                                 + 1.29384 w_k1_DIS_s102
                                 + 1.27829 w_k1_FDX_s102
                                 + 1.02885 w_k1_IBM_s102
                                 + 1.06449 w_k1_INTC_s102
                                 + 1.21419 w_k1_KO_s102
                                 + 0.896618 w_k1_MAT_s102
                                 + 1.23532 w_k1_MSFT_s102
                                 + 1.12954 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.942513 w_k1_ADM_s102
                                 + 0.823142 w_k1_AXP_s102
                                 + 0.829532 w_k1_DIS_s102
                                 + 0.998139 w_k1_FDX_s102
                                 + 0.798232 w_k1_IBM_s102
                                 + 0.897498 w_k1_INTC_s102
                                 + 0.836746 w_k1_KO_s102
                                 + 0.93357 w_k1_MAT_s102
                                 + 0.858759 w_k1_MSFT_s102
                                 + 0.895464 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.861233 w_k1_ADM_s102
                                 + 0.954353 w_k1_AXP_s102
                                 + 1.01299 w_k1_DIS_s102
                                 + 0.962003 w_k1_FDX_s102
                                 + 1.14809 w_k1_IBM_s102
                                 + 0.958625 w_k1_INTC_s102
                                 + 0.995287 w_k1_KO_s102
                                 + 1.19862 w_k1_MAT_s102
                                 + 0.919149 w_k1_MSFT_s102
                                 + 1.02224 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.673835 w_k1_ADM_s102
                                 + 0.695432 w_k1_AXP_s102
                                 + 0.739754 w_k1_DIS_s102
                                 + 0.790978 w_k1_FDX_s102
                                 + 0.774643 w_k1_IBM_s102
                                 + 0.844305 w_k1_INTC_s102
                                 + 0.847593 w_k1_KO_s102
                                 + 0.719853 w_k1_MAT_s102
                                 + 0.762588 w_k1_MSFT_s102
                                 + 0.820662 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.07773 w_k1_ADM_s102
                                 + 0.866682 w_k1_AXP_s102
                                 + 0.950429 w_k1_DIS_s102
                                 + 1.06797 w_k1_FDX_s102
                                 + 0.692109 w_k1_IBM_s102
                                 + 0.842887 w_k1_INTC_s102
                                 + 0.773861 w_k1_KO_s102
                                 + 0.50353 w_k1_MAT_s102
                                 + 0.936944 w_k1_MSFT_s102
                                 + 0.868693 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.32395 w_k1_ADM_s103
                                 + 1.53283 w_k1_AXP_s103 + 1.4684 w_k1_DIS_s103
                                 + 1.29919 w_k1_FDX_s103
                                 + 0.972728 w_k1_IBM_s103
                                 + 0.944383 w_k1_INTC_s103
                                 + 1.15355 w_k1_KO_s103
                                 + 0.978992 w_k1_MAT_s103
                                 + 1.08681 w_k1_MSFT_s103
                                 + 1.04819 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.04362 w_k1_ADM_s103
                                 + 0.910034 w_k1_AXP_s103
                                 + 1.02594 w_k1_DIS_s103
                                 + 0.962485 w_k1_FDX_s103
                                 + 0.923156 w_k1_IBM_s103
                                 + 1.02307 w_k1_INTC_s103
                                 + 0.98524 w_k1_KO_s103
                                 + 0.645042 w_k1_MAT_s103
                                 + 0.887792 w_k1_MSFT_s103
                                 + 0.922888 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.17951 w_k1_ADM_s103
                                 + 1.0997 w_k1_AXP_s103 + 1.01308 w_k1_DIS_s103
                                 + 1.08274 w_k1_FDX_s103
                                 + 0.678236 w_k1_IBM_s103
                                 + 1.0118 w_k1_INTC_s103
                                 + 0.910415 w_k1_KO_s103
                                 + 0.659119 w_k1_MAT_s103
                                 + 1.06212 w_k1_MSFT_s103
                                 + 1.00069 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.733986 w_k1_ADM_s103
                                 + 0.689974 w_k1_AXP_s103
                                 + 0.782579 w_k1_DIS_s103
                                 + 0.768978 w_k1_FDX_s103
                                 + 0.80706 w_k1_IBM_s103
                                 + 0.916329 w_k1_INTC_s103
                                 + 0.816731 w_k1_KO_s103
                                 + 0.686523 w_k1_MAT_s103
                                 + 0.835278 w_k1_MSFT_s103
                                 + 0.801284 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.894202 w_k1_ADM_s103
                                 + 0.751017 w_k1_AXP_s103
                                 + 0.872946 w_k1_DIS_s103
                                 + 0.754448 w_k1_FDX_s103
                                 + 1.35879 w_k1_IBM_s103
                                 + 0.694056 w_k1_INTC_s103
                                 + 1.14914 w_k1_KO_s103 + 1.01604 w_k1_MAT_s103
                                 + 0.787591 w_k1_MSFT_s103
                                 + 0.870192 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.810291 w_k1_ADM_s103
                                 + 1.01951 w_k1_AXP_s103
                                 + 0.977114 w_k1_DIS_s103
                                 + 0.94186 w_k1_FDX_s103
                                 + 1.23121 w_k1_IBM_s103
                                 + 0.701397 w_k1_INTC_s103
                                 + 0.986337 w_k1_KO_s103
                                 + 1.12306 w_k1_MAT_s103
                                 + 0.887064 w_k1_MSFT_s103
                                 + 0.952445 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.07952 w_k1_ADM_s103
                                 + 1.22694 w_k1_AXP_s103
                                 + 1.54369 w_k1_DIS_s103
                                 + 1.14372 w_k1_FDX_s103 + 1.3197 w_k1_IBM_s103
                                 + 0.977983 w_k1_INTC_s103
                                 + 1.42455 w_k1_KO_s103 + 1.44817 w_k1_MAT_s103
                                 + 1.00404 w_k1_MSFT_s103 + 1.1292 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.833932 w_k1_ADM_s103
                                 + 1.07638 w_k1_AXP_s103
                                 + 0.780831 w_k1_DIS_s103
                                 + 0.897079 w_k1_FDX_s103
                                 + 1.11919 w_k1_IBM_s103
                                 + 0.841539 w_k1_INTC_s103
                                 + 1.03279 w_k1_KO_s103 + 1.10995 w_k1_MAT_s103
                                 + 0.797143 w_k1_MSFT_s103
                                 + 0.889614 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.30587 w_k1_ADM_s104
                                 + 1.35018 w_k1_AXP_s104
                                 + 1.57846 w_k1_DIS_s104
                                 + 1.45468 w_k1_FDX_s104
                                 + 1.14606 w_k1_IBM_s104
                                 + 1.09099 w_k1_INTC_s104
                                 + 1.40657 w_k1_KO_s104 + 1.33583 w_k1_MAT_s104
                                 + 1.31186 w_k1_MSFT_s104
                                 + 1.14103 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.955667 w_k1_ADM_s104
                                 + 0.786227 w_k1_AXP_s104
                                 + 0.660153 w_k1_DIS_s104
                                 + 0.848249 w_k1_FDX_s104
                                 + 0.704362 w_k1_IBM_s104
                                 + 0.745205 w_k1_INTC_s104
                                 + 0.741428 w_k1_KO_s104
                                 + 0.799667 w_k1_MAT_s104
                                 + 0.889758 w_k1_MSFT_s104
                                 + 0.867044 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.26279 w_k1_ADM_s104
                                 + 1.04079 w_k1_AXP_s104
                                 + 1.08914 w_k1_DIS_s104
                                 + 1.17103 w_k1_FDX_s104
                                 + 0.777777 w_k1_IBM_s104
                                 + 0.911746 w_k1_INTC_s104
                                 + 1.00874 w_k1_KO_s104
                                 + 0.805144 w_k1_MAT_s104
                                 + 1.10552 w_k1_MSFT_s104
                                 + 0.972925 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.91487 w_k1_ADM_s104
                                 + 0.839812 w_k1_AXP_s104
                                 + 0.862835 w_k1_DIS_s104
                                 + 0.952438 w_k1_FDX_s104
                                 + 0.756884 w_k1_IBM_s104
                                 + 0.883015 w_k1_INTC_s104
                                 + 0.897527 w_k1_KO_s104
                                 + 0.876449 w_k1_MAT_s104
                                 + 0.904522 w_k1_MSFT_s104
                                 + 0.933109 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.616704 w_k1_ADM_s104
                                 + 0.644572 w_k1_AXP_s104
                                 + 0.582729 w_k1_DIS_s104
                                 + 0.73783 w_k1_FDX_s104
                                 + 0.740813 w_k1_IBM_s104
                                 + 0.682752 w_k1_INTC_s104
                                 + 0.730158 w_k1_KO_s104
                                 + 0.72687 w_k1_MAT_s104
                                 + 0.729145 w_k1_MSFT_s104
                                 + 0.691193 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.0525 w_k1_ADM_s104
                                 + 0.970393 w_k1_AXP_s104
                                 + 0.924027 w_k1_DIS_s104
                                 + 0.884012 w_k1_FDX_s104
                                 + 1.11107 w_k1_IBM_s104
                                 + 0.862194 w_k1_INTC_s104
                                 + 1.05386 w_k1_KO_s104 + 1.28475 w_k1_MAT_s104
                                 + 0.982374 w_k1_MSFT_s104
                                 + 0.845842 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.15083 w_k1_ADM_s104
                                 + 1.60199 w_k1_AXP_s104
                                 + 1.91747 w_k1_DIS_s104
                                 + 1.29974 w_k1_FDX_s104
                                 + 1.90491 w_k1_IBM_s104
                                 + 1.38247 w_k1_INTC_s104
                                 + 1.58928 w_k1_KO_s104 + 1.50736 w_k1_MAT_s104
                                 + 1.38971 w_k1_MSFT_s104
                                 + 1.14793 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.24457 w_k1_ADM_s104
                                 + 1.19042 w_k1_AXP_s104
                                 + 1.37644 w_k1_DIS_s104
                                 + 1.11138 w_k1_FDX_s104
                                 + 1.37488 w_k1_IBM_s104
                                 + 1.13422 w_k1_INTC_s104
                                 + 1.39367 w_k1_KO_s104 + 1.15185 w_k1_MAT_s104
                                 + 1.29046 w_k1_MSFT_s104
                                 + 1.05492 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.665084 w_k1_ADM_s105
                                 + 0.767776 w_k1_AXP_s105
                                 + 0.786488 w_k1_DIS_s105
                                 + 0.826234 w_k1_FDX_s105
                                 + 0.869579 w_k1_IBM_s105
                                 + 0.841022 w_k1_INTC_s105
                                 + 0.741642 w_k1_KO_s105
                                 + 0.757035 w_k1_MAT_s105
                                 + 0.816134 w_k1_MSFT_s105
                                 + 0.823128 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.98523 w_k1_ADM_s105
                                 + 0.973932 w_k1_AXP_s105
                                 + 0.961189 w_k1_DIS_s105
                                 + 1.07581 w_k1_FDX_s105
                                 + 0.828371 w_k1_IBM_s105
                                 + 1.04466 w_k1_INTC_s105
                                 + 0.785341 w_k1_KO_s105
                                 + 0.879641 w_k1_MAT_s105
                                 + 0.884788 w_k1_MSFT_s105
                                 + 0.986472 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.13987 w_k1_ADM_s105
                                 + 0.930679 w_k1_AXP_s105
                                 + 0.750414 w_k1_DIS_s105
                                 + 0.848073 w_k1_FDX_s105
                                 + 0.843459 w_k1_IBM_s105
                                 + 0.779352 w_k1_INTC_s105
                                 + 0.700319 w_k1_KO_s105
                                 + 0.876148 w_k1_MAT_s105
                                 + 0.812926 w_k1_MSFT_s105
                                 + 0.9194 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.11806 w_k1_ADM_s105
                                 + 1.4452 w_k1_AXP_s105 + 1.23131 w_k1_DIS_s105
                                 + 1.06392 w_k1_FDX_s105
                                 + 1.43357 w_k1_IBM_s105
                                 + 1.03176 w_k1_INTC_s105
                                 + 1.13027 w_k1_KO_s105 + 1.61609 w_k1_MAT_s105
                                 + 0.994233 w_k1_MSFT_s105
                                 + 1.0894 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.09012 w_k1_ADM_s105
                                 + 0.839781 w_k1_AXP_s105
                                 + 0.977843 w_k1_DIS_s105
                                 + 0.860538 w_k1_FDX_s105 + 1.597 w_k1_IBM_s105
                                 + 1.07539 w_k1_INTC_s105
                                 + 1.18666 w_k1_KO_s105 + 1.47907 w_k1_MAT_s105
                                 + 0.921122 w_k1_MSFT_s105
                                 + 1.06304 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.21435 w_k1_ADM_s105
                                 + 1.48113 w_k1_AXP_s105
                                 + 1.43152 w_k1_DIS_s105
                                 + 1.44405 w_k1_FDX_s105
                                 + 0.916846 w_k1_IBM_s105
                                 + 1.16758 w_k1_INTC_s105
                                 + 1.07834 w_k1_KO_s105 + 1.22079 w_k1_MAT_s105
                                 + 1.15949 w_k1_MSFT_s105
                                 + 1.17486 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.724717 w_k1_ADM_s105
                                 + 0.752178 w_k1_AXP_s105
                                 + 0.553566 w_k1_DIS_s105
                                 + 0.708551 w_k1_FDX_s105
                                 + 0.967169 w_k1_IBM_s105
                                 + 0.706626 w_k1_INTC_s105
                                 + 0.692035 w_k1_KO_s105
                                 + 0.997524 w_k1_MAT_s105
                                 + 0.637096 w_k1_MSFT_s105
                                 + 0.824708 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.899952 w_k1_ADM_s105
                                 + 0.902272 w_k1_AXP_s105
                                 + 1.01386 w_k1_DIS_s105
                                 + 0.92551 w_k1_FDX_s105
                                 + 1.17265 w_k1_IBM_s105
                                 + 0.875178 w_k1_INTC_s105
                                 + 0.989574 w_k1_KO_s105
                                 + 1.10735 w_k1_MAT_s105
                                 + 0.919227 w_k1_MSFT_s105
                                 + 0.99352 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.00014 w_k1_ADM_s106
                                 + 0.73154 w_k1_AXP_s106
                                 + 0.852573 w_k1_DIS_s106
                                 + 0.949008 w_k1_FDX_s106
                                 + 0.608083 w_k1_IBM_s106
                                 + 0.719606 w_k1_INTC_s106
                                 + 0.59788 w_k1_KO_s106
                                 + 0.454127 w_k1_MAT_s106
                                 + 0.821831 w_k1_MSFT_s106
                                 + 0.813256 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.51151 w_k1_ADM_s106
                                 + 1.46423 w_k1_AXP_s106
                                 + 1.55381 w_k1_DIS_s106
                                 + 1.30477 w_k1_FDX_s106
                                 + 1.08064 w_k1_IBM_s106
                                 + 1.34326 w_k1_INTC_s106
                                 + 1.18079 w_k1_KO_s106
                                 + 0.818023 w_k1_MAT_s106
                                 + 1.16029 w_k1_MSFT_s106
                                 + 1.10929 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.93443 w_k1_ADM_s106
                                 + 0.923152 w_k1_AXP_s106
                                 + 0.827463 w_k1_DIS_s106
                                 + 0.877102 w_k1_FDX_s106
                                 + 0.978078 w_k1_IBM_s106
                                 + 0.845584 w_k1_INTC_s106
                                 + 0.759185 w_k1_KO_s106
                                 + 0.905057 w_k1_MAT_s106
                                 + 0.814852 w_k1_MSFT_s106
                                 + 0.932926 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.905666 w_k1_ADM_s106
                                 + 0.703716 w_k1_AXP_s106
                                 + 0.693208 w_k1_DIS_s106
                                 + 0.751298 w_k1_FDX_s106
                                 + 0.789373 w_k1_IBM_s106
                                 + 0.788801 w_k1_INTC_s106
                                 + 0.726797 w_k1_KO_s106
                                 + 0.443883 w_k1_MAT_s106
                                 + 0.737362 w_k1_MSFT_s106
                                 + 0.761962 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.14511 w_k1_ADM_s106
                                 + 0.807955 w_k1_AXP_s106
                                 + 1.11636 w_k1_DIS_s106
                                 + 1.03365 w_k1_FDX_s106
                                 + 0.868623 w_k1_IBM_s106
                                 + 1.00806 w_k1_INTC_s106
                                 + 0.89624 w_k1_KO_s106 + 0.77865 w_k1_MAT_s106
                                 + 1.0204 w_k1_MSFT_s106
                                 + 0.930124 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.02426 w_k1_ADM_s106
                                 + 0.9714 w_k1_AXP_s106 + 1.10779 w_k1_DIS_s106
                                 + 1.11983 w_k1_FDX_s106
                                 + 0.728083 w_k1_IBM_s106
                                 + 0.911 w_k1_INTC_s106 + 0.87805 w_k1_KO_s106
                                 + 0.731552 w_k1_MAT_s106
                                 + 0.949809 w_k1_MSFT_s106
                                 + 1.06959 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.41861 w_k1_ADM_s106
                                 + 1.03518 w_k1_AXP_s106
                                 + 1.07397 w_k1_DIS_s106
                                 + 1.23118 w_k1_FDX_s106
                                 + 0.536584 w_k1_IBM_s106
                                 + 1.0861 w_k1_INTC_s106
                                 + 0.849971 w_k1_KO_s106
                                 + 0.582192 w_k1_MAT_s106
                                 + 0.974154 w_k1_MSFT_s106
                                 + 0.97641 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.15677 w_k1_ADM_s106
                                 + 1.00647 w_k1_AXP_s106
                                 + 1.17547 w_k1_DIS_s106
                                 + 1.04408 w_k1_FDX_s106
                                 + 1.49188 w_k1_IBM_s106
                                 + 0.96095 w_k1_INTC_s106
                                 + 1.22857 w_k1_KO_s106 + 1.1312 w_k1_MAT_s106
                                 + 0.961448 w_k1_MSFT_s106
                                 + 0.968916 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.823361 w_k1_ADM_s107
                                 + 0.623527 w_k1_AXP_s107
                                 + 0.629434 w_k1_DIS_s107
                                 + 0.689631 w_k1_FDX_s107
                                 + 0.852081 w_k1_IBM_s107
                                 + 0.774034 w_k1_INTC_s107
                                 + 0.864096 w_k1_KO_s107
                                 + 1.06168 w_k1_MAT_s107
                                 + 0.826384 w_k1_MSFT_s107
                                 + 0.721445 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.937642 w_k1_ADM_s107
                                 + 1.21088 w_k1_AXP_s107
                                 + 1.14562 w_k1_DIS_s107 + 1.027 w_k1_FDX_s107
                                 + 1.40184 w_k1_IBM_s107
                                 + 1.11999 w_k1_INTC_s107
                                 + 1.16753 w_k1_KO_s107 + 2.10534 w_k1_MAT_s107
                                 + 1.0106 w_k1_MSFT_s107
                                 + 0.962896 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.17151 w_k1_ADM_s107
                                 + 1.41589 w_k1_AXP_s107
                                 + 1.44695 w_k1_DIS_s107
                                 + 1.31232 w_k1_FDX_s107
                                 + 1.23049 w_k1_IBM_s107
                                 + 1.06459 w_k1_INTC_s107
                                 + 1.09363 w_k1_KO_s107 + 1.78632 w_k1_MAT_s107
                                 + 1.2144 w_k1_MSFT_s107 + 1.09654 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.04894 w_k1_ADM_s107
                                 + 1.1242 w_k1_AXP_s107 + 1.31943 w_k1_DIS_s107
                                 + 1.21815 w_k1_FDX_s107
                                 + 1.06299 w_k1_IBM_s107
                                 + 0.942747 w_k1_INTC_s107
                                 + 1.11232 w_k1_KO_s107 + 1.35593 w_k1_MAT_s107
                                 + 1.0488 w_k1_MSFT_s107
                                 + 0.991852 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.895563 w_k1_ADM_s107
                                 + 0.947243 w_k1_AXP_s107
                                 + 1.24242 w_k1_DIS_s107
                                 + 1.20588 w_k1_FDX_s107
                                 + 0.824981 w_k1_IBM_s107
                                 + 1.02689 w_k1_INTC_s107
                                 + 1.01528 w_k1_KO_s107 + 1.26091 w_k1_MAT_s107
                                 + 1.0837 w_k1_MSFT_s107 + 1.00362 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.850414 w_k1_ADM_s107
                                 + 0.86795 w_k1_AXP_s107
                                 + 0.74487 w_k1_DIS_s107
                                 + 0.872185 w_k1_FDX_s107
                                 + 0.725791 w_k1_IBM_s107
                                 + 0.839379 w_k1_INTC_s107
                                 + 0.822526 w_k1_KO_s107
                                 + 0.686107 w_k1_MAT_s107
                                 + 0.80537 w_k1_MSFT_s107
                                 + 0.793524 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.00959 w_k1_ADM_s107
                                 + 1.25489 w_k1_AXP_s107
                                 + 0.945061 w_k1_DIS_s107
                                 + 1.01547 w_k1_FDX_s107
                                 + 1.00332 w_k1_IBM_s107
                                 + 0.857073 w_k1_INTC_s107
                                 + 1.0655 w_k1_KO_s107 + 1.45745 w_k1_MAT_s107
                                 + 0.920088 w_k1_MSFT_s107
                                 + 0.967067 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.696759 w_k1_ADM_s107
                                 + 0.786407 w_k1_AXP_s107
                                 + 0.826771 w_k1_DIS_s107
                                 + 0.86542 w_k1_FDX_s107
                                 + 1.07211 w_k1_IBM_s107
                                 + 0.776455 w_k1_INTC_s107
                                 + 0.934222 w_k1_KO_s107
                                 + 1.33862 w_k1_MAT_s107
                                 + 0.826722 w_k1_MSFT_s107
                                 + 0.881314 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.789826 w_k1_ADM_s110
                                 + 0.867915 w_k1_AXP_s110
                                 + 0.752099 w_k1_DIS_s110
                                 + 0.969651 w_k1_FDX_s110
                                 + 0.658276 w_k1_IBM_s110
                                 + 1.05291 w_k1_INTC_s110
                                 + 0.803791 w_k1_KO_s110
                                 + 0.546978 w_k1_MAT_s110
                                 + 0.911682 w_k1_MSFT_s110
                                 + 0.898697 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.796242 w_k1_ADM_s110
                                 + 0.995375 w_k1_AXP_s110
                                 + 1.07777 w_k1_DIS_s110 + 0.9288 w_k1_FDX_s110
                                 + 1.40718 w_k1_IBM_s110
                                 + 0.87173 w_k1_INTC_s110
                                 + 1.19272 w_k1_KO_s110 + 1.4031 w_k1_MAT_s110
                                 + 1.0445 w_k1_MSFT_s110 + 1.06296 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.01729 w_k1_ADM_s110
                                 + 0.96671 w_k1_AXP_s110 + 1.1561 w_k1_DIS_s110
                                 + 1.14389 w_k1_FDX_s110
                                 + 0.799493 w_k1_IBM_s110
                                 + 1.13821 w_k1_INTC_s110
                                 + 1.04161 w_k1_KO_s110
                                 + 0.716273 w_k1_MAT_s110
                                 + 1.07392 w_k1_MSFT_s110
                                 + 0.957482 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.11431 w_k1_ADM_s110
                                 + 1.31351 w_k1_AXP_s110 + 1.3952 w_k1_DIS_s110
                                 + 1.38659 w_k1_FDX_s110
                                 + 0.805307 w_k1_IBM_s110
                                 + 1.21971 w_k1_INTC_s110
                                 + 1.01955 w_k1_KO_s110
                                 + 0.763537 w_k1_MAT_s110
                                 + 1.27173 w_k1_MSFT_s110
                                 + 1.18941 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.31572 w_k1_ADM_s110
                                 + 1.56446 w_k1_AXP_s110
                                 + 1.66966 w_k1_DIS_s110
                                 + 1.37859 w_k1_FDX_s110
                                 + 1.27029 w_k1_IBM_s110
                                 + 1.34631 w_k1_INTC_s110
                                 + 1.32464 w_k1_KO_s110
                                 + 0.997918 w_k1_MAT_s110
                                 + 1.51588 w_k1_MSFT_s110
                                 + 1.15145 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.896932 w_k1_ADM_s110
                                 + 1.00368 w_k1_AXP_s110
                                 + 0.903478 w_k1_DIS_s110
                                 + 0.968714 w_k1_FDX_s110
                                 + 0.862516 w_k1_IBM_s110
                                 + 1.03216 w_k1_INTC_s110
                                 + 0.947629 w_k1_KO_s110
                                 + 0.961511 w_k1_MAT_s110
                                 + 0.973535 w_k1_MSFT_s110
                                 + 0.96135 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.753624 w_k1_ADM_s110
                                 + 0.790285 w_k1_AXP_s110
                                 + 0.736424 w_k1_DIS_s110
                                 + 0.839017 w_k1_FDX_s110
                                 + 0.937829 w_k1_IBM_s110
                                 + 0.846959 w_k1_INTC_s110
                                 + 0.946896 w_k1_KO_s110
                                 + 0.761186 w_k1_MAT_s110
                                 + 0.907016 w_k1_MSFT_s110
                                 + 0.898251 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.708468 w_k1_ADM_s110
                                 + 0.613093 w_k1_AXP_s110
                                 + 0.584712 w_k1_DIS_s110
                                 + 0.669521 w_k1_FDX_s110
                                 + 0.820306 w_k1_IBM_s110
                                 + 0.676372 w_k1_INTC_s110
                                 + 0.712322 w_k1_KO_s110
                                 + 0.606571 w_k1_MAT_s110
                                 + 0.737822 w_k1_MSFT_s110
                                 + 0.81672 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.879672 w_k1_ADM_s111
                                 + 0.92096 w_k1_AXP_s111
                                 + 0.786489 w_k1_DIS_s111
                                 + 0.845165 w_k1_FDX_s111
                                 + 1.15224 w_k1_IBM_s111
                                 + 0.751605 w_k1_INTC_s111
                                 + 0.84786 w_k1_KO_s111 + 1.26211 w_k1_MAT_s111
                                 + 0.904954 w_k1_MSFT_s111
                                 + 0.908063 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.761043 w_k1_ADM_s111
                                 + 0.576392 w_k1_AXP_s111
                                 + 0.631695 w_k1_DIS_s111
                                 + 0.696075 w_k1_FDX_s111
                                 + 0.793781 w_k1_IBM_s111
                                 + 0.714609 w_k1_INTC_s111
                                 + 0.731507 w_k1_KO_s111
                                 + 0.548891 w_k1_MAT_s111
                                 + 0.756136 w_k1_MSFT_s111
                                 + 0.843805 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.17734 w_k1_ADM_s111
                                 + 1.27207 w_k1_AXP_s111
                                 + 1.38569 w_k1_DIS_s111
                                 + 1.29579 w_k1_FDX_s111
                                 + 0.867133 w_k1_IBM_s111
                                 + 1.06136 w_k1_INTC_s111
                                 + 0.938425 w_k1_KO_s111
                                 + 1.12592 w_k1_MAT_s111
                                 + 1.15227 w_k1_MSFT_s111
                                 + 1.05736 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.876732 w_k1_ADM_s111
                                 + 0.875027 w_k1_AXP_s111
                                 + 0.731673 w_k1_DIS_s111
                                 + 0.767509 w_k1_FDX_s111
                                 + 1.0235 w_k1_IBM_s111
                                 + 0.770501 w_k1_INTC_s111
                                 + 0.833569 w_k1_KO_s111
                                 + 1.28456 w_k1_MAT_s111
                                 + 0.845641 w_k1_MSFT_s111
                                 + 0.981854 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02102 w_k1_ADM_s111
                                 + 0.899594 w_k1_AXP_s111
                                 + 1.18129 w_k1_DIS_s111
                                 + 1.04897 w_k1_FDX_s111
                                 + 1.00683 w_k1_IBM_s111
                                 + 1.11298 w_k1_INTC_s111
                                 + 1.13999 w_k1_KO_s111
                                 + 0.915123 w_k1_MAT_s111
                                 + 1.11745 w_k1_MSFT_s111
                                 + 1.08893 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.697586 w_k1_ADM_s111
                                 + 0.636369 w_k1_AXP_s111
                                 + 0.833104 w_k1_DIS_s111
                                 + 0.816634 w_k1_FDX_s111
                                 + 0.842957 w_k1_IBM_s111
                                 + 0.801564 w_k1_INTC_s111
                                 + 0.799062 w_k1_KO_s111
                                 + 0.644053 w_k1_MAT_s111
                                 + 0.891255 w_k1_MSFT_s111
                                 + 0.869668 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.966846 w_k1_ADM_s111
                                 + 1.09828 w_k1_AXP_s111
                                 + 1.32867 w_k1_DIS_s111
                                 + 1.00203 w_k1_FDX_s111
                                 + 1.55802 w_k1_IBM_s111
                                 + 1.08611 w_k1_INTC_s111
                                 + 1.09816 w_k1_KO_s111 + 1.25157 w_k1_MAT_s111
                                 + 1.07538 w_k1_MSFT_s111
                                 + 1.04636 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.03193 w_k1_ADM_s111
                                 + 0.773258 w_k1_AXP_s111
                                 + 0.847531 w_k1_DIS_s111
                                 + 0.962547 w_k1_FDX_s111
                                 + 0.614885 w_k1_IBM_s111
                                 + 0.958695 w_k1_INTC_s111
                                 + 0.705633 w_k1_KO_s111
                                 + 0.602602 w_k1_MAT_s111
                                 + 0.939856 w_k1_MSFT_s111
                                 + 0.948741 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.20017 w_k1_ADM_s112
                                 + 1.19876 w_k1_AXP_s112
                                 + 1.22009 w_k1_DIS_s112
                                 + 1.19576 w_k1_FDX_s112
                                 + 0.825901 w_k1_IBM_s112
                                 + 1.08439 w_k1_INTC_s112
                                 + 0.993873 w_k1_KO_s112
                                 + 1.06099 w_k1_MAT_s112
                                 + 1.22998 w_k1_MSFT_s112
                                 + 1.07939 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.872095 w_k1_ADM_s112
                                 + 1.22263 w_k1_AXP_s112
                                 + 1.21513 w_k1_DIS_s112 + 1.0196 w_k1_FDX_s112
                                 + 1.70197 w_k1_IBM_s112
                                 + 1.24689 w_k1_INTC_s112
                                 + 1.24722 w_k1_KO_s112 + 1.67356 w_k1_MAT_s112
                                 + 1.21981 w_k1_MSFT_s112
                                 + 1.12638 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.846277 w_k1_ADM_s112
                                 + 0.748478 w_k1_AXP_s112
                                 + 0.611967 w_k1_DIS_s112
                                 + 0.727105 w_k1_FDX_s112
                                 + 0.694316 w_k1_IBM_s112
                                 + 0.763863 w_k1_INTC_s112
                                 + 0.742132 w_k1_KO_s112
                                 + 0.63216 w_k1_MAT_s112
                                 + 0.794806 w_k1_MSFT_s112
                                 + 0.806748 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.916034 w_k1_ADM_s112
                                 + 0.875897 w_k1_AXP_s112
                                 + 0.861841 w_k1_DIS_s112
                                 + 0.861592 w_k1_FDX_s112
                                 + 0.826758 w_k1_IBM_s112
                                 + 0.935011 w_k1_INTC_s112
                                 + 0.870564 w_k1_KO_s112
                                 + 0.807463 w_k1_MAT_s112
                                 + 0.897749 w_k1_MSFT_s112
                                 + 0.951006 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.921284 w_k1_ADM_s112
                                 + 1.21143 w_k1_AXP_s112
                                 + 0.984399 w_k1_DIS_s112
                                 + 1.07178 w_k1_FDX_s112
                                 + 0.740098 w_k1_IBM_s112
                                 + 0.933194 w_k1_INTC_s112
                                 + 0.842634 w_k1_KO_s112
                                 + 1.00444 w_k1_MAT_s112
                                 + 0.996415 w_k1_MSFT_s112
                                 + 0.99831 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.841498 w_k1_ADM_s112
                                 + 0.81452 w_k1_AXP_s112
                                 + 0.874555 w_k1_DIS_s112
                                 + 0.770835 w_k1_FDX_s112
                                 + 1.1854 w_k1_IBM_s112
                                 + 0.90889 w_k1_INTC_s112
                                 + 1.07476 w_k1_KO_s112 + 1.19 w_k1_MAT_s112
                                 + 0.965156 w_k1_MSFT_s112
                                 + 1.07638 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.899901 w_k1_ADM_s112
                                 + 0.921523 w_k1_AXP_s112
                                 + 0.811457 w_k1_DIS_s112
                                 + 0.815438 w_k1_FDX_s112
                                 + 0.982718 w_k1_IBM_s112
                                 + 0.957573 w_k1_INTC_s112
                                 + 0.849571 w_k1_KO_s112
                                 + 0.986994 w_k1_MAT_s112
                                 + 1.0159 w_k1_MSFT_s112 + 0.96671 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.2005 w_k1_ADM_s112
                                 + 1.43981 w_k1_AXP_s112
                                 + 1.54682 w_k1_DIS_s112
                                 + 1.22221 w_k1_FDX_s112
                                 + 1.16878 w_k1_IBM_s112
                                 + 1.08385 w_k1_INTC_s112 + 1.2391 w_k1_KO_s112
                                 + 1.13881 w_k1_MAT_s112
                                 + 1.26608 w_k1_MSFT_s112
                                 + 1.13453 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.887434 w_k1_ADM_s113
                                 + 0.741612 w_k1_AXP_s113
                                 + 0.841936 w_k1_DIS_s113
                                 + 0.920003 w_k1_FDX_s113
                                 + 0.742539 w_k1_IBM_s113
                                 + 0.839923 w_k1_INTC_s113
                                 + 0.684704 w_k1_KO_s113
                                 + 0.508195 w_k1_MAT_s113
                                 + 0.747799 w_k1_MSFT_s113
                                 + 0.799152 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.22165 w_k1_ADM_s113
                                 + 1.0269 w_k1_AXP_s113 + 1.31664 w_k1_DIS_s113
                                 + 1.15316 w_k1_FDX_s113
                                 + 0.890478 w_k1_IBM_s113
                                 + 0.931531 w_k1_INTC_s113
                                 + 1.0146 w_k1_KO_s113 + 0.811598 w_k1_MAT_s113
                                 + 1.08125 w_k1_MSFT_s113
                                 + 1.03276 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.909557 w_k1_ADM_s113
                                 + 0.953477 w_k1_AXP_s113
                                 + 1.11043 w_k1_DIS_s113
                                 + 0.975281 w_k1_FDX_s113
                                 + 1.18077 w_k1_IBM_s113
                                 + 1.21017 w_k1_INTC_s113
                                 + 1.05928 w_k1_KO_s113
                                 + 0.865389 w_k1_MAT_s113
                                 + 0.89218 w_k1_MSFT_s113
                                 + 0.918882 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.928445 w_k1_ADM_s113
                                 + 1.18095 w_k1_AXP_s113
                                 + 1.07895 w_k1_DIS_s113 + 1.1146 w_k1_FDX_s113
                                 + 0.917658 w_k1_IBM_s113
                                 + 1.1907 w_k1_INTC_s113
                                 + 0.913205 w_k1_KO_s113
                                 + 0.902004 w_k1_MAT_s113
                                 + 0.939643 w_k1_MSFT_s113
                                 + 0.88959 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.01124 w_k1_ADM_s113
                                 + 1.2279 w_k1_AXP_s113 + 1.17553 w_k1_DIS_s113
                                 + 1.09818 w_k1_FDX_s113
                                 + 1.16349 w_k1_IBM_s113
                                 + 1.10364 w_k1_INTC_s113
                                 + 1.05117 w_k1_KO_s113 + 1.01683 w_k1_MAT_s113
                                 + 1.15257 w_k1_MSFT_s113
                                 + 1.02503 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.897222 w_k1_ADM_s113
                                 + 1.16862 w_k1_AXP_s113
                                 + 1.26375 w_k1_DIS_s113
                                 + 0.963491 w_k1_FDX_s113
                                 + 1.54786 w_k1_IBM_s113
                                 + 1.0865 w_k1_INTC_s113 + 1.19292 w_k1_KO_s113
                                 + 1.62569 w_k1_MAT_s113
                                 + 1.0951 w_k1_MSFT_s113 + 1.05904 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.09095 w_k1_ADM_s113
                                 + 1.29163 w_k1_AXP_s113
                                 + 2.07756 w_k1_DIS_s113
                                 + 1.29508 w_k1_FDX_s113
                                 + 1.32425 w_k1_IBM_s113
                                 + 1.33091 w_k1_INTC_s113
                                 + 1.44749 w_k1_KO_s113 + 1.15124 w_k1_MAT_s113
                                 + 1.42554 w_k1_MSFT_s113
                                 + 1.21833 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.933586 w_k1_ADM_s113
                                 + 0.996574 w_k1_AXP_s113
                                 + 0.775387 w_k1_DIS_s113
                                 + 0.833692 w_k1_FDX_s113
                                 + 1.0811 w_k1_IBM_s113
                                 + 1.06298 w_k1_INTC_s113
                                 + 0.950749 w_k1_KO_s113
                                 + 1.02797 w_k1_MAT_s113
                                 + 0.788008 w_k1_MSFT_s113
                                 + 0.913367 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.0352 w_k1_ADM_s114
                                 + 0.99012 w_k1_AXP_s114
                                 + 0.966001 w_k1_DIS_s114
                                 + 1.02899 w_k1_FDX_s114
                                 + 0.713927 w_k1_IBM_s114
                                 + 1.15683 w_k1_INTC_s114
                                 + 0.722392 w_k1_KO_s114
                                 + 0.437838 w_k1_MAT_s114
                                 + 1.00334 w_k1_MSFT_s114
                                 + 0.783306 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.726915 w_k1_ADM_s114
                                 + 0.79736 w_k1_AXP_s114
                                 + 0.724208 w_k1_DIS_s114
                                 + 0.825989 w_k1_FDX_s114
                                 + 0.780228 w_k1_IBM_s114
                                 + 0.689117 w_k1_INTC_s114
                                 + 0.777853 w_k1_KO_s114
                                 + 1.00208 w_k1_MAT_s114
                                 + 0.779012 w_k1_MSFT_s114
                                 + 0.847734 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.80918 w_k1_ADM_s114
                                 + 0.852008 w_k1_AXP_s114
                                 + 0.758967 w_k1_DIS_s114
                                 + 0.794141 w_k1_FDX_s114
                                 + 1.02131 w_k1_IBM_s114
                                 + 0.778492 w_k1_INTC_s114
                                 + 0.810983 w_k1_KO_s114
                                 + 0.916356 w_k1_MAT_s114
                                 + 0.779297 w_k1_MSFT_s114
                                 + 0.809926 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.02838 w_k1_ADM_s114
                                 + 1.14287 w_k1_AXP_s114
                                 + 1.45303 w_k1_DIS_s114
                                 + 1.14736 w_k1_FDX_s114
                                 + 1.28057 w_k1_IBM_s114
                                 + 1.2495 w_k1_INTC_s114 + 1.33713 w_k1_KO_s114
                                 + 1.21377 w_k1_MAT_s114
                                 + 1.20187 w_k1_MSFT_s114
                                 + 1.12654 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.01832 w_k1_ADM_s114
                                 + 1.1729 w_k1_AXP_s114 + 1.21631 w_k1_DIS_s114
                                 + 1.14 w_k1_FDX_s114 + 0.853491 w_k1_IBM_s114
                                 + 0.949229 w_k1_INTC_s114
                                 + 0.936091 w_k1_KO_s114
                                 + 1.01458 w_k1_MAT_s114
                                 + 1.01552 w_k1_MSFT_s114
                                 + 0.996219 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.787972 w_k1_ADM_s114
                                 + 0.840998 w_k1_AXP_s114
                                 + 0.95872 w_k1_DIS_s114
                                 + 0.897325 w_k1_FDX_s114
                                 + 1.1136 w_k1_IBM_s114
                                 + 0.930941 w_k1_INTC_s114
                                 + 1.02813 w_k1_KO_s114
                                 + 0.926381 w_k1_MAT_s114
                                 + 1.0529 w_k1_MSFT_s114 + 1.04016 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.786807 w_k1_ADM_s114
                                 + 0.993367 w_k1_AXP_s114
                                 + 1.0248 w_k1_DIS_s114
                                 + 0.930187 w_k1_FDX_s114
                                 + 1.32882 w_k1_IBM_s114
                                 + 0.939207 w_k1_INTC_s114
                                 + 1.02126 w_k1_KO_s114 + 1.14922 w_k1_MAT_s114
                                 + 1.00823 w_k1_MSFT_s114
                                 + 0.918998 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.752269 w_k1_ADM_s114
                                 + 0.659305 w_k1_AXP_s114
                                 + 0.600433 w_k1_DIS_s114
                                 + 0.720225 w_k1_FDX_s114
                                 + 0.803148 w_k1_IBM_s114
                                 + 0.868395 w_k1_INTC_s114
                                 + 0.747593 w_k1_KO_s114
                                 + 0.883993 w_k1_MAT_s114
                                 + 0.844172 w_k1_MSFT_s114
                                 + 0.771199 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.982324 w_k1_ADM_s115
                                 + 1.11889 w_k1_AXP_s115
                                 + 0.957329 w_k1_DIS_s115
                                 + 0.939871 w_k1_FDX_s115
                                 + 1.32497 w_k1_IBM_s115
                                 + 0.863409 w_k1_INTC_s115
                                 + 0.976721 w_k1_KO_s115
                                 + 1.34242 w_k1_MAT_s115
                                 + 0.819311 w_k1_MSFT_s115
                                 + 0.938697 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.727815 w_k1_ADM_s115
                                 + 0.943184 w_k1_AXP_s115
                                 + 1.06913 w_k1_DIS_s115
                                 + 1.00537 w_k1_FDX_s115
                                 + 0.69939 w_k1_IBM_s115
                                 + 1.21547 w_k1_INTC_s115
                                 + 1.00666 w_k1_KO_s115
                                 + 0.828526 w_k1_MAT_s115
                                 + 1.1663 w_k1_MSFT_s115
                                 + 0.960697 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.19788 w_k1_ADM_s115
                                 + 1.509 w_k1_AXP_s115 + 1.63899 w_k1_DIS_s115
                                 + 1.36509 w_k1_FDX_s115
                                 + 1.45755 w_k1_IBM_s115
                                 + 1.24835 w_k1_INTC_s115 + 1.5077 w_k1_KO_s115
                                 + 1.41273 w_k1_MAT_s115
                                 + 1.30618 w_k1_MSFT_s115
                                 + 1.20273 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.690623 w_k1_ADM_s115
                                 + 0.75745 w_k1_AXP_s115
                                 + 0.724813 w_k1_DIS_s115
                                 + 0.764512 w_k1_FDX_s115
                                 + 0.905262 w_k1_IBM_s115
                                 + 0.733442 w_k1_INTC_s115
                                 + 0.706896 w_k1_KO_s115
                                 + 0.604639 w_k1_MAT_s115
                                 + 0.783307 w_k1_MSFT_s115
                                 + 0.800779 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.708339 w_k1_ADM_s115
                                 + 0.995425 w_k1_AXP_s115
                                 + 0.647381 w_k1_DIS_s115
                                 + 0.880645 w_k1_FDX_s115
                                 + 0.734702 w_k1_IBM_s115
                                 + 0.981336 w_k1_INTC_s115
                                 + 0.619094 w_k1_KO_s115
                                 + 0.826436 w_k1_MAT_s115
                                 + 0.895105 w_k1_MSFT_s115
                                 + 0.834179 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.724001 w_k1_ADM_s115
                                 + 1.26467 w_k1_AXP_s115 + 1.4263 w_k1_DIS_s115
                                 + 1.18626 w_k1_FDX_s115
                                 + 0.935962 w_k1_IBM_s115
                                 + 1.33999 w_k1_INTC_s115
                                 + 1.09222 w_k1_KO_s115 + 1.05007 w_k1_MAT_s115
                                 + 1.22512 w_k1_MSFT_s115
                                 + 1.14871 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.761586 w_k1_ADM_s115
                                 + 1.06094 w_k1_AXP_s115
                                 + 1.05503 w_k1_DIS_s115
                                 + 0.937556 w_k1_FDX_s115
                                 + 0.955014 w_k1_IBM_s115
                                 + 1.07421 w_k1_INTC_s115
                                 + 0.985166 w_k1_KO_s115
                                 + 1.09257 w_k1_MAT_s115
                                 + 1.1256 w_k1_MSFT_s115
                                 + 0.959037 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.83778 w_k1_ADM_s115
                                 + 0.953716 w_k1_AXP_s115
                                 + 0.742541 w_k1_DIS_s115
                                 + 0.956857 w_k1_FDX_s115
                                 + 0.562379 w_k1_IBM_s115
                                 + 1.03442 w_k1_INTC_s115
                                 + 0.731459 w_k1_KO_s115
                                 + 0.463266 w_k1_MAT_s115
                                 + 0.83886 w_k1_MSFT_s115
                                 + 0.875361 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.48503 w_k1_ADM_s116
                                 + 1.43855 w_k1_AXP_s116
                                 + 1.74253 w_k1_DIS_s116
                                 + 1.45063 w_k1_FDX_s116
                                 + 1.10793 w_k1_IBM_s116
                                 + 0.925381 w_k1_INTC_s116
                                 + 1.09657 w_k1_KO_s116 + 1.23147 w_k1_MAT_s116
                                 + 1.25985 w_k1_MSFT_s116
                                 + 1.25443 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.36264 w_k1_ADM_s116
                                 + 1.21874 w_k1_AXP_s116
                                 + 1.18698 w_k1_DIS_s116
                                 + 1.21019 w_k1_FDX_s116
                                 + 0.902677 w_k1_IBM_s116
                                 + 0.843232 w_k1_INTC_s116
                                 + 0.911799 w_k1_KO_s116
                                 + 1.51254 w_k1_MAT_s116
                                 + 1.15032 w_k1_MSFT_s116
                                 + 1.15831 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.10483 w_k1_ADM_s116
                                 + 0.955149 w_k1_AXP_s116
                                 + 0.828739 w_k1_DIS_s116
                                 + 0.985101 w_k1_FDX_s116
                                 + 0.822516 w_k1_IBM_s116
                                 + 0.735866 w_k1_INTC_s116
                                 + 0.88641 w_k1_KO_s116 + 0.82812 w_k1_MAT_s116
                                 + 1.00342 w_k1_MSFT_s116
                                 + 0.916346 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.42555 w_k1_ADM_s116
                                 + 1.62944 w_k1_AXP_s116
                                 + 1.09894 w_k1_DIS_s116 + 1.1998 w_k1_FDX_s116
                                 + 1.17243 w_k1_IBM_s116
                                 + 1.08369 w_k1_INTC_s116
                                 + 0.916672 w_k1_KO_s116
                                 + 1.01293 w_k1_MAT_s116
                                 + 1.0143 w_k1_MSFT_s116 + 1.07607 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.17592 w_k1_ADM_s116
                                 + 1.09312 w_k1_AXP_s116
                                 + 1.57828 w_k1_DIS_s116
                                 + 1.12607 w_k1_FDX_s116
                                 + 1.24368 w_k1_IBM_s116
                                 + 0.928456 w_k1_INTC_s116
                                 + 1.2258 w_k1_KO_s116 + 1.27598 w_k1_MAT_s116
                                 + 1.21134 w_k1_MSFT_s116
                                 + 1.05125 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.967035 w_k1_ADM_s116
                                 + 0.886408 w_k1_AXP_s116
                                 + 0.882206 w_k1_DIS_s116
                                 + 0.908966 w_k1_FDX_s116
                                 + 1.22136 w_k1_IBM_s116
                                 + 0.750391 w_k1_INTC_s116
                                 + 1.01949 w_k1_KO_s116 + 1.25218 w_k1_MAT_s116
                                 + 1.02058 w_k1_MSFT_s116
                                 + 0.903434 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.13937 w_k1_ADM_s116
                                 + 1.02662 w_k1_AXP_s116
                                 + 1.19717 w_k1_DIS_s116
                                 + 1.05863 w_k1_FDX_s116
                                 + 1.16022 w_k1_IBM_s116
                                 + 0.92541 w_k1_INTC_s116
                                 + 1.11147 w_k1_KO_s116 + 1.17082 w_k1_MAT_s116
                                 + 1.08291 w_k1_MSFT_s116
                                 + 1.11272 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.826379 w_k1_ADM_s116
                                 + 0.725294 w_k1_AXP_s116
                                 + 0.673574 w_k1_DIS_s116
                                 + 0.815777 w_k1_FDX_s116
                                 + 0.665277 w_k1_IBM_s116
                                 + 0.796112 w_k1_INTC_s116
                                 + 0.726867 w_k1_KO_s116
                                 + 0.672668 w_k1_MAT_s116
                                 + 0.949336 w_k1_MSFT_s116
                                 + 0.86855 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.993136 w_k1_ADM_s117
                                 + 0.870992 w_k1_AXP_s117
                                 + 1.09535 w_k1_DIS_s117
                                 + 1.03377 w_k1_FDX_s117
                                 + 1.08301 w_k1_IBM_s117
                                 + 1.2657 w_k1_INTC_s117 + 1.25951 w_k1_KO_s117
                                 + 1.11232 w_k1_MAT_s117
                                 + 1.14594 w_k1_MSFT_s117
                                 + 0.969987 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.98661 w_k1_ADM_s117
                                 + 0.893627 w_k1_AXP_s117
                                 + 0.995073 w_k1_DIS_s117
                                 + 1.1321 w_k1_FDX_s117
                                 + 0.776905 w_k1_IBM_s117
                                 + 1.32453 w_k1_INTC_s117
                                 + 0.866523 w_k1_KO_s117
                                 + 0.738846 w_k1_MAT_s117
                                 + 1.11726 w_k1_MSFT_s117
                                 + 0.972767 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.21719 w_k1_ADM_s117
                                 + 1.05701 w_k1_AXP_s117
                                 + 1.31961 w_k1_DIS_s117
                                 + 1.20698 w_k1_FDX_s117
                                 + 1.07079 w_k1_IBM_s117
                                 + 1.29768 w_k1_INTC_s117
                                 + 1.13039 w_k1_KO_s117
                                 + 0.932262 w_k1_MAT_s117
                                 + 1.23539 w_k1_MSFT_s117
                                 + 0.931732 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.674093 w_k1_ADM_s117
                                 + 0.704339 w_k1_AXP_s117
                                 + 0.807849 w_k1_DIS_s117
                                 + 0.822648 w_k1_FDX_s117
                                 + 1.21177 w_k1_IBM_s117
                                 + 1.11806 w_k1_INTC_s117
                                 + 0.907618 w_k1_KO_s117
                                 + 0.877278 w_k1_MAT_s117
                                 + 0.848297 w_k1_MSFT_s117
                                 + 0.897816 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.943917 w_k1_ADM_s117
                                 + 1.00001 w_k1_AXP_s117
                                 + 1.05009 w_k1_DIS_s117 + 1.0032 w_k1_FDX_s117
                                 + 1.55348 w_k1_IBM_s117
                                 + 1.19549 w_k1_INTC_s117
                                 + 1.34837 w_k1_KO_s117 + 1.6828 w_k1_MAT_s117
                                 + 1.19844 w_k1_MSFT_s117
                                 + 1.00874 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.16608 w_k1_ADM_s117
                                 + 1.44722 w_k1_AXP_s117
                                 + 1.81832 w_k1_DIS_s117
                                 + 1.50172 w_k1_FDX_s117
                                 + 1.31281 w_k1_IBM_s117
                                 + 1.51522 w_k1_INTC_s117
                                 + 1.61011 w_k1_KO_s117 + 1.85555 w_k1_MAT_s117
                                 + 1.67368 w_k1_MSFT_s117
                                 + 1.23171 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.942792 w_k1_ADM_s117
                                 + 0.662702 w_k1_AXP_s117
                                 + 0.69978 w_k1_DIS_s117
                                 + 0.850942 w_k1_FDX_s117
                                 + 0.807893 w_k1_IBM_s117
                                 + 0.894222 w_k1_INTC_s117
                                 + 0.83968 w_k1_KO_s117 + 0.58431 w_k1_MAT_s117
                                 + 0.859734 w_k1_MSFT_s117
                                 + 0.841166 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.02819 w_k1_ADM_s117
                                 + 1.19838 w_k1_AXP_s117
                                 + 0.900371 w_k1_DIS_s117
                                 + 1.07565 w_k1_FDX_s117
                                 + 1.04918 w_k1_IBM_s117
                                 + 1.11629 w_k1_INTC_s117
                                 + 1.03837 w_k1_KO_s117 + 1.08498 w_k1_MAT_s117
                                 + 1.02256 w_k1_MSFT_s117
                                 + 0.936854 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0169383642164831 w_k1_ADM_s000
                                 + 0.0151525023604283 w_k1_ADM_s001
                                 + 0.0157952157917588 w_k1_ADM_s002
                                 + 0.0105928989542985 w_k1_ADM_s003
                                 + 0.0167079010268606 w_k1_ADM_s004
                                 + 0.0586748306904672 w_k1_ADM_s005
                                 + 0.0316516730293009 w_k1_ADM_s006
                                 + 0.0164257659054868 w_k1_ADM_s007
                                 + 0.0139066202037666 w_k1_ADM_s010
                                 + 0.0271071151112151 w_k1_ADM_s011
                                 + 0.0634960529145457 w_k1_ADM_s012
                                 + 0.0150699102395124 w_k1_ADM_s013
                                 + 0.0422736552289806 w_k1_ADM_s014
                                 + 0.0181333312484482 w_k1_ADM_s015
                                 + 0.0559302812451139 w_k1_ADM_s016
                                 + 0.0450105139529565 w_k1_ADM_s017
                                 + 0.0376447920146078 w_k1_ADM_s100
                                 + 0.0201658430711384 w_k1_ADM_s101
                                 + 0.0201497643983641 w_k1_ADM_s102
                                 + 0.0362628537398198 w_k1_ADM_s103
                                 + 0.0244071437026166 w_k1_ADM_s104
                                 + 0.0217948456683195 w_k1_ADM_s105
                                 + 0.0302500055832944 w_k1_ADM_s106
                                 + 0.0273108749790176 w_k1_ADM_s107
                                 + 0.0807539489647398 w_k1_ADM_s110
                                 + 0.0377365835343283 w_k1_ADM_s111
                                 + 0.0513645529874417 w_k1_ADM_s112
                                 + 0.0185833854542457 w_k1_ADM_s113
                                 + 0.0277845479169308 w_k1_ADM_s114
                                 + 0.0251872566881282 w_k1_ADM_s115
                                 + 0.0353618726270389 w_k1_ADM_s116
                                 + 0.0223917309029401 w_k1_ADM_s117
                                 + 0.0176336503363053 w_k1_AXP_s000
                                 + 0.0143980081031224 w_k1_AXP_s001
                                 + 0.0186920057777835 w_k1_AXP_s002
                                 + 0.0123850867441044 w_k1_AXP_s003
                                 + 0.0176257623550764 w_k1_AXP_s004
                                 + 0.0634960560376079 w_k1_AXP_s005
                                 + 0.0321334219223144 w_k1_AXP_s006
                                 + 0.018250826828957 w_k1_AXP_s007
                                 + 0.0159964227582645 w_k1_AXP_s010
                                 + 0.0277234408051241 w_k1_AXP_s011
                                 + 0.0662577554582803 w_k1_AXP_s012
                                 + 0.0153093693712454 w_k1_AXP_s013
                                 + 0.0484083807932254 w_k1_AXP_s014
                                 + 0.0150457088164965 w_k1_AXP_s015
                                 + 0.0634686960118156 w_k1_AXP_s016
                                 + 0.0453117481388451 w_k1_AXP_s017
                                 + 0.0374744475822111 w_k1_AXP_s100
                                 + 0.0225114195102487 w_k1_AXP_s101
                                 + 0.0191404810450576 w_k1_AXP_s102
                                 + 0.0373905309129555 w_k1_AXP_s103
                                 + 0.0229167090413401 w_k1_AXP_s104
                                 + 0.0234112683501364 w_k1_AXP_s105
                                 + 0.0259719167898124 w_k1_AXP_s106
                                 + 0.0307489567582111 w_k1_AXP_s107
                                 + 0.0902118653550727 w_k1_AXP_s110
                                 + 0.0366551295645628 w_k1_AXP_s111
                                 + 0.0565212384420877 w_k1_AXP_s112
                                 + 0.0195024607192063 w_k1_AXP_s113
                                 + 0.0305400723578364 w_k1_AXP_s114
                                 + 0.0333569069037544 w_k1_AXP_s115
                                 + 0.0324644635801053 w_k1_AXP_s116
                                 + 0.020450002341888 w_k1_AXP_s117
                                 + 0.0180644990243504 w_k1_DIS_s000
                                 + 0.0152305142006645 w_k1_DIS_s001
                                 + 0.0193343809746223 w_k1_DIS_s002
                                 + 0.0131193325336182 w_k1_DIS_s003
                                 + 0.0167869759844973 w_k1_DIS_s004
                                 + 0.0610044491969988 w_k1_DIS_s005
                                 + 0.0332441924254502 w_k1_DIS_s006
                                 + 0.0182550762325833 w_k1_DIS_s007
                                 + 0.0161534154548846 w_k1_DIS_s010
                                 + 0.028691687973565 w_k1_DIS_s011
                                 + 0.068142190289226 w_k1_DIS_s012
                                 + 0.0135456969598333 w_k1_DIS_s013
                                 + 0.046606764345529 w_k1_DIS_s014
                                 + 0.01604901125157 w_k1_DIS_s015
                                 + 0.0621115017680519 w_k1_DIS_s016
                                 + 0.0436891426465728 w_k1_DIS_s017
                                 + 0.0407470207884513 w_k1_DIS_s100
                                 + 0.0230537626686315 w_k1_DIS_s101
                                 + 0.0188913504090603 w_k1_DIS_s102
                                 + 0.0398485379324768 w_k1_DIS_s103
                                 + 0.0239338909663147 w_k1_DIS_s104
                                 + 0.0221844563461542 w_k1_DIS_s105
                                 + 0.0288335044993702 w_k1_DIS_s106
                                 + 0.0312618309049799 w_k1_DIS_s107
                                 + 0.0937016465984413 w_k1_DIS_s110
                                 + 0.0413075375982063 w_k1_DIS_s111
                                 + 0.055562510653893 w_k1_DIS_s112
                                 + 0.0207011756430009 w_k1_DIS_s113
                                 + 0.032545501280817 w_k1_DIS_s114
                                 + 0.0324216868099145 w_k1_DIS_s115
                                 + 0.0329501645030646 w_k1_DIS_s116
                                 + 0.0229299180940033 w_k1_DIS_s117
                                 + 0.0170443386368808 w_k1_FDX_s000
                                 + 0.0153021067272207 w_k1_FDX_s001
                                 + 0.0195357754868275 w_k1_FDX_s002
                                 + 0.0124295450487396 w_k1_FDX_s003
                                 + 0.0166810392789319 w_k1_FDX_s004
                                 + 0.0602909708368866 w_k1_FDX_s005
                                 + 0.0317102688337983 w_k1_FDX_s006
                                 + 0.0177802188423659 w_k1_FDX_s007
                                 + 0.016370414104141 w_k1_FDX_s010
                                 + 0.0277387381101228 w_k1_FDX_s011
                                 + 0.0675985734552964 w_k1_FDX_s012
                                 + 0.01392618751479 w_k1_FDX_s013
                                 + 0.0437439372595964 w_k1_FDX_s014
                                 + 0.0142842479021683 w_k1_FDX_s015
                                 + 0.0585579208201943 w_k1_FDX_s016
                                 + 0.0430940802298172 w_k1_FDX_s017
                                 + 0.0408458315126991 w_k1_FDX_s100
                                 + 0.0236385221833205 w_k1_FDX_s101
                                 + 0.0198960727066569 w_k1_FDX_s102
                                 + 0.0359583391419077 w_k1_FDX_s103
                                 + 0.0230558936429168 w_k1_FDX_s104
                                 + 0.0225194484063755 w_k1_FDX_s105
                                 + 0.0274147211499528 w_k1_FDX_s106
                                 + 0.030833065337417 w_k1_FDX_s107
                                 + 0.0905358817472704 w_k1_FDX_s110
                                 + 0.0388201155056485 w_k1_FDX_s111
                                 + 0.0511168760079033 w_k1_FDX_s112
                                 + 0.0193619085705567 w_k1_FDX_s113
                                 + 0.0303983954920356 w_k1_FDX_s114
                                 + 0.0306119595370763 w_k1_FDX_s115
                                 + 0.0328339331202568 w_k1_FDX_s116
                                 + 0.0232816273146531 w_k1_FDX_s117
                                 + 0.0165866796262557 w_k1_IBM_s000
                                 + 0.0150843857311685 w_k1_IBM_s001
                                 + 0.0191692767318497 w_k1_IBM_s002
                                 + 0.0125866223668585 w_k1_IBM_s003
                                 + 0.0184022606096226 w_k1_IBM_s004
                                 + 0.064232138459167 w_k1_IBM_s005
                                 + 0.0325500471597853 w_k1_IBM_s006
                                 + 0.0186613290951243 w_k1_IBM_s007
                                 + 0.0165641783984775 w_k1_IBM_s010
                                 + 0.0281737979420551 w_k1_IBM_s011
                                 + 0.0704072112033188 w_k1_IBM_s012
                                 + 0.0138451335270151 w_k1_IBM_s013
                                 + 0.0471193635828756 w_k1_IBM_s014
                                 + 0.0155504365829971 w_k1_IBM_s015
                                 + 0.0585005476774166 w_k1_IBM_s016
                                 + 0.0466271327495399 w_k1_IBM_s017
                                 + 0.0420525223587834 w_k1_IBM_s100
                                 + 0.0236084807409662 w_k1_IBM_s101
                                 + 0.0206442755066891 w_k1_IBM_s102
                                 + 0.0366895233489662 w_k1_IBM_s103
                                 + 0.0229178705792916 w_k1_IBM_s104
                                 + 0.0232332797635444 w_k1_IBM_s105
                                 + 0.0265817329110906 w_k1_IBM_s106
                                 + 0.0313654931921732 w_k1_IBM_s107
                                 + 0.0888132272582043 w_k1_IBM_s110
                                 + 0.0393947363386515 w_k1_IBM_s111
                                 + 0.0540708411631834 w_k1_IBM_s112
                                 + 0.0195598485725987 w_k1_IBM_s113
                                 + 0.0305731549808674 w_k1_IBM_s114
                                 + 0.0298201020409099 w_k1_IBM_s115
                                 + 0.0317930422894144 w_k1_IBM_s116
                                 + 0.0236898349411949 w_k1_IBM_s117
                                 + 0.0186371089074484 w_k1_INTC_s000
                                 + 0.0154612847423913 w_k1_INTC_s001
                                 + 0.0191226325742479 w_k1_INTC_s002
                                 + 0.0121271241471111 w_k1_INTC_s003
                                 + 0.0171383691288755 w_k1_INTC_s004
                                 + 0.0598891530386721 w_k1_INTC_s005
                                 + 0.0341335101446849 w_k1_INTC_s006
                                 + 0.0178121863545333 w_k1_INTC_s007
                                 + 0.0158984263379208 w_k1_INTC_s010
                                 + 0.0281908150373651 w_k1_INTC_s011
                                 + 0.0762103143124692 w_k1_INTC_s012
                                 + 0.0133889419586002 w_k1_INTC_s013
                                 + 0.0427842617777729 w_k1_INTC_s014
                                 + 0.0170319414178813 w_k1_INTC_s015
                                 + 0.0558063959507209 w_k1_INTC_s016
                                 + 0.0459725303073826 w_k1_INTC_s017
                                 + 0.038657507943772 w_k1_INTC_s100
                                 + 0.0248886418359538 w_k1_INTC_s101
                                 + 0.0187954814112071 w_k1_INTC_s102
                                 + 0.0335388430577489 w_k1_INTC_s103
                                 + 0.020872146982384 w_k1_INTC_s104
                                 + 0.0217196884987645 w_k1_INTC_s105
                                 + 0.02565165754329 w_k1_INTC_s106
                                 + 0.0283614036983115 w_k1_INTC_s107
                                 + 0.0885833704305369 w_k1_INTC_s110
                                 + 0.0374113372460399 w_k1_INTC_s111
                                 + 0.052214669275006 w_k1_INTC_s112
                                 + 0.0200733733143824 w_k1_INTC_s113
                                 + 0.0299196286782929 w_k1_INTC_s114
                                 + 0.0321769371747425 w_k1_INTC_s115
                                 + 0.0264390251588119 w_k1_INTC_s116
                                 + 0.0260033355067337 w_k1_INTC_s117
                                 + 0.018426014841944 w_k1_KO_s000
                                 + 0.0141358849019338 w_k1_KO_s001
                                 + 0.0169586688343421 w_k1_KO_s002
                                 + 0.0111835454499661 w_k1_KO_s003
                                 + 0.0170143679277493 w_k1_KO_s004
                                 + 0.0632782987195355 w_k1_KO_s005
                                 + 0.0305973175067207 w_k1_KO_s006
                                 + 0.0167680322178624 w_k1_KO_s007
                                 + 0.0162212702881668 w_k1_KO_s010
                                 + 0.0285068537263955 w_k1_KO_s011
                                 + 0.0720526885750663 w_k1_KO_s012
                                 + 0.0153202721722108 w_k1_KO_s013
                                 + 0.043813745186903 w_k1_KO_s014
                                 + 0.0154270966710479 w_k1_KO_s015
                                 + 0.0555924841497718 w_k1_KO_s016
                                 + 0.0458566888160076 w_k1_KO_s017
                                 + 0.0423992804733058 w_k1_KO_s100
                                 + 0.023376603313402 w_k1_KO_s101
                                 + 0.0191095841294391 w_k1_KO_s102
                                 + 0.0383585573976747 w_k1_KO_s103
                                 + 0.0240893197986112 w_k1_KO_s104
                                 + 0.0201013512320511 w_k1_KO_s105
                                 + 0.0252514572618488 w_k1_KO_s106
                                 + 0.0304779271714314 w_k1_KO_s107
                                 + 0.0909149592503132 w_k1_KO_s110
                                 + 0.0365589196115461 w_k1_KO_s111
                                 + 0.0526112251593101 w_k1_KO_s112
                                 + 0.0183040019539649 w_k1_KO_s113
                                 + 0.0296549083184212 w_k1_KO_s114
                                 + 0.0296672151632034 w_k1_KO_s115
                                 + 0.0306965948781909 w_k1_KO_s116
                                 + 0.0244018285186874 w_k1_KO_s117
                                 + 0.0186025966366462 w_k1_MAT_s000
                                 + 0.0165102040689076 w_k1_MAT_s001
                                 + 0.0190698899140777 w_k1_MAT_s002
                                 + 0.0134012574523397 w_k1_MAT_s003
                                 + 0.0240569928296055 w_k1_MAT_s004
                                 + 0.0591548739689549 w_k1_MAT_s005
                                 + 0.0334535591337967 w_k1_MAT_s006
                                 + 0.01927993748312 w_k1_MAT_s007
                                 + 0.0175001327786352 w_k1_MAT_s010
                                 + 0.0346097716531415 w_k1_MAT_s011
                                 + 0.0668886132467725 w_k1_MAT_s012
                                 + 0.0148940570215039 w_k1_MAT_s013
                                 + 0.0475608125516847 w_k1_MAT_s014
                                 + 0.0161575761790463 w_k1_MAT_s015
                                 + 0.0625338291376552 w_k1_MAT_s016
                                 + 0.0500724046648943 w_k1_MAT_s017
                                 + 0.0424809702335773 w_k1_MAT_s100
                                 + 0.0244157577173438 w_k1_MAT_s101
                                 + 0.0192046342768325 w_k1_MAT_s102
                                 + 0.0335493422651811 w_k1_MAT_s103
                                 + 0.0235598820359773 w_k1_MAT_s104
                                 + 0.0243589879580173 w_k1_MAT_s105
                                 + 0.0211964686646071 w_k1_MAT_s106
                                 + 0.042466561831901 w_k1_MAT_s107
                                 + 0.0812040162971191 w_k1_MAT_s110
                                 + 0.0384819779310191 w_k1_MAT_s111
                                 + 0.0554477315657301 w_k1_MAT_s112
                                 + 0.0171310216018035 w_k1_MAT_s113
                                 + 0.0300120420920942 w_k1_MAT_s114
                                 + 0.0307643261609827 w_k1_MAT_s115
                                 + 0.0333212082031609 w_k1_MAT_s116
                                 + 0.0226525807468011 w_k1_MAT_s117
                                 + 0.0181391495660724 w_k1_MSFT_s000
                                 + 0.0156296870173912 w_k1_MSFT_s001
                                 + 0.0183215204390958 w_k1_MSFT_s002
                                 + 0.0125143019740686 w_k1_MSFT_s003
                                 + 0.0169317588113606 w_k1_MSFT_s004
                                 + 0.0574961913654713 w_k1_MSFT_s005
                                 + 0.034263355171647 w_k1_MSFT_s006
                                 + 0.0165368682059783 w_k1_MSFT_s007
                                 + 0.0153742698375202 w_k1_MSFT_s010
                                 + 0.0321729879498048 w_k1_MSFT_s011
                                 + 0.0680881089892521 w_k1_MSFT_s012
                                 + 0.0149723373311097 w_k1_MSFT_s013
                                 + 0.0445794673488923 w_k1_MSFT_s014
                                 + 0.0165328100453196 w_k1_MSFT_s015
                                 + 0.062130622323128 w_k1_MSFT_s016
                                 + 0.0452341258371527 w_k1_MSFT_s017
                                 + 0.0368407293311202 w_k1_MSFT_s100
                                 + 0.0222715896218668 w_k1_MSFT_s101
                                 + 0.019130129665923 w_k1_MSFT_s102
                                 + 0.0336399727179973 w_k1_MSFT_s103
                                 + 0.0237322484431693 w_k1_MSFT_s104
                                 + 0.0202818330288285 w_k1_MSFT_s105
                                 + 0.0248546291790792 w_k1_MSFT_s106
                                 + 0.0289973982991373 w_k1_MSFT_s107
                                 + 0.0935431450808131 w_k1_MSFT_s110
                                 + 0.0395856005519456 w_k1_MSFT_s111
                                 + 0.0556232522387152 w_k1_MSFT_s112
                                 + 0.0180901528962611 w_k1_MSFT_s113
                                 + 0.0303648674187323 w_k1_MSFT_s114
                                 + 0.0312424347560878 w_k1_MSFT_s115
                                 + 0.0336943277597355 w_k1_MSFT_s116
                                 + 0.0245364206438804 w_k1_MSFT_s117
                                 + 0.016716759275095 w_k1_PG_s000
                                 + 0.0156054683761722 w_k1_PG_s001
                                 + 0.0189155030493841 w_k1_PG_s002
                                 + 0.0118995528225538 w_k1_PG_s003
                                 + 0.0181939021021382 w_k1_PG_s004
                                 + 0.0614213935563804 w_k1_PG_s005
                                 + 0.0328920920551001 w_k1_PG_s006
                                 + 0.0166371871313677 w_k1_PG_s007
                                 + 0.0142070802211863 w_k1_PG_s010
                                 + 0.0289046402524287 w_k1_PG_s011
                                 + 0.0634856551354861 w_k1_PG_s012
                                 + 0.0135453772950552 w_k1_PG_s013
                                 + 0.0449916334898431 w_k1_PG_s014
                                 + 0.015760747908599 w_k1_PG_s015
                                 + 0.0538623521981697 w_k1_PG_s016
                                 + 0.0456096915625535 w_k1_PG_s017
                                 + 0.0406569225496385 w_k1_PG_s100
                                 + 0.0220131947722839 w_k1_PG_s101
                                 + 0.0191746131812627 w_k1_PG_s102
                                 + 0.0345357203119724 w_k1_PG_s103
                                 + 0.0209656931712391 w_k1_PG_s104
                                 + 0.0221367925046294 w_k1_PG_s105
                                 + 0.0250369449619598 w_k1_PG_s106
                                 + 0.0279845403920326 w_k1_PG_s107
                                 + 0.0877872295804637 w_k1_PG_s110
                                 + 0.0389063833099778 w_k1_PG_s111
                                 + 0.0535687081236222 w_k1_PG_s112
                                 + 0.0180105660036515 w_k1_PG_s113
                                 + 0.0288772559602026 w_k1_PG_s114
                                 + 0.0293986304584656 w_k1_PG_s115
                                 + 0.031715841508345 w_k1_PG_s116
                                 + 0.0213626807161289 w_k1_PG_s117 >= 1.041
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
