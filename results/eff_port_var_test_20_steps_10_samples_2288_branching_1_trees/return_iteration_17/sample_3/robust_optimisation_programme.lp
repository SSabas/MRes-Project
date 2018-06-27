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
 w_k1_ADM_s0_return_constraint:  1.18517 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.00487 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.95532 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.13172 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09742 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.919022 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.872607 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.31815 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.928168 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.90276 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.811059 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.24221 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.06879 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.07671 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.878429 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.869162 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.1569 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.695277 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.25996 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.21175 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.02481 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.930026 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.969172 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.848902 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.973801 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.48072 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.874851 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.09488 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.20754 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.924954 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.596287 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.842578 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.16195 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.972149 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.25169 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.01711 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.12827 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.52292 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24489 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.982515 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.873384 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.23628 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.14619 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.957784 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.929422 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.50383 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.01467 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.720952 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.854255 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.23007 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.965554 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.27612 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.0742 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.965904 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.089 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.665994 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.70472 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.31095 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.15817 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.02925 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.944194 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.594874 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.00026 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.86912 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.768803 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.20254 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.26698 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.918395 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.6302 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.626874 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.55641 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.934925 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.43897 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.869363 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.25073 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.58944 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.43632 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.933471 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.852613 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.30503 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.32147 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.977493 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.68232 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.67902 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.88276 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.979679 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.760657 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.59767 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.987507 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.2453 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.799262 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.09478 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.10643 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.677321 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.78125 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.33121 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.08841 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.88997 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.17365 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.533568 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.00911 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.68038 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.716343 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.31078 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.902185 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.836714 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.517441 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.618381 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.41003 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.880833 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.34671 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.03348 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.20731 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.94224 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.27671 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.95294 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.943473 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.15822 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.23146 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.921372 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.902502 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.35322 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.907663 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.896514 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.742542 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.42511 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.03497 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.10332 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.895652 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.936361 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.15264 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.729073 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.43674 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.13036 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.897626 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.918768 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.00648 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.711665 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.929519 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.3664 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.849421 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.2257 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.07376 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.939716 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.678425 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.819049 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.21478 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.01272 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.21808 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.91778 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.14178 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.68093 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.07834 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.03768 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.826565 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.38568 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.817217 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.06559 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.61334 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.41871 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.16608 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.931392 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.28361 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.849079 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.793669 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.33494 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.859412 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.05138 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.693119 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.805556 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.28566 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.19776 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.78178 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.15963 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.9761 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.613492 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.45407 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.84988 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.710783 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.06111 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.830419 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.960565 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.877105 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.47222 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.98596 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.726516 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.29155 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.09262 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.015 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.19836 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.22063 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.967962 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.905193 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.05485 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.14852 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.904251 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.840575 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14188 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.00116 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06168 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.840774 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.30988 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.11777 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.20202 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.987684 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.946356 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.21097 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.888035 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.26211 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.3101 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.09793 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.928878 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01648 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.778239 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.914602 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.24712 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.81823 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.11894 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.959066 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.99009 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.870657 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.843726 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.35652 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.03485 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.24644 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.962959 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.27937 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.43284 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17598 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.994198 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.858361 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.39496 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.958324 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.01737 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.685011 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.4223 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.00062 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06486 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.02597 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.18138 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.884228 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.23809 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.810922 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.990655 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.923459 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.727651 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.3265 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.25315 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.28137 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.938573 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.10642 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.657876 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.25601 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.55405 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.845969 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.2457 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.948971 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.00993 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.689375 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.575278 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.48204 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.829293 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.17846 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.984914 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.07368 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.33574 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.1115 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.03746 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.872153 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.49507 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.79626 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.02228 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.670769 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.55084 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.04905 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.758897 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.12761 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.868701 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.684889 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.02168 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.780219 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.07962 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.719207 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.653835 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.10977 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.16598 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.78243 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.14294 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.15792 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.638272 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.73226 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.17951 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.80174 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.33943 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.0297 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.0747 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.721584 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.459818 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 2.21812 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.847492 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.55595 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.12318 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.0273 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.31019 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.27326 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.953405 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.957099 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.0983 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15031 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.992982 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.801968 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.14946 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.849625 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.986544 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.769475 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.15172 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.932177 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.05716 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.848323 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.989109 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.14383 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.767302 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.39225 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.20309 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.02034 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.910847 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.05302 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.763471 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.925267 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.24701 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.846046 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.25635 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.03289 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.920918 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.651492 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.771645 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.08931 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.976186 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15616 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.927728 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.0756 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.42779 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06704 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.981246 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.966821 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.13386 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.0272 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.01309 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.863145 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.2076 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.93617 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02182 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.979887 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.14838 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.965199 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.11107 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.02105 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.10752 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.02949 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.828552 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.38817 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.26078 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.27628 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.02291 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.08486 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.759363 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.03353 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.37641 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.92333 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.12284 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.02294 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.962546 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.787235 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.742157 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.16059 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.931571 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.19367 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.999901 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04497 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.21903 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.140999798942364 z_k1_s0000
                                 - 0.0640000223759981 z_k1_s0001
                                 - 0.0580000349188773 z_k1_s0002
                                 - 0.169000177675715 z_k1_s0003
                                 - 0.148000127875767 z_k1_s0004
                                 - 0.0819999847473605 z_k1_s0005
                                 - 0.164000344294824 z_k1_s0006
                                 - 0.11099986165676 z_k1_s0007
                                 - 0.413998552797233 z_k1_s0010
                                 - 0.127999713382883 z_k1_s0011
                                 - 0.129999727706015 z_k1_s0012
                                 - 0.129999727706015 z_k1_s0013
                                 - 0.09399987098883 z_k1_s0014
                                 - 0.18699913316729 z_k1_s0015
                                 - 0.166000252921872 z_k1_s0016
                                 - 0.087999828019432 z_k1_s0017
                                 - 0.181000007486085 z_k1_s0020
                                 - 0.087000198657276 z_k1_s0021
                                 - 0.13799976905517 z_k1_s0022
                                 - 0.149999805407917 z_k1_s0023
                                 - 0.13799976905517 z_k1_s0024
                                 - 0.130999639682496 z_k1_s0025
                                 - 0.16300060195839 z_k1_s0026
                                 - 0.310000722778832 z_k1_s0027
                                 - 0.112999725741358 z_k1_s0030
                                 - 0.105999736279517 z_k1_s0031
                                 - 0.088999782700718 z_k1_s0032
                                 - 0.199999282339162 z_k1_s0033
                                 - 0.125999810313134 z_k1_s0034
                                 - 0.218999899419979 z_k1_s0035
                                 - 0.374999747883689 z_k1_s0036
                                 - 0.229999028890335 z_k1_s0037
                                 - 0.282000029602779 z_k1_s0040
                                 - 0.389000293250219 z_k1_s0041
                                 - 0.451001171738347 z_k1_s0042
                                 - 0.392000683661331 z_k1_s0043
                                 - 0.23000117148584 z_k1_s0044
                                 - 0.110000222658062 z_k1_s0045
                                 - 0.11500015434243 z_k1_s0046
                                 - 0.188000235435431 z_k1_s0047
                                 - 0.0540000067953017 z_k1_s0050
                                 - 0.0439999613147036 z_k1_s0051
                                 - 0.042000011918566 z_k1_s0052
                                 - 0.0449999957127544 z_k1_s0053
                                 - 0.068000070468139 z_k1_s0054
                                 - 0.094000009617748 z_k1_s0055
                                 - 0.118999974069288 z_k1_s0056
                                 - 0.130999670446114 z_k1_s0057
                                 - 0.30199911279105 z_k1_s0060
                                 - 0.217999749941363 z_k1_s0061
                                 - 0.650998570830393 z_k1_s0062
                                 - 0.227999674090135 z_k1_s0063
                                 - 0.463999287499288 z_k1_s0064
                                 - 0.459999317839779 z_k1_s0065
                                 - 0.256999594471387 z_k1_s0066
                                 - 0.22699982202507 z_k1_s0067
                                 - 0.0579999700187348 z_k1_s0070
                                 - 0.0689999699395301 z_k1_s0071
                                 - 0.147999624461558 z_k1_s0072
                                 - 0.135999742802525 z_k1_s0073
                                 - 0.216000037862895 z_k1_s0074
                                 - 0.139000255217358 z_k1_s0075
                                 - 0.094000048595894 z_k1_s0076
                                 - 0.223999958968917 z_k1_s0077
                                 - 0.129000079156212 z_k1_s0100
                                 - 0.140000521418318 z_k1_s0101
                                 - 0.0770001486298453 z_k1_s0102
                                 - 0.225000673529166 z_k1_s0103
                                 - 0.0880001303911862 z_k1_s0104
                                 - 0.0760001251696817 z_k1_s0105
                                 - 0.108000323294 z_k1_s0106
                                 - 0.0780001720900089 z_k1_s0107
                                 - 0.0879999009207614 z_k1_s0110
                                 - 0.0610000006200645 z_k1_s0111
                                 - 0.0749999155574671 z_k1_s0112
                                 - 0.189999786078917 z_k1_s0113
                                 - 0.0590000028718654 z_k1_s0114
                                 - 0.0610000006200645 z_k1_s0115
                                 - 0.111999873899151 z_k1_s0116
                                 - 0.0470000163826707 z_k1_s0117
                                 - 0.224000305092296 z_k1_s0120
                                 - 0.12799984776741 z_k1_s0121
                                 - 0.13299978109902 z_k1_s0122
                                 - 0.124999857288479 z_k1_s0123
                                 - 0.208999590698537 z_k1_s0124
                                 - 0.349999905207383 z_k1_s0125
                                 - 0.265000428970717 z_k1_s0126
                                 - 0.0899998667677406 z_k1_s0127
                                 - 0.252999790869718 z_k1_s0130
                                 - 0.124000406971814 z_k1_s0131
                                 - 0.198000100680698 z_k1_s0132
                                 - 0.120000009430717 z_k1_s0133
                                 - 0.102000044496116 z_k1_s0134
                                 - 0.0670000316121549 z_k1_s0135
                                 - 0.115000059704446 z_k1_s0136
                                 - 0.236999659905599 z_k1_s0137
                                 - 0.0900002179670698 z_k1_s0140
                                 - 0.113000235336403 z_k1_s0141
                                 - 0.0360000296867845 z_k1_s0142
                                 - 0.0530000325249799 z_k1_s0143
                                 - 0.0900002179670698 z_k1_s0144
                                 - 0.0420000442179225 z_k1_s0145
                                 - 0.109999883070574 z_k1_s0146
                                 - 0.0410000034627039 z_k1_s0147
                                 - 0.236000248112755 z_k1_s0150
                                 - 0.0770000900893402 z_k1_s0151
                                 - 0.0760001184804176 z_k1_s0152
                                 - 0.239000043139402 z_k1_s0153
                                 - 0.100000155895286 z_k1_s0154
                                 - 0.125999777127638 z_k1_s0155
                                 - 0.146000407307299 z_k1_s0156
                                 - 0.197999649772002 z_k1_s0157
                                 - 0.0600000715405959 z_k1_s0160
                                 - 0.0579997925559115 z_k1_s0161
                                 - 0.0259999849009252 z_k1_s0162
                                 - 0.0840001923568335 z_k1_s0163
                                 - 0.0619998895252838 z_k1_s0164
                                 - 0.044000006363104 z_k1_s0165
                                 - 0.0490001506248192 z_k1_s0166
                                 - 0.0779998164027767 z_k1_s0167
                                 - 0.0540000077820736 z_k1_s0170
                                 - 0.227999687057773 z_k1_s0171
                                 - 0.135999813332707 z_k1_s0172
                                 - 0.0540000077820736 z_k1_s0173
                                 - 0.0849998833329416 z_k1_s0174
                                 - 0.109999849019101 z_k1_s0175
                                 - 0.058000002291859 z_k1_s0176
                                 - 0.0939998709799589 z_k1_s0177
                                 - 0.102999770739053 z_k1_s1000
                                 - 0.147000205902279 z_k1_s1001
                                 - 0.0839998810589021 z_k1_s1002
                                 - 0.201999953607849 z_k1_s1003
                                 - 0.239999478168644 z_k1_s1004
                                 - 0.134999149095939 z_k1_s1005
                                 - 0.281999354998218 z_k1_s1006
                                 - 0.0809998716568249 z_k1_s1007
                                 - 0.400997189888182 z_k1_s1010
                                 - 0.359999058685967 z_k1_s1011
                                 - 0.52799920660361 z_k1_s1012
                                 - 0.253999042251892 z_k1_s1013
                                 - 0.43599709837154 z_k1_s1014
                                 - 0.219999131153773 z_k1_s1015
                                 - 0.3859972291096 z_k1_s1016
                                 - 0.350997320626243 z_k1_s1017
                                 - 0.304997364969486 z_k1_s1020
                                 - 0.719996200671342 z_k1_s1021
                                 - 0.149999484755306 z_k1_s1022
                                 - 0.251998820934302 z_k1_s1023
                                 - 0.48899823591067 z_k1_s1024
                                 - 0.378999779164951 z_k1_s1025
                                 - 0.504996477044117 z_k1_s1026
                                 - 0.213999080059722 z_k1_s1027
                                 - 0.386998837449777 z_k1_s1030
                                 - 0.284997715485562 z_k1_s1031
                                 - 0.239999965399578 z_k1_s1032
                                 - 0.430998711456957 z_k1_s1033
                                 - 0.0949994777940073 z_k1_s1034
                                 - 0.382997761181762 z_k1_s1035
                                 - 0.283997446418559 z_k1_s1036
                                 - 0.287998522686574 z_k1_s1037
                                 - 0.231999237553337 z_k1_s1040
                                 - 0.644996077277763 z_k1_s1041
                                 - 0.328999469722317 z_k1_s1042
                                 - 0.187999430413733 z_k1_s1043
                                 - 0.695996779663443 z_k1_s1044
                                 - 0.430997764505118 z_k1_s1045
                                 - 0.312000272256611 z_k1_s1046
                                 - 0.276998906014789 z_k1_s1047
                                 - 0.551999748968841 z_k1_s1050
                                 - 0.23799974063312 z_k1_s1051
                                 - 0.351999410387883 z_k1_s1052
                                 - 0.474000937819712 z_k1_s1053
                                 - 0.303999607212125 z_k1_s1054
                                 - 0.270999856872407 z_k1_s1055
                                 - 0.970998663558562 z_k1_s1056
                                 - 0.49699772573885 z_k1_s1057
                                 - 0.0929998565383199 z_k1_s1060
                                 - 0.266999283601 z_k1_s1061
                                 - 0.24999973555436 z_k1_s1062
                                 - 0.151999771585198 z_k1_s1063
                                 - 0.133999694813492 z_k1_s1064
                                 - 0.220999760594067 z_k1_s1065
                                 - 0.0609998452994599 z_k1_s1066
                                 - 0.230999411856976 z_k1_s1067
                                 - 0.508000481861222 z_k1_s1070
                                 - 0.223999902123172 z_k1_s1071
                                 - 0.194000049915555 z_k1_s1072
                                 - 0.274999456236193 z_k1_s1073
                                 - 0.340999423052843 z_k1_s1074
                                 - 0.170999936143132 z_k1_s1075
                                 - 0.122000063997399 z_k1_s1076
                                 - 0.597999551884252 z_k1_s1077
                                 - 0.54900099016382 z_k1_s1100
                                 - 1.71600289414546 z_k1_s1101
                                 - 0.69500171263233 z_k1_s1102
                                 - 1.68200634000514 z_k1_s1103
                                 - 0.861003520649537 z_k1_s1104
                                 - 0.672001493113754 z_k1_s1105
                                 - 0.430001292245011 z_k1_s1106
                                 - 0.743999815835896 z_k1_s1107
                                 - 0.486000863595766 z_k1_s1110
                                 - 0.616001124271631 z_k1_s1111
                                 - 0.803998143067214 z_k1_s1112
                                 - 0.412000705126412 z_k1_s1113
                                 - 1.71100170642368 z_k1_s1114
                                 - 1.24900463235046 z_k1_s1115
                                 - 0.576001144909974 z_k1_s1116
                                 - 0.700999277787524 z_k1_s1117
                                 - 0.901003382782318 z_k1_s1120
                                 - 1.15000293553057 z_k1_s1121
                                 - 0.717997601831768 z_k1_s1122
                                 - 0.480000220551392 z_k1_s1123
                                 - 0.497000308600127 z_k1_s1124
                                 - 0.62400028671681 z_k1_s1125
                                 - 1.20799862935338 z_k1_s1126
                                 - 0.840998540961001 z_k1_s1127
                                 - 2.0360072519152 z_k1_s1130
                                 - 1.20300116176097 z_k1_s1131
                                 - 2.28699927406864 z_k1_s1132
                                 - 2.55199632152823 z_k1_s1133
                                 - 1.58200338689794 z_k1_s1134
                                 - 1.6009971303687 z_k1_s1135
                                 - 2.70400729336976 z_k1_s1136
                                 - 1.29100092970319 z_k1_s1137
                                 - 1.59100080924191 z_k1_s1140
                                 - 1.13499794936595 z_k1_s1141
                                 - 0.864999813655552 z_k1_s1142
                                 - 0.843999670170483 z_k1_s1143
                                 - 1.51399610506597 z_k1_s1144
                                 - 0.684999862782714 z_k1_s1145
                                 - 0.770999640355419 z_k1_s1146
                                 - 1.54799957750641 z_k1_s1147
                                 - 0.680001398283611 z_k1_s1150
                                 - 1.50000088532778 z_k1_s1151
                                 - 0.696001457573858 z_k1_s1152
                                 - 1.80900338557922 z_k1_s1153
                                 - 0.499000867756677 z_k1_s1154
                                 - 1.08000213283853 z_k1_s1155
                                 - 0.420000995593041 z_k1_s1156
                                 - 0.793004480706724 z_k1_s1157
                                 - 0.249000191863509 z_k1_s1160
                                 - 0.937000606243735 z_k1_s1161
                                 - 0.721003112946919 z_k1_s1162
                                 - 0.964000292905837 z_k1_s1163
                                 - 0.664001961770252 z_k1_s1164
                                 - 0.277999976159915 z_k1_s1165
                                 - 0.503000084012443 z_k1_s1166
                                 - 1.12200148041185 z_k1_s1167
                                 - 0.595000153896939 z_k1_s1170
                                 - 0.367999708203187 z_k1_s1171
                                 - 0.555999805883266 z_k1_s1172
                                 - 0.342999401963589 z_k1_s1173
                                 - 0.321000170233542 z_k1_s1174
                                 - 0.507001281175252 z_k1_s1175
                                 - 0.255000204941652 z_k1_s1176
                                 - 0.298000147933312 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.869614 w_k1_ADM_s000
                                 + 1.10527 w_k1_AXP_s000
                                 + 0.887315 w_k1_DIS_s000
                                 + 0.915045 w_k1_FDX_s000
                                 + 1.23553 w_k1_IBM_s000
                                 + 1.02199 w_k1_INTC_s000
                                 + 1.01166 w_k1_KO_s000
                                 + 0.953222 w_k1_MAT_s000
                                 + 0.87294 w_k1_MSFT_s000
                                 + 1.01511 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.14396 w_k1_ADM_s000
                                 + 0.908418 w_k1_AXP_s000
                                 + 0.950466 w_k1_DIS_s000
                                 + 0.999999 w_k1_FDX_s000
                                 + 1.14368 w_k1_IBM_s000
                                 + 1.13154 w_k1_INTC_s000
                                 + 1.03807 w_k1_KO_s000
                                 + 0.794081 w_k1_MAT_s000
                                 + 0.958589 w_k1_MSFT_s000
                                 + 1.0518 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.905138 w_k1_ADM_s000
                                 + 0.844783 w_k1_AXP_s000
                                 + 0.764251 w_k1_DIS_s000
                                 + 0.727704 w_k1_FDX_s000
                                 + 1.27045 w_k1_IBM_s000
                                 + 0.884538 w_k1_INTC_s000
                                 + 1.02208 w_k1_KO_s000 + 1.10621 w_k1_MAT_s000
                                 + 0.791195 w_k1_MSFT_s000
                                 + 0.981208 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.86502 w_k1_ADM_s000
                                 + 0.677101 w_k1_AXP_s000
                                 + 0.558159 w_k1_DIS_s000
                                 + 0.694651 w_k1_FDX_s000
                                 + 0.721173 w_k1_IBM_s000
                                 + 0.633833 w_k1_INTC_s000
                                 + 0.739718 w_k1_KO_s000
                                 + 0.622824 w_k1_MAT_s000
                                 + 0.689208 w_k1_MSFT_s000
                                 + 0.880359 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.25368 w_k1_ADM_s000
                                 + 1.66077 w_k1_AXP_s000
                                 + 1.34615 w_k1_DIS_s000
                                 + 1.27982 w_k1_FDX_s000
                                 + 1.77158 w_k1_IBM_s000
                                 + 1.58152 w_k1_INTC_s000
                                 + 1.61673 w_k1_KO_s000 + 1.7768 w_k1_MAT_s000
                                 + 1.27292 w_k1_MSFT_s000 + 1.3068 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.19904 w_k1_ADM_s000
                                 + 1.20986 w_k1_AXP_s000
                                 + 1.16411 w_k1_DIS_s000 + 1.1231 w_k1_FDX_s000
                                 + 1.12156 w_k1_IBM_s000
                                 + 1.10435 w_k1_INTC_s000 + 1.1927 w_k1_KO_s000
                                 + 1.32599 w_k1_MAT_s000
                                 + 1.15893 w_k1_MSFT_s000
                                 + 1.01691 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.796929 w_k1_ADM_s000
                                 + 0.596863 w_k1_AXP_s000
                                 + 0.824857 w_k1_DIS_s000
                                 + 0.853338 w_k1_FDX_s000
                                 + 0.748913 w_k1_IBM_s000
                                 + 1.05898 w_k1_INTC_s000
                                 + 1.05746 w_k1_KO_s000
                                 + 0.498741 w_k1_MAT_s000
                                 + 0.931273 w_k1_MSFT_s000
                                 + 0.914886 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.26528 w_k1_ADM_s000
                                 + 1.12291 w_k1_AXP_s000
                                 + 1.34777 w_k1_DIS_s000
                                 + 1.17058 w_k1_FDX_s000
                                 + 0.809579 w_k1_IBM_s000
                                 + 1.29158 w_k1_INTC_s000
                                 + 1.22107 w_k1_KO_s000
                                 + 0.665267 w_k1_MAT_s000
                                 + 1.17078 w_k1_MSFT_s000
                                 + 1.10479 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.757069 w_k1_ADM_s001
                                 + 0.872622 w_k1_AXP_s001
                                 + 0.889533 w_k1_DIS_s001
                                 + 0.931588 w_k1_FDX_s001
                                 + 0.866027 w_k1_IBM_s001
                                 + 1.18098 w_k1_INTC_s001
                                 + 0.956058 w_k1_KO_s001
                                 + 0.653991 w_k1_MAT_s001
                                 + 1.02859 w_k1_MSFT_s001
                                 + 1.00898 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.904324 w_k1_ADM_s001
                                 + 0.92164 w_k1_AXP_s001
                                 + 1.04044 w_k1_DIS_s001
                                 + 0.877457 w_k1_FDX_s001
                                 + 1.53334 w_k1_IBM_s001
                                 + 0.9908 w_k1_INTC_s001 + 1.11441 w_k1_KO_s001
                                 + 1.22885 w_k1_MAT_s001
                                 + 0.989716 w_k1_MSFT_s001
                                 + 1.18824 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.02248 w_k1_ADM_s001
                                 + 1.22887 w_k1_AXP_s001
                                 + 0.902469 w_k1_DIS_s001
                                 + 0.96245 w_k1_FDX_s001
                                 + 1.40982 w_k1_IBM_s001
                                 + 1.0628 w_k1_INTC_s001 + 1.19004 w_k1_KO_s001
                                 + 1.52417 w_k1_MAT_s001
                                 + 1.12497 w_k1_MSFT_s001
                                 + 1.18694 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.80755 w_k1_ADM_s001
                                 + 1.46831 w_k1_AXP_s001
                                 + 1.84377 w_k1_DIS_s001
                                 + 1.42441 w_k1_FDX_s001
                                 + 1.29671 w_k1_IBM_s001
                                 + 1.39795 w_k1_INTC_s001
                                 + 1.50523 w_k1_KO_s001 + 1.49576 w_k1_MAT_s001
                                 + 1.67841 w_k1_MSFT_s001
                                 + 1.40089 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.871708 w_k1_ADM_s001
                                 + 1.04438 w_k1_AXP_s001
                                 + 0.690097 w_k1_DIS_s001
                                 + 0.904756 w_k1_FDX_s001
                                 + 0.718914 w_k1_IBM_s001
                                 + 1.10958 w_k1_INTC_s001
                                 + 0.76777 w_k1_KO_s001
                                 + 0.605231 w_k1_MAT_s001
                                 + 0.905741 w_k1_MSFT_s001
                                 + 0.848187 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.02427 w_k1_ADM_s001
                                 + 1.02302 w_k1_AXP_s001
                                 + 1.01604 w_k1_DIS_s001
                                 + 1.17141 w_k1_FDX_s001
                                 + 0.688899 w_k1_IBM_s001
                                 + 1.08851 w_k1_INTC_s001
                                 + 0.771609 w_k1_KO_s001
                                 + 0.772009 w_k1_MAT_s001
                                 + 1.13086 w_k1_MSFT_s001
                                 + 0.990469 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.29066 w_k1_ADM_s001
                                 + 1.50023 w_k1_AXP_s001
                                 + 1.32131 w_k1_DIS_s001
                                 + 1.26631 w_k1_FDX_s001
                                 + 1.22569 w_k1_IBM_s001
                                 + 1.21712 w_k1_INTC_s001
                                 + 1.09226 w_k1_KO_s001 + 1.28184 w_k1_MAT_s001
                                 + 1.26571 w_k1_MSFT_s001
                                 + 1.11892 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.898478 w_k1_ADM_s001
                                 + 0.641888 w_k1_AXP_s001
                                 + 0.572759 w_k1_DIS_s001
                                 + 0.725417 w_k1_FDX_s001
                                 + 0.741669 w_k1_IBM_s001
                                 + 0.985196 w_k1_INTC_s001
                                 + 0.774415 w_k1_KO_s001
                                 + 0.692119 w_k1_MAT_s001
                                 + 0.864587 w_k1_MSFT_s001
                                 + 0.879304 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.09002 w_k1_ADM_s002
                                 + 1.41509 w_k1_AXP_s002
                                 + 1.28663 w_k1_DIS_s002
                                 + 1.24223 w_k1_FDX_s002
                                 + 0.939719 w_k1_IBM_s002
                                 + 1.05224 w_k1_INTC_s002
                                 + 1.12452 w_k1_KO_s002 + 1.25877 w_k1_MAT_s002
                                 + 1.12453 w_k1_MSFT_s002
                                 + 1.17084 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.02866 w_k1_ADM_s002
                                 + 1.43989 w_k1_AXP_s002
                                 + 1.17082 w_k1_DIS_s002
                                 + 1.02395 w_k1_FDX_s002
                                 + 1.55746 w_k1_IBM_s002
                                 + 1.03916 w_k1_INTC_s002 + 1.3248 w_k1_KO_s002
                                 + 1.72002 w_k1_MAT_s002
                                 + 0.98846 w_k1_MSFT_s002
                                 + 1.04607 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.972527 w_k1_ADM_s002
                                 + 1.2381 w_k1_AXP_s002 + 1.04114 w_k1_DIS_s002
                                 + 1.05606 w_k1_FDX_s002
                                 + 1.22095 w_k1_IBM_s002
                                 + 1.00737 w_k1_INTC_s002
                                 + 1.24273 w_k1_KO_s002 + 1.65751 w_k1_MAT_s002
                                 + 1.04761 w_k1_MSFT_s002
                                 + 1.15658 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.08116 w_k1_ADM_s002
                                 + 1.11354 w_k1_AXP_s002
                                 + 0.838548 w_k1_DIS_s002
                                 + 0.966799 w_k1_FDX_s002
                                 + 0.709881 w_k1_IBM_s002
                                 + 0.852495 w_k1_INTC_s002
                                 + 0.797648 w_k1_KO_s002 + 0.8642 w_k1_MAT_s002
                                 + 0.882317 w_k1_MSFT_s002
                                 + 1.05468 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.786528 w_k1_ADM_s002
                                 + 0.706154 w_k1_AXP_s002
                                 + 0.567119 w_k1_DIS_s002
                                 + 0.749306 w_k1_FDX_s002
                                 + 0.601226 w_k1_IBM_s002
                                 + 0.819374 w_k1_INTC_s002
                                 + 0.720073 w_k1_KO_s002
                                 + 0.785692 w_k1_MAT_s002
                                 + 0.873655 w_k1_MSFT_s002
                                 + 0.833008 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.58275 w_k1_ADM_s002
                                 + 1.60621 w_k1_AXP_s002
                                 + 1.81774 w_k1_DIS_s002
                                 + 1.33466 w_k1_FDX_s002
                                 + 1.36656 w_k1_IBM_s002
                                 + 1.11648 w_k1_INTC_s002
                                 + 1.58138 w_k1_KO_s002 + 1.88707 w_k1_MAT_s002
                                 + 1.39699 w_k1_MSFT_s002
                                 + 1.43285 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.88351 w_k1_ADM_s002
                                 + 0.81808 w_k1_AXP_s002
                                 + 0.660483 w_k1_DIS_s002
                                 + 0.788692 w_k1_FDX_s002
                                 + 0.937542 w_k1_IBM_s002
                                 + 0.702847 w_k1_INTC_s002
                                 + 0.939768 w_k1_KO_s002
                                 + 1.08329 w_k1_MAT_s002
                                 + 0.831747 w_k1_MSFT_s002
                                 + 0.951692 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.932542 w_k1_ADM_s002
                                 + 0.92475 w_k1_AXP_s002
                                 + 0.918787 w_k1_DIS_s002
                                 + 0.986278 w_k1_FDX_s002
                                 + 1.02223 w_k1_IBM_s002
                                 + 0.986113 w_k1_INTC_s002
                                 + 1.00781 w_k1_KO_s002 + 1.30367 w_k1_MAT_s002
                                 + 0.987297 w_k1_MSFT_s002
                                 + 1.08949 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.839734 w_k1_ADM_s003
                                 + 0.785239 w_k1_AXP_s003
                                 + 0.69299 w_k1_DIS_s003
                                 + 0.663341 w_k1_FDX_s003
                                 + 0.977614 w_k1_IBM_s003
                                 + 0.668757 w_k1_INTC_s003
                                 + 0.73339 w_k1_KO_s003 + 0.50602 w_k1_MAT_s003
                                 + 0.679921 w_k1_MSFT_s003
                                 + 0.734522 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.31308 w_k1_ADM_s003
                                 + 1.51156 w_k1_AXP_s003
                                 + 1.62183 w_k1_DIS_s003
                                 + 1.39459 w_k1_FDX_s003
                                 + 1.16713 w_k1_IBM_s003
                                 + 0.98876 w_k1_INTC_s003
                                 + 1.20594 w_k1_KO_s003 + 1.42246 w_k1_MAT_s003
                                 + 1.25849 w_k1_MSFT_s003
                                 + 1.08379 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.53384 w_k1_ADM_s003
                                 + 1.63013 w_k1_AXP_s003
                                 + 2.15227 w_k1_DIS_s003
                                 + 1.45821 w_k1_FDX_s003
                                 + 1.67669 w_k1_IBM_s003
                                 + 1.15556 w_k1_INTC_s003
                                 + 1.53761 w_k1_KO_s003 + 1.21637 w_k1_MAT_s003
                                 + 1.21652 w_k1_MSFT_s003
                                 + 1.24695 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.07076 w_k1_ADM_s003
                                 + 1.19792 w_k1_AXP_s003
                                 + 1.04054 w_k1_DIS_s003
                                 + 1.05081 w_k1_FDX_s003
                                 + 0.826844 w_k1_IBM_s003
                                 + 0.906293 w_k1_INTC_s003
                                 + 0.781982 w_k1_KO_s003
                                 + 0.905768 w_k1_MAT_s003
                                 + 0.965244 w_k1_MSFT_s003
                                 + 0.901668 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.63263 w_k1_ADM_s003
                                 + 0.635728 w_k1_AXP_s003
                                 + 0.547652 w_k1_DIS_s003
                                 + 0.718353 w_k1_FDX_s003
                                 + 0.647964 w_k1_IBM_s003
                                 + 0.752708 w_k1_INTC_s003
                                 + 0.718521 w_k1_KO_s003
                                 + 0.503098 w_k1_MAT_s003
                                 + 0.793018 w_k1_MSFT_s003
                                 + 0.670025 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.04745 w_k1_ADM_s003
                                 + 1.26399 w_k1_AXP_s003
                                 + 1.24256 w_k1_DIS_s003
                                 + 1.15807 w_k1_FDX_s003
                                 + 1.15801 w_k1_IBM_s003
                                 + 0.955809 w_k1_INTC_s003
                                 + 1.03384 w_k1_KO_s003 + 1.34245 w_k1_MAT_s003
                                 + 1.0351 w_k1_MSFT_s003 + 1.09975 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.06605 w_k1_ADM_s003
                                 + 0.873078 w_k1_AXP_s003
                                 + 0.870039 w_k1_DIS_s003
                                 + 0.893734 w_k1_FDX_s003
                                 + 0.827406 w_k1_IBM_s003
                                 + 0.906703 w_k1_INTC_s003
                                 + 0.827346 w_k1_KO_s003
                                 + 0.638809 w_k1_MAT_s003
                                 + 0.816563 w_k1_MSFT_s003
                                 + 0.919303 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.13025 w_k1_ADM_s003
                                 + 1.12509 w_k1_AXP_s003
                                 + 1.12425 w_k1_DIS_s003
                                 + 1.07613 w_k1_FDX_s003
                                 + 1.48213 w_k1_IBM_s003
                                 + 1.03601 w_k1_INTC_s003
                                 + 1.26299 w_k1_KO_s003 + 1.46485 w_k1_MAT_s003
                                 + 1.03254 w_k1_MSFT_s003
                                 + 1.07497 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.03357 w_k1_ADM_s004
                                 + 0.873165 w_k1_AXP_s004
                                 + 0.786245 w_k1_DIS_s004
                                 + 0.962113 w_k1_FDX_s004
                                 + 0.812208 w_k1_IBM_s004
                                 + 0.907964 w_k1_INTC_s004
                                 + 0.913234 w_k1_KO_s004
                                 + 0.919447 w_k1_MAT_s004
                                 + 0.877944 w_k1_MSFT_s004
                                 + 0.879666 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.28657 w_k1_ADM_s004
                                 + 1.19455 w_k1_AXP_s004
                                 + 1.41613 w_k1_DIS_s004
                                 + 1.31966 w_k1_FDX_s004
                                 + 1.39689 w_k1_IBM_s004
                                 + 1.01759 w_k1_INTC_s004
                                 + 1.44158 w_k1_KO_s004 + 1.62131 w_k1_MAT_s004
                                 + 1.21178 w_k1_MSFT_s004
                                 + 1.07145 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01823 w_k1_ADM_s004
                                 + 0.602508 w_k1_AXP_s004
                                 + 0.612881 w_k1_DIS_s004
                                 + 0.77672 w_k1_FDX_s004
                                 + 0.620961 w_k1_IBM_s004
                                 + 0.782981 w_k1_INTC_s004
                                 + 0.65614 w_k1_KO_s004
                                 + 0.421453 w_k1_MAT_s004
                                 + 0.729394 w_k1_MSFT_s004
                                 + 0.737442 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.996743 w_k1_ADM_s004
                                 + 0.845224 w_k1_AXP_s004
                                 + 1.04665 w_k1_DIS_s004
                                 + 0.99804 w_k1_FDX_s004 + 1.0011 w_k1_IBM_s004
                                 + 0.991762 w_k1_INTC_s004
                                 + 0.949993 w_k1_KO_s004
                                 + 0.876679 w_k1_MAT_s004
                                 + 1.00982 w_k1_MSFT_s004
                                 + 0.981373 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02138 w_k1_ADM_s004
                                 + 0.954848 w_k1_AXP_s004
                                 + 1.19261 w_k1_DIS_s004
                                 + 1.21263 w_k1_FDX_s004
                                 + 0.808821 w_k1_IBM_s004
                                 + 0.960723 w_k1_INTC_s004
                                 + 1.03466 w_k1_KO_s004 + 0.71344 w_k1_MAT_s004
                                 + 1.10794 w_k1_MSFT_s004
                                 + 1.05624 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.796678 w_k1_ADM_s004
                                 + 0.628235 w_k1_AXP_s004
                                 + 0.608995 w_k1_DIS_s004
                                 + 0.715516 w_k1_FDX_s004
                                 + 0.923076 w_k1_IBM_s004
                                 + 0.638274 w_k1_INTC_s004
                                 + 0.760519 w_k1_KO_s004
                                 + 0.885505 w_k1_MAT_s004
                                 + 0.658407 w_k1_MSFT_s004
                                 + 0.80715 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.26516 w_k1_ADM_s004
                                 + 1.52693 w_k1_AXP_s004
                                 + 2.24272 w_k1_DIS_s004
                                 + 1.48909 w_k1_FDX_s004
                                 + 1.67759 w_k1_IBM_s004
                                 + 1.46577 w_k1_INTC_s004
                                 + 1.89919 w_k1_KO_s004 + 2.22452 w_k1_MAT_s004
                                 + 1.6853 w_k1_MSFT_s004 + 1.28648 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.947869 w_k1_ADM_s004
                                 + 0.81524 w_k1_AXP_s004
                                 + 0.827579 w_k1_DIS_s004
                                 + 0.862463 w_k1_FDX_s004
                                 + 1.44202 w_k1_IBM_s004
                                 + 0.911806 w_k1_INTC_s004
                                 + 1.02164 w_k1_KO_s004 + 1.09312 w_k1_MAT_s004
                                 + 0.837374 w_k1_MSFT_s004
                                 + 0.923926 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.80288 w_k1_ADM_s005
                                 + 0.665183 w_k1_AXP_s005
                                 + 0.677078 w_k1_DIS_s005
                                 + 0.762513 w_k1_FDX_s005
                                 + 0.629587 w_k1_IBM_s005
                                 + 0.848192 w_k1_INTC_s005
                                 + 0.713765 w_k1_KO_s005
                                 + 0.419466 w_k1_MAT_s005
                                 + 0.925979 w_k1_MSFT_s005
                                 + 0.805785 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07575 w_k1_ADM_s005
                                 + 0.985616 w_k1_AXP_s005
                                 + 0.708334 w_k1_DIS_s005
                                 + 0.843162 w_k1_FDX_s005
                                 + 1.03417 w_k1_IBM_s005
                                 + 0.870576 w_k1_INTC_s005
                                 + 0.786193 w_k1_KO_s005
                                 + 1.33598 w_k1_MAT_s005
                                 + 0.772135 w_k1_MSFT_s005
                                 + 0.966749 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.01596 w_k1_ADM_s005
                                 + 1.00923 w_k1_AXP_s005
                                 + 0.625373 w_k1_DIS_s005
                                 + 0.955004 w_k1_FDX_s005
                                 + 0.679525 w_k1_IBM_s005
                                 + 0.798339 w_k1_INTC_s005
                                 + 0.664782 w_k1_KO_s005
                                 + 0.753695 w_k1_MAT_s005
                                 + 0.8972 w_k1_MSFT_s005
                                 + 0.949965 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.50048 w_k1_ADM_s005
                                 + 1.67019 w_k1_AXP_s005
                                 + 1.79706 w_k1_DIS_s005
                                 + 1.45996 w_k1_FDX_s005
                                 + 1.18433 w_k1_IBM_s005
                                 + 1.46667 w_k1_INTC_s005
                                 + 1.40011 w_k1_KO_s005 + 1.40225 w_k1_MAT_s005
                                 + 1.67722 w_k1_MSFT_s005
                                 + 1.35063 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10548 w_k1_ADM_s005
                                 + 1.0467 w_k1_AXP_s005 + 1.17972 w_k1_DIS_s005
                                 + 1.02984 w_k1_FDX_s005
                                 + 1.26957 w_k1_IBM_s005
                                 + 0.940809 w_k1_INTC_s005
                                 + 1.10984 w_k1_KO_s005 + 1.51668 w_k1_MAT_s005
                                 + 1.21884 w_k1_MSFT_s005
                                 + 1.06171 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.13537 w_k1_ADM_s005
                                 + 1.16044 w_k1_AXP_s005
                                 + 0.988627 w_k1_DIS_s005
                                 + 1.12882 w_k1_FDX_s005
                                 + 0.77898 w_k1_IBM_s005
                                 + 1.15732 w_k1_INTC_s005
                                 + 0.826475 w_k1_KO_s005
                                 + 0.829494 w_k1_MAT_s005
                                 + 1.13473 w_k1_MSFT_s005
                                 + 0.912635 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.44742 w_k1_ADM_s005
                                 + 1.22071 w_k1_AXP_s005
                                 + 1.26353 w_k1_DIS_s005 + 1.2042 w_k1_FDX_s005
                                 + 1.07285 w_k1_IBM_s005 + 1.219 w_k1_INTC_s005
                                 + 1.10567 w_k1_KO_s005
                                 + 0.974535 w_k1_MAT_s005
                                 + 1.335 w_k1_MSFT_s005 + 1.18818 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.752859 w_k1_ADM_s005
                                 + 0.75656 w_k1_AXP_s005
                                 + 0.817221 w_k1_DIS_s005
                                 + 0.811008 w_k1_FDX_s005
                                 + 0.973378 w_k1_IBM_s005
                                 + 0.913948 w_k1_INTC_s005
                                 + 0.851884 w_k1_KO_s005
                                 + 0.864732 w_k1_MAT_s005
                                 + 0.949699 w_k1_MSFT_s005
                                 + 0.892793 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.12447 w_k1_ADM_s006
                                 + 1.53545 w_k1_AXP_s006
                                 + 1.74681 w_k1_DIS_s006
                                 + 1.43284 w_k1_FDX_s006
                                 + 1.21562 w_k1_IBM_s006
                                 + 1.28196 w_k1_INTC_s006 + 1.3737 w_k1_KO_s006
                                 + 1.77709 w_k1_MAT_s006
                                 + 1.54893 w_k1_MSFT_s006
                                 + 1.26944 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.998011 w_k1_ADM_s006
                                 + 0.966509 w_k1_AXP_s006
                                 + 1.18577 w_k1_DIS_s006
                                 + 1.11082 w_k1_FDX_s006
                                 + 0.907745 w_k1_IBM_s006
                                 + 1.07407 w_k1_INTC_s006
                                 + 1.03643 w_k1_KO_s006
                                 + 0.840885 w_k1_MAT_s006
                                 + 1.14928 w_k1_MSFT_s006
                                 + 1.08391 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.10597 w_k1_ADM_s006
                                 + 1.13905 w_k1_AXP_s006
                                 + 0.98778 w_k1_DIS_s006
                                 + 0.993267 w_k1_FDX_s006
                                 + 1.06873 w_k1_IBM_s006
                                 + 1.03986 w_k1_INTC_s006
                                 + 1.06529 w_k1_KO_s006 + 1.06864 w_k1_MAT_s006
                                 + 1.03586 w_k1_MSFT_s006
                                 + 1.04629 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.31848 w_k1_ADM_s006
                                 + 1.26612 w_k1_AXP_s006
                                 + 1.14577 w_k1_DIS_s006 + 1.2665 w_k1_FDX_s006
                                 + 0.73319 w_k1_IBM_s006
                                 + 0.866035 w_k1_INTC_s006
                                 + 0.952449 w_k1_KO_s006
                                 + 0.854757 w_k1_MAT_s006
                                 + 1.20443 w_k1_MSFT_s006
                                 + 1.03154 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.814593 w_k1_ADM_s006
                                 + 0.915515 w_k1_AXP_s006
                                 + 0.95545 w_k1_DIS_s006
                                 + 0.909289 w_k1_FDX_s006
                                 + 1.03308 w_k1_IBM_s006
                                 + 0.858465 w_k1_INTC_s006
                                 + 1.01562 w_k1_KO_s006
                                 + 0.766115 w_k1_MAT_s006
                                 + 0.950795 w_k1_MSFT_s006
                                 + 1.00382 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.743949 w_k1_ADM_s006
                                 + 0.778494 w_k1_AXP_s006
                                 + 0.663065 w_k1_DIS_s006
                                 + 0.762929 w_k1_FDX_s006
                                 + 0.908755 w_k1_IBM_s006
                                 + 0.735865 w_k1_INTC_s006
                                 + 0.728848 w_k1_KO_s006 + 1.0307 w_k1_MAT_s006
                                 + 0.823281 w_k1_MSFT_s006
                                 + 0.934893 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.09324 w_k1_ADM_s006
                                 + 1.18059 w_k1_AXP_s006
                                 + 1.41163 w_k1_DIS_s006
                                 + 1.20117 w_k1_FDX_s006
                                 + 1.34917 w_k1_IBM_s006
                                 + 1.00115 w_k1_INTC_s006
                                 + 1.10453 w_k1_KO_s006
                                 + 0.963373 w_k1_MAT_s006
                                 + 1.11787 w_k1_MSFT_s006
                                 + 1.15323 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.686992 w_k1_ADM_s006
                                 + 0.695054 w_k1_AXP_s006
                                 + 0.709663 w_k1_DIS_s006
                                 + 0.805849 w_k1_FDX_s006
                                 + 0.560955 w_k1_IBM_s006
                                 + 0.734332 w_k1_INTC_s006
                                 + 0.6335 w_k1_KO_s006 + 0.632536 w_k1_MAT_s006
                                 + 0.853359 w_k1_MSFT_s006
                                 + 0.748071 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.935493 w_k1_ADM_s007
                                 + 0.830348 w_k1_AXP_s007
                                 + 1.06579 w_k1_DIS_s007
                                 + 0.824128 w_k1_FDX_s007
                                 + 1.38505 w_k1_IBM_s007
                                 + 1.04283 w_k1_INTC_s007
                                 + 1.23802 w_k1_KO_s007 + 1.43826 w_k1_MAT_s007
                                 + 0.937155 w_k1_MSFT_s007
                                 + 0.953225 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.816729 w_k1_ADM_s007
                                 + 0.884464 w_k1_AXP_s007
                                 + 0.70371 w_k1_DIS_s007
                                 + 0.759017 w_k1_FDX_s007
                                 + 0.938943 w_k1_IBM_s007
                                 + 0.617715 w_k1_INTC_s007
                                 + 0.775517 w_k1_KO_s007
                                 + 1.54318 w_k1_MAT_s007
                                 + 0.646341 w_k1_MSFT_s007
                                 + 0.80715 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.807473 w_k1_ADM_s007
                                 + 0.856629 w_k1_AXP_s007
                                 + 0.895408 w_k1_DIS_s007
                                 + 0.949065 w_k1_FDX_s007
                                 + 0.851156 w_k1_IBM_s007
                                 + 0.77388 w_k1_INTC_s007
                                 + 0.861129 w_k1_KO_s007
                                 + 1.17085 w_k1_MAT_s007
                                 + 0.898394 w_k1_MSFT_s007
                                 + 0.863162 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.45325 w_k1_ADM_s007
                                 + 1.54854 w_k1_AXP_s007
                                 + 1.51597 w_k1_DIS_s007 + 1.2714 w_k1_FDX_s007
                                 + 1.31994 w_k1_IBM_s007
                                 + 0.949452 w_k1_INTC_s007
                                 + 1.38959 w_k1_KO_s007 + 1.93398 w_k1_MAT_s007
                                 + 1.19071 w_k1_MSFT_s007 + 1.2172 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.17854 w_k1_ADM_s007
                                 + 1.28037 w_k1_AXP_s007 + 1.0473 w_k1_DIS_s007
                                 + 1.02234 w_k1_FDX_s007
                                 + 1.21967 w_k1_IBM_s007
                                 + 0.803394 w_k1_INTC_s007
                                 + 1.03015 w_k1_KO_s007 + 1.85553 w_k1_MAT_s007
                                 + 0.922174 w_k1_MSFT_s007
                                 + 1.13063 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.949291 w_k1_ADM_s007
                                 + 0.883596 w_k1_AXP_s007
                                 + 0.666149 w_k1_DIS_s007
                                 + 0.877926 w_k1_FDX_s007
                                 + 0.675137 w_k1_IBM_s007
                                 + 0.763131 w_k1_INTC_s007
                                 + 0.683351 w_k1_KO_s007
                                 + 1.07957 w_k1_MAT_s007
                                 + 0.864172 w_k1_MSFT_s007
                                 + 0.934514 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.16572 w_k1_ADM_s007
                                 + 1.4157 w_k1_AXP_s007
                                 + 0.921965 w_k1_DIS_s007
                                 + 1.07262 w_k1_FDX_s007
                                 + 0.849335 w_k1_IBM_s007
                                 + 0.956351 w_k1_INTC_s007
                                 + 0.937624 w_k1_KO_s007
                                 + 1.32875 w_k1_MAT_s007
                                 + 0.951285 w_k1_MSFT_s007
                                 + 0.982323 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.17712 w_k1_ADM_s007
                                 + 1.22757 w_k1_AXP_s007
                                 + 1.34037 w_k1_DIS_s007
                                 + 1.22858 w_k1_FDX_s007
                                 + 0.942132 w_k1_IBM_s007
                                 + 0.983289 w_k1_INTC_s007
                                 + 1.0408 w_k1_KO_s007 + 1.33528 w_k1_MAT_s007
                                 + 1.17765 w_k1_MSFT_s007
                                 + 1.21388 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.995457 w_k1_ADM_s010
                                 + 1.05461 w_k1_AXP_s010
                                 + 1.15087 w_k1_DIS_s010
                                 + 1.17018 w_k1_FDX_s010
                                 + 0.897596 w_k1_IBM_s010
                                 + 0.954113 w_k1_INTC_s010
                                 + 1.00618 w_k1_KO_s010 + 0.83815 w_k1_MAT_s010
                                 + 1.13335 w_k1_MSFT_s010
                                 + 1.09813 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.1935 w_k1_ADM_s010
                                 + 1.01257 w_k1_AXP_s010
                                 + 0.948649 w_k1_DIS_s010
                                 + 1.03512 w_k1_FDX_s010
                                 + 1.34509 w_k1_IBM_s010
                                 + 1.05412 w_k1_INTC_s010
                                 + 1.13285 w_k1_KO_s010 + 1.70278 w_k1_MAT_s010
                                 + 1.38141 w_k1_MSFT_s010
                                 + 0.987162 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.17655 w_k1_ADM_s010
                                 + 1.2693 w_k1_AXP_s010
                                 + 0.964076 w_k1_DIS_s010
                                 + 0.943072 w_k1_FDX_s010
                                 + 1.40997 w_k1_IBM_s010
                                 + 0.892658 w_k1_INTC_s010
                                 + 1.07887 w_k1_KO_s010 + 2.19681 w_k1_MAT_s010
                                 + 0.897598 w_k1_MSFT_s010
                                 + 1.08878 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.739192 w_k1_ADM_s010
                                 + 0.810091 w_k1_AXP_s010
                                 + 0.769956 w_k1_DIS_s010
                                 + 0.829441 w_k1_FDX_s010
                                 + 0.89631 w_k1_IBM_s010
                                 + 0.98797 w_k1_INTC_s010
                                 + 0.889933 w_k1_KO_s010
                                 + 0.799852 w_k1_MAT_s010
                                 + 0.889972 w_k1_MSFT_s010
                                 + 0.858898 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.18997 w_k1_ADM_s010
                                 + 0.955498 w_k1_AXP_s010
                                 + 0.756445 w_k1_DIS_s010
                                 + 0.994426 w_k1_FDX_s010
                                 + 0.675773 w_k1_IBM_s010
                                 + 0.812358 w_k1_INTC_s010
                                 + 0.763366 w_k1_KO_s010
                                 + 1.20607 w_k1_MAT_s010
                                 + 0.965791 w_k1_MSFT_s010
                                 + 0.879794 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.17867 w_k1_ADM_s010
                                 + 0.751442 w_k1_AXP_s010
                                 + 0.758991 w_k1_DIS_s010
                                 + 0.944491 w_k1_FDX_s010
                                 + 0.621616 w_k1_IBM_s010
                                 + 0.995263 w_k1_INTC_s010
                                 + 0.659431 w_k1_KO_s010
                                 + 0.512491 w_k1_MAT_s010
                                 + 1.07927 w_k1_MSFT_s010
                                 + 0.772121 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.16705 w_k1_ADM_s010
                                 + 1.1434 w_k1_AXP_s010 + 1.59686 w_k1_DIS_s010
                                 + 1.223 w_k1_FDX_s010 + 1.15478 w_k1_IBM_s010
                                 + 1.07152 w_k1_INTC_s010
                                 + 1.13475 w_k1_KO_s010 + 1.2059 w_k1_MAT_s010
                                 + 1.18076 w_k1_MSFT_s010
                                 + 1.02875 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.24677 w_k1_ADM_s010
                                 + 1.39051 w_k1_AXP_s010
                                 + 1.29912 w_k1_DIS_s010
                                 + 1.20466 w_k1_FDX_s010
                                 + 1.11112 w_k1_IBM_s010
                                 + 1.06987 w_k1_INTC_s010
                                 + 1.03531 w_k1_KO_s010 + 1.39404 w_k1_MAT_s010
                                 + 1.16899 w_k1_MSFT_s010
                                 + 1.02771 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.02929 w_k1_ADM_s011
                                 + 0.923238 w_k1_AXP_s011
                                 + 0.814953 w_k1_DIS_s011
                                 + 0.784268 w_k1_FDX_s011
                                 + 1.18226 w_k1_IBM_s011
                                 + 0.86399 w_k1_INTC_s011
                                 + 0.912057 w_k1_KO_s011
                                 + 1.02432 w_k1_MAT_s011
                                 + 0.907051 w_k1_MSFT_s011
                                 + 0.918064 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.37988 w_k1_ADM_s011
                                 + 0.854276 w_k1_AXP_s011
                                 + 1.15755 w_k1_DIS_s011
                                 + 0.807863 w_k1_FDX_s011
                                 + 1.61245 w_k1_IBM_s011
                                 + 0.814881 w_k1_INTC_s011
                                 + 1.28392 w_k1_KO_s011 + 1.68872 w_k1_MAT_s011
                                 + 0.988145 w_k1_MSFT_s011
                                 + 1.03663 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.19897 w_k1_ADM_s011
                                 + 0.865636 w_k1_AXP_s011
                                 + 1.31564 w_k1_DIS_s011
                                 + 1.02874 w_k1_FDX_s011
                                 + 1.12529 w_k1_IBM_s011
                                 + 1.43268 w_k1_INTC_s011
                                 + 1.23822 w_k1_KO_s011
                                 + 0.674249 w_k1_MAT_s011
                                 + 1.25681 w_k1_MSFT_s011
                                 + 1.05171 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.953367 w_k1_ADM_s011
                                 + 0.744268 w_k1_AXP_s011
                                 + 0.698019 w_k1_DIS_s011
                                 + 0.785947 w_k1_FDX_s011
                                 + 0.72793 w_k1_IBM_s011
                                 + 0.949275 w_k1_INTC_s011
                                 + 0.816272 w_k1_KO_s011
                                 + 0.61373 w_k1_MAT_s011
                                 + 0.90799 w_k1_MSFT_s011
                                 + 0.842694 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.17647 w_k1_ADM_s011
                                 + 1.31167 w_k1_AXP_s011
                                 + 1.31999 w_k1_DIS_s011
                                 + 1.27852 w_k1_FDX_s011
                                 + 1.13057 w_k1_IBM_s011
                                 + 0.942458 w_k1_INTC_s011
                                 + 1.05903 w_k1_KO_s011 + 1.09873 w_k1_MAT_s011
                                 + 1.16645 w_k1_MSFT_s011
                                 + 0.973046 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.21298 w_k1_ADM_s011
                                 + 0.957248 w_k1_AXP_s011
                                 + 1.00604 w_k1_DIS_s011
                                 + 0.946123 w_k1_FDX_s011
                                 + 0.932736 w_k1_IBM_s011
                                 + 0.993511 w_k1_INTC_s011
                                 + 0.955275 w_k1_KO_s011
                                 + 0.803204 w_k1_MAT_s011
                                 + 0.94697 w_k1_MSFT_s011
                                 + 0.940009 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.31732 w_k1_ADM_s011
                                 + 0.953495 w_k1_AXP_s011
                                 + 1.07511 w_k1_DIS_s011
                                 + 1.11016 w_k1_FDX_s011
                                 + 0.726952 w_k1_IBM_s011
                                 + 0.920293 w_k1_INTC_s011
                                 + 0.909736 w_k1_KO_s011
                                 + 0.62689 w_k1_MAT_s011
                                 + 1.03904 w_k1_MSFT_s011
                                 + 0.995944 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.36911 w_k1_ADM_s011
                                 + 1.0612 w_k1_AXP_s011 + 1.71351 w_k1_DIS_s011
                                 + 1.28519 w_k1_FDX_s011
                                 + 0.853384 w_k1_IBM_s011
                                 + 0.992431 w_k1_INTC_s011
                                 + 1.16983 w_k1_KO_s011
                                 + 0.840922 w_k1_MAT_s011
                                 + 1.18913 w_k1_MSFT_s011
                                 + 1.11661 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.973286 w_k1_ADM_s012
                                 + 1.30543 w_k1_AXP_s012
                                 + 1.17117 w_k1_DIS_s012
                                 + 1.13135 w_k1_FDX_s012 + 1.5183 w_k1_IBM_s012
                                 + 1.22212 w_k1_INTC_s012
                                 + 1.11812 w_k1_KO_s012 + 1.5807 w_k1_MAT_s012
                                 + 1.19354 w_k1_MSFT_s012
                                 + 1.12313 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.865349 w_k1_ADM_s012
                                 + 0.786445 w_k1_AXP_s012
                                 + 0.854787 w_k1_DIS_s012
                                 + 0.895846 w_k1_FDX_s012
                                 + 0.934503 w_k1_IBM_s012
                                 + 0.941493 w_k1_INTC_s012
                                 + 0.811954 w_k1_KO_s012
                                 + 0.643306 w_k1_MAT_s012
                                 + 0.941199 w_k1_MSFT_s012
                                 + 0.937806 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.54576 w_k1_ADM_s012
                                 + 1.67401 w_k1_AXP_s012
                                 + 1.53326 w_k1_DIS_s012
                                 + 1.51179 w_k1_FDX_s012
                                 + 1.26379 w_k1_IBM_s012
                                 + 1.12222 w_k1_INTC_s012
                                 + 1.17816 w_k1_KO_s012 + 1.77254 w_k1_MAT_s012
                                 + 1.29469 w_k1_MSFT_s012
                                 + 1.29146 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08174 w_k1_ADM_s012
                                 + 1.02895 w_k1_AXP_s012
                                 + 1.25149 w_k1_DIS_s012
                                 + 1.22472 w_k1_FDX_s012
                                 + 0.930142 w_k1_IBM_s012
                                 + 1.04714 w_k1_INTC_s012
                                 + 1.00012 w_k1_KO_s012
                                 + 0.969174 w_k1_MAT_s012
                                 + 1.13218 w_k1_MSFT_s012 + 1.1809 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.707838 w_k1_ADM_s012
                                 + 0.656156 w_k1_AXP_s012
                                 + 0.613525 w_k1_DIS_s012
                                 + 0.751437 w_k1_FDX_s012
                                 + 0.766415 w_k1_IBM_s012
                                 + 0.761995 w_k1_INTC_s012
                                 + 0.68268 w_k1_KO_s012
                                 + 0.856002 w_k1_MAT_s012
                                 + 0.905694 w_k1_MSFT_s012
                                 + 0.850259 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.96651 w_k1_ADM_s012
                                 + 1.25537 w_k1_AXP_s012
                                 + 0.975907 w_k1_DIS_s012
                                 + 1.07546 w_k1_FDX_s012
                                 + 0.920564 w_k1_IBM_s012
                                 + 0.870294 w_k1_INTC_s012
                                 + 0.83751 w_k1_KO_s012
                                 + 0.895209 w_k1_MAT_s012
                                 + 0.888027 w_k1_MSFT_s012
                                 + 1.12729 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.798417 w_k1_ADM_s012
                                 + 1.03795 w_k1_AXP_s012
                                 + 0.778228 w_k1_DIS_s012
                                 + 0.845116 w_k1_FDX_s012
                                 + 1.22303 w_k1_IBM_s012
                                 + 0.909128 w_k1_INTC_s012
                                 + 0.971188 w_k1_KO_s012
                                 + 1.26854 w_k1_MAT_s012
                                 + 0.796123 w_k1_MSFT_s012
                                 + 1.02773 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.23368 w_k1_ADM_s012
                                 + 0.94132 w_k1_AXP_s012
                                 + 1.50058 w_k1_DIS_s012
                                 + 1.29952 w_k1_FDX_s012
                                 + 0.957979 w_k1_IBM_s012
                                 + 1.0404 w_k1_INTC_s012
                                 + 0.901063 w_k1_KO_s012
                                 + 0.96504 w_k1_MAT_s012
                                 + 1.31673 w_k1_MSFT_s012
                                 + 1.17491 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.15389 w_k1_ADM_s013
                                 + 1.25109 w_k1_AXP_s013
                                 + 1.35868 w_k1_DIS_s013
                                 + 1.29599 w_k1_FDX_s013
                                 + 0.835973 w_k1_IBM_s013
                                 + 1.054 w_k1_INTC_s013 + 1.07914 w_k1_KO_s013
                                 + 0.774477 w_k1_MAT_s013
                                 + 1.22582 w_k1_MSFT_s013
                                 + 1.25885 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.775302 w_k1_ADM_s013
                                 + 0.721751 w_k1_AXP_s013
                                 + 0.669205 w_k1_DIS_s013
                                 + 0.704017 w_k1_FDX_s013
                                 + 1.12189 w_k1_IBM_s013
                                 + 0.810827 w_k1_INTC_s013
                                 + 0.771827 w_k1_KO_s013
                                 + 0.902411 w_k1_MAT_s013
                                 + 0.833985 w_k1_MSFT_s013
                                 + 0.965909 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.12764 w_k1_ADM_s013
                                 + 0.854861 w_k1_AXP_s013
                                 + 0.723741 w_k1_DIS_s013
                                 + 0.895786 w_k1_FDX_s013
                                 + 0.61435 w_k1_IBM_s013
                                 + 0.789189 w_k1_INTC_s013
                                 + 0.657829 w_k1_KO_s013
                                 + 0.73589 w_k1_MAT_s013
                                 + 0.938009 w_k1_MSFT_s013
                                 + 0.911738 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.63361 w_k1_ADM_s013
                                 + 1.65151 w_k1_AXP_s013
                                 + 1.33127 w_k1_DIS_s013
                                 + 1.24098 w_k1_FDX_s013
                                 + 1.26936 w_k1_IBM_s013
                                 + 1.10275 w_k1_INTC_s013
                                 + 1.22896 w_k1_KO_s013 + 1.70177 w_k1_MAT_s013
                                 + 1.1782 w_k1_MSFT_s013 + 1.40213 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.10254 w_k1_ADM_s013
                                 + 1.15869 w_k1_AXP_s013
                                 + 0.911166 w_k1_DIS_s013
                                 + 0.894725 w_k1_FDX_s013
                                 + 1.23834 w_k1_IBM_s013
                                 + 1.04238 w_k1_INTC_s013
                                 + 1.09697 w_k1_KO_s013 + 1.31991 w_k1_MAT_s013
                                 + 1.05834 w_k1_MSFT_s013
                                 + 1.04499 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.05261 w_k1_ADM_s013
                                 + 1.08501 w_k1_AXP_s013
                                 + 1.10787 w_k1_DIS_s013
                                 + 0.892743 w_k1_FDX_s013
                                 + 1.21175 w_k1_IBM_s013
                                 + 0.846524 w_k1_INTC_s013
                                 + 1.16374 w_k1_KO_s013 + 1.17119 w_k1_MAT_s013
                                 + 1.02057 w_k1_MSFT_s013
                                 + 1.02122 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.15092 w_k1_ADM_s013
                                 + 1.15954 w_k1_AXP_s013
                                 + 1.18772 w_k1_DIS_s013
                                 + 0.974641 w_k1_FDX_s013
                                 + 1.81227 w_k1_IBM_s013
                                 + 0.881587 w_k1_INTC_s013
                                 + 1.21772 w_k1_KO_s013 + 1.90845 w_k1_MAT_s013
                                 + 1.0029 w_k1_MSFT_s013 + 1.21392 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.14149 w_k1_ADM_s013
                                 + 0.875393 w_k1_AXP_s013
                                 + 0.771596 w_k1_DIS_s013
                                 + 0.942334 w_k1_FDX_s013
                                 + 0.859883 w_k1_IBM_s013
                                 + 0.872486 w_k1_INTC_s013
                                 + 0.810867 w_k1_KO_s013
                                 + 0.774053 w_k1_MAT_s013
                                 + 0.944848 w_k1_MSFT_s013
                                 + 0.91427 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.932817 w_k1_ADM_s014
                                 + 0.89639 w_k1_AXP_s014
                                 + 0.707573 w_k1_DIS_s014
                                 + 0.818062 w_k1_FDX_s014
                                 + 1.28317 w_k1_IBM_s014
                                 + 1.0204 w_k1_INTC_s014 + 1.04406 w_k1_KO_s014
                                 + 0.828092 w_k1_MAT_s014
                                 + 0.821364 w_k1_MSFT_s014
                                 + 0.949903 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.917767 w_k1_ADM_s014
                                 + 0.945421 w_k1_AXP_s014
                                 + 0.628868 w_k1_DIS_s014
                                 + 0.854196 w_k1_FDX_s014
                                 + 0.838591 w_k1_IBM_s014
                                 + 0.79246 w_k1_INTC_s014
                                 + 0.83817 w_k1_KO_s014
                                 + 0.750455 w_k1_MAT_s014
                                 + 0.741268 w_k1_MSFT_s014
                                 + 0.856253 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.29165 w_k1_ADM_s014
                                 + 1.29069 w_k1_AXP_s014
                                 + 1.48745 w_k1_DIS_s014
                                 + 1.05662 w_k1_FDX_s014
                                 + 2.03143 w_k1_IBM_s014
                                 + 1.31972 w_k1_INTC_s014 + 1.4935 w_k1_KO_s014
                                 + 1.17236 w_k1_MAT_s014
                                 + 1.0217 w_k1_MSFT_s014 + 1.13501 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.45924 w_k1_ADM_s014
                                 + 1.46272 w_k1_AXP_s014
                                 + 1.62295 w_k1_DIS_s014
                                 + 1.42933 w_k1_FDX_s014
                                 + 1.61572 w_k1_IBM_s014
                                 + 1.48549 w_k1_INTC_s014
                                 + 1.48417 w_k1_KO_s014 + 1.24495 w_k1_MAT_s014
                                 + 1.35406 w_k1_MSFT_s014
                                 + 1.21217 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.0457 w_k1_ADM_s014
                                 + 0.977435 w_k1_AXP_s014
                                 + 1.06702 w_k1_DIS_s014
                                 + 1.09314 w_k1_FDX_s014
                                 + 0.922766 w_k1_IBM_s014
                                 + 0.942249 w_k1_INTC_s014
                                 + 1.01996 w_k1_KO_s014
                                 + 0.734668 w_k1_MAT_s014
                                 + 0.987974 w_k1_MSFT_s014
                                 + 0.95325 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.07885 w_k1_ADM_s014
                                 + 1.51747 w_k1_AXP_s014
                                 + 1.14504 w_k1_DIS_s014
                                 + 1.24156 w_k1_FDX_s014
                                 + 0.72819 w_k1_IBM_s014
                                 + 1.3797 w_k1_INTC_s014
                                 + 0.792635 w_k1_KO_s014
                                 + 0.701025 w_k1_MAT_s014
                                 + 0.989198 w_k1_MSFT_s014
                                 + 0.968147 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.917292 w_k1_ADM_s014
                                 + 1.15171 w_k1_AXP_s014
                                 + 1.11314 w_k1_DIS_s014
                                 + 1.05087 w_k1_FDX_s014
                                 + 1.22545 w_k1_IBM_s014
                                 + 1.12616 w_k1_INTC_s014
                                 + 1.02425 w_k1_KO_s014 + 0.93621 w_k1_MAT_s014
                                 + 1.02951 w_k1_MSFT_s014
                                 + 0.911531 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.03769 w_k1_ADM_s014
                                 + 1.06784 w_k1_AXP_s014
                                 + 1.49426 w_k1_DIS_s014
                                 + 1.47562 w_k1_FDX_s014
                                 + 0.683803 w_k1_IBM_s014
                                 + 1.24719 w_k1_INTC_s014
                                 + 0.98242 w_k1_KO_s014
                                 + 0.500704 w_k1_MAT_s014
                                 + 1.30512 w_k1_MSFT_s014
                                 + 1.11386 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.866993 w_k1_ADM_s015
                                 + 1.07536 w_k1_AXP_s015
                                 + 1.55064 w_k1_DIS_s015
                                 + 1.28339 w_k1_FDX_s015
                                 + 0.910809 w_k1_IBM_s015
                                 + 0.887297 w_k1_INTC_s015
                                 + 0.923877 w_k1_KO_s015
                                 + 0.947222 w_k1_MAT_s015
                                 + 1.12521 w_k1_MSFT_s015
                                 + 1.12657 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.785999 w_k1_ADM_s015
                                 + 0.709548 w_k1_AXP_s015
                                 + 0.544735 w_k1_DIS_s015
                                 + 0.688718 w_k1_FDX_s015
                                 + 0.957891 w_k1_IBM_s015
                                 + 0.661562 w_k1_INTC_s015
                                 + 0.586749 w_k1_KO_s015
                                 + 0.770861 w_k1_MAT_s015
                                 + 0.658334 w_k1_MSFT_s015
                                 + 0.776993 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.630733 w_k1_ADM_s015
                                 + 0.6374 w_k1_AXP_s015
                                 + 0.673844 w_k1_DIS_s015
                                 + 0.824493 w_k1_FDX_s015
                                 + 0.817954 w_k1_IBM_s015
                                 + 0.712571 w_k1_INTC_s015
                                 + 0.630255 w_k1_KO_s015
                                 + 0.610009 w_k1_MAT_s015
                                 + 0.792432 w_k1_MSFT_s015
                                 + 0.753974 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.934217 w_k1_ADM_s015
                                 + 1.21331 w_k1_AXP_s015
                                 + 1.07404 w_k1_DIS_s015
                                 + 1.04082 w_k1_FDX_s015
                                 + 1.36767 w_k1_IBM_s015
                                 + 0.963651 w_k1_INTC_s015
                                 + 0.981793 w_k1_KO_s015
                                 + 1.12512 w_k1_MAT_s015
                                 + 1.02982 w_k1_MSFT_s015
                                 + 0.966981 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.24934 w_k1_ADM_s015
                                 + 1.51417 w_k1_AXP_s015 + 1.4539 w_k1_DIS_s015
                                 + 1.32338 w_k1_FDX_s015
                                 + 1.60233 w_k1_IBM_s015
                                 + 1.02835 w_k1_INTC_s015
                                 + 1.14744 w_k1_KO_s015 + 1.09148 w_k1_MAT_s015
                                 + 1.02495 w_k1_MSFT_s015
                                 + 1.03064 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.964058 w_k1_ADM_s015
                                 + 1.14255 w_k1_AXP_s015
                                 + 0.807589 w_k1_DIS_s015
                                 + 0.951536 w_k1_FDX_s015
                                 + 1.08242 w_k1_IBM_s015
                                 + 0.686014 w_k1_INTC_s015
                                 + 0.748093 w_k1_KO_s015
                                 + 1.33736 w_k1_MAT_s015
                                 + 0.829828 w_k1_MSFT_s015
                                 + 0.896511 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.0209 w_k1_ADM_s015
                                 + 0.964868 w_k1_AXP_s015
                                 + 0.864103 w_k1_DIS_s015
                                 + 0.996155 w_k1_FDX_s015
                                 + 0.661159 w_k1_IBM_s015
                                 + 1.04437 w_k1_INTC_s015
                                 + 0.659235 w_k1_KO_s015
                                 + 0.486051 w_k1_MAT_s015
                                 + 1.05291 w_k1_MSFT_s015
                                 + 0.924099 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.7749 w_k1_ADM_s015
                                 + 0.773965 w_k1_AXP_s015
                                 + 0.838133 w_k1_DIS_s015
                                 + 0.905484 w_k1_FDX_s015
                                 + 1.0327 w_k1_IBM_s015
                                 + 0.801828 w_k1_INTC_s015
                                 + 0.815855 w_k1_KO_s015
                                 + 0.974066 w_k1_MAT_s015
                                 + 0.897625 w_k1_MSFT_s015
                                 + 0.946871 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.855577 w_k1_ADM_s016
                                 + 0.809425 w_k1_AXP_s016
                                 + 0.663252 w_k1_DIS_s016
                                 + 0.844081 w_k1_FDX_s016
                                 + 0.962215 w_k1_IBM_s016
                                 + 0.936826 w_k1_INTC_s016
                                 + 0.807807 w_k1_KO_s016
                                 + 1.02153 w_k1_MAT_s016
                                 + 0.810047 w_k1_MSFT_s016
                                 + 0.728826 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.05908 w_k1_ADM_s016
                                 + 0.831662 w_k1_AXP_s016
                                 + 1.08471 w_k1_DIS_s016
                                 + 0.959881 w_k1_FDX_s016
                                 + 0.747371 w_k1_IBM_s016
                                 + 0.760755 w_k1_INTC_s016
                                 + 0.832907 w_k1_KO_s016
                                 + 0.828415 w_k1_MAT_s016
                                 + 0.933738 w_k1_MSFT_s016
                                 + 0.959488 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.07089 w_k1_ADM_s016
                                 + 0.685086 w_k1_AXP_s016
                                 + 0.498304 w_k1_DIS_s016
                                 + 0.652991 w_k1_FDX_s016
                                 + 0.899329 w_k1_IBM_s016
                                 + 0.57794 w_k1_INTC_s016
                                 + 0.756747 w_k1_KO_s016
                                 + 0.711247 w_k1_MAT_s016
                                 + 0.631758 w_k1_MSFT_s016
                                 + 0.815159 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.868755 w_k1_ADM_s016
                                 + 1.14532 w_k1_AXP_s016 + 1.0321 w_k1_DIS_s016
                                 + 1.03746 w_k1_FDX_s016
                                 + 1.11406 w_k1_IBM_s016
                                 + 0.893551 w_k1_INTC_s016
                                 + 0.972816 w_k1_KO_s016
                                 + 1.13552 w_k1_MAT_s016
                                 + 0.912557 w_k1_MSFT_s016
                                 + 1.00169 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.762281 w_k1_ADM_s016
                                 + 0.616258 w_k1_AXP_s016
                                 + 0.658519 w_k1_DIS_s016
                                 + 0.900734 w_k1_FDX_s016
                                 + 0.704356 w_k1_IBM_s016
                                 + 0.676347 w_k1_INTC_s016
                                 + 0.596645 w_k1_KO_s016
                                 + 0.639446 w_k1_MAT_s016
                                 + 0.704578 w_k1_MSFT_s016
                                 + 0.714015 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.686334 w_k1_ADM_s016
                                 + 0.703008 w_k1_AXP_s016
                                 + 0.630833 w_k1_DIS_s016
                                 + 0.725073 w_k1_FDX_s016
                                 + 1.18148 w_k1_IBM_s016
                                 + 0.774755 w_k1_INTC_s016
                                 + 0.865034 w_k1_KO_s016
                                 + 1.13594 w_k1_MAT_s016
                                 + 0.774849 w_k1_MSFT_s016
                                 + 0.780405 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.22411 w_k1_ADM_s016
                                 + 1.50119 w_k1_AXP_s016
                                 + 1.60764 w_k1_DIS_s016
                                 + 1.34043 w_k1_FDX_s016
                                 + 1.67755 w_k1_IBM_s016
                                 + 1.09597 w_k1_INTC_s016
                                 + 1.32724 w_k1_KO_s016 + 1.2758 w_k1_MAT_s016
                                 + 0.977506 w_k1_MSFT_s016
                                 + 1.22758 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.25165 w_k1_ADM_s016
                                 + 1.16255 w_k1_AXP_s016 + 1.5994 w_k1_DIS_s016
                                 + 1.27229 w_k1_FDX_s016
                                 + 1.03994 w_k1_IBM_s016
                                 + 1.07052 w_k1_INTC_s016
                                 + 1.25458 w_k1_KO_s016 + 0.98268 w_k1_MAT_s016
                                 + 1.19515 w_k1_MSFT_s016
                                 + 1.13054 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.783548 w_k1_ADM_s017
                                 + 0.606103 w_k1_AXP_s017
                                 + 0.635088 w_k1_DIS_s017
                                 + 0.817658 w_k1_FDX_s017
                                 + 0.571622 w_k1_IBM_s017
                                 + 1.11156 w_k1_INTC_s017
                                 + 0.709444 w_k1_KO_s017
                                 + 0.594454 w_k1_MAT_s017
                                 + 0.935667 w_k1_MSFT_s017
                                 + 0.864073 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.23127 w_k1_ADM_s017
                                 + 1.1792 w_k1_AXP_s017 + 1.1448 w_k1_DIS_s017
                                 + 1.02281 w_k1_FDX_s017
                                 + 1.12647 w_k1_IBM_s017
                                 + 1.25594 w_k1_INTC_s017
                                 + 1.10374 w_k1_KO_s017 + 1.01957 w_k1_MAT_s017
                                 + 1.05899 w_k1_MSFT_s017 + 1.2069 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.21986 w_k1_ADM_s017
                                 + 1.07205 w_k1_AXP_s017
                                 + 1.18867 w_k1_DIS_s017 + 1.1409 w_k1_FDX_s017
                                 + 0.906852 w_k1_IBM_s017
                                 + 1.12069 w_k1_INTC_s017
                                 + 1.00488 w_k1_KO_s017 + 1.12237 w_k1_MAT_s017
                                 + 1.13324 w_k1_MSFT_s017
                                 + 1.05946 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.86705 w_k1_ADM_s017
                                 + 0.826044 w_k1_AXP_s017
                                 + 0.671421 w_k1_DIS_s017
                                 + 0.731322 w_k1_FDX_s017
                                 + 1.01444 w_k1_IBM_s017
                                 + 0.789388 w_k1_INTC_s017
                                 + 0.845567 w_k1_KO_s017
                                 + 1.10602 w_k1_MAT_s017
                                 + 0.78098 w_k1_MSFT_s017
                                 + 0.913772 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.4514 w_k1_ADM_s017
                                 + 1.23843 w_k1_AXP_s017
                                 + 1.73878 w_k1_DIS_s017
                                 + 1.48503 w_k1_FDX_s017
                                 + 0.954177 w_k1_IBM_s017
                                 + 1.6113 w_k1_INTC_s017 + 1.15451 w_k1_KO_s017
                                 + 0.804661 w_k1_MAT_s017
                                 + 1.33869 w_k1_MSFT_s017 + 1.1356 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.958808 w_k1_ADM_s017
                                 + 0.681121 w_k1_AXP_s017
                                 + 0.909033 w_k1_DIS_s017
                                 + 0.924277 w_k1_FDX_s017
                                 + 0.787662 w_k1_IBM_s017
                                 + 0.928115 w_k1_INTC_s017
                                 + 0.922162 w_k1_KO_s017
                                 + 0.729122 w_k1_MAT_s017
                                 + 0.980708 w_k1_MSFT_s017
                                 + 0.950234 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.788022 w_k1_ADM_s017
                                 + 0.805332 w_k1_AXP_s017
                                 + 0.598519 w_k1_DIS_s017
                                 + 0.751968 w_k1_FDX_s017
                                 + 0.877003 w_k1_IBM_s017
                                 + 0.841923 w_k1_INTC_s017
                                 + 0.714348 w_k1_KO_s017
                                 + 0.860312 w_k1_MAT_s017
                                 + 0.694287 w_k1_MSFT_s017
                                 + 0.834439 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.24842 w_k1_ADM_s017
                                 + 0.931443 w_k1_AXP_s017
                                 + 0.957493 w_k1_DIS_s017
                                 + 0.86258 w_k1_FDX_s017
                                 + 1.46555 w_k1_IBM_s017
                                 + 1.04755 w_k1_INTC_s017
                                 + 1.08442 w_k1_KO_s017 + 1.09221 w_k1_MAT_s017
                                 + 1.01672 w_k1_MSFT_s017
                                 + 0.909965 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.41498 w_k1_ADM_s100
                                 + 1.37354 w_k1_AXP_s100
                                 + 1.60227 w_k1_DIS_s100
                                 + 1.36315 w_k1_FDX_s100
                                 + 1.26529 w_k1_IBM_s100
                                 + 1.36382 w_k1_INTC_s100
                                 + 1.47813 w_k1_KO_s100 + 1.60238 w_k1_MAT_s100
                                 + 1.48033 w_k1_MSFT_s100
                                 + 1.16393 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.920536 w_k1_ADM_s100
                                 + 0.823291 w_k1_AXP_s100
                                 + 0.894859 w_k1_DIS_s100
                                 + 0.833524 w_k1_FDX_s100
                                 + 1.12706 w_k1_IBM_s100
                                 + 0.913359 w_k1_INTC_s100
                                 + 0.953804 w_k1_KO_s100
                                 + 0.859875 w_k1_MAT_s100
                                 + 0.968853 w_k1_MSFT_s100
                                 + 0.988302 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.771668 w_k1_ADM_s100
                                 + 0.575371 w_k1_AXP_s100
                                 + 0.651008 w_k1_DIS_s100
                                 + 0.767555 w_k1_FDX_s100
                                 + 0.606894 w_k1_IBM_s100
                                 + 0.847136 w_k1_INTC_s100
                                 + 0.739793 w_k1_KO_s100
                                 + 0.478275 w_k1_MAT_s100
                                 + 0.960267 w_k1_MSFT_s100
                                 + 0.695014 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.25114 w_k1_ADM_s100
                                 + 1.19979 w_k1_AXP_s100
                                 + 1.15598 w_k1_DIS_s100
                                 + 1.11914 w_k1_FDX_s100
                                 + 0.905485 w_k1_IBM_s100
                                 + 1.1184 w_k1_INTC_s100
                                 + 0.992757 w_k1_KO_s100
                                 + 0.808117 w_k1_MAT_s100
                                 + 1.16749 w_k1_MSFT_s100
                                 + 0.956246 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.830469 w_k1_ADM_s100
                                 + 0.866212 w_k1_AXP_s100
                                 + 0.673796 w_k1_DIS_s100
                                 + 0.81204 w_k1_FDX_s100
                                 + 0.799415 w_k1_IBM_s100
                                 + 0.86127 w_k1_INTC_s100
                                 + 0.866835 w_k1_KO_s100
                                 + 0.753261 w_k1_MAT_s100
                                 + 0.941212 w_k1_MSFT_s100
                                 + 0.83492 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.19244 w_k1_ADM_s100
                                 + 1.18518 w_k1_AXP_s100
                                 + 1.17698 w_k1_DIS_s100
                                 + 1.11469 w_k1_FDX_s100
                                 + 1.18451 w_k1_IBM_s100
                                 + 1.28793 w_k1_INTC_s100
                                 + 1.20406 w_k1_KO_s100
                                 + 0.949876 w_k1_MAT_s100
                                 + 1.17042 w_k1_MSFT_s100 + 1.0734 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.06902 w_k1_ADM_s100
                                 + 1.04765 w_k1_AXP_s100
                                 + 0.922734 w_k1_DIS_s100
                                 + 0.98624 w_k1_FDX_s100
                                 + 0.996476 w_k1_IBM_s100
                                 + 1.05719 w_k1_INTC_s100
                                 + 1.03987 w_k1_KO_s100
                                 + 0.776879 w_k1_MAT_s100
                                 + 0.904832 w_k1_MSFT_s100
                                 + 0.882395 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.998172 w_k1_ADM_s100
                                 + 1.45072 w_k1_AXP_s100
                                 + 1.21873 w_k1_DIS_s100
                                 + 1.04099 w_k1_FDX_s100
                                 + 1.74564 w_k1_IBM_s100
                                 + 1.15834 w_k1_INTC_s100
                                 + 1.35484 w_k1_KO_s100 + 2.14808 w_k1_MAT_s100
                                 + 1.17563 w_k1_MSFT_s100
                                 + 1.11763 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.21717 w_k1_ADM_s101
                                 + 1.5436 w_k1_AXP_s101 + 1.58856 w_k1_DIS_s101
                                 + 1.35092 w_k1_FDX_s101
                                 + 1.48075 w_k1_IBM_s101
                                 + 1.26436 w_k1_INTC_s101 + 1.4025 w_k1_KO_s101
                                 + 2.31496 w_k1_MAT_s101
                                 + 1.22227 w_k1_MSFT_s101
                                 + 1.38441 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.914051 w_k1_ADM_s101
                                 + 1.05732 w_k1_AXP_s101
                                 + 0.83954 w_k1_DIS_s101
                                 + 0.844762 w_k1_FDX_s101
                                 + 1.28831 w_k1_IBM_s101
                                 + 0.779675 w_k1_INTC_s101
                                 + 1.06726 w_k1_KO_s101 + 1.26192 w_k1_MAT_s101
                                 + 0.781775 w_k1_MSFT_s101
                                 + 1.01161 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.991059 w_k1_ADM_s101
                                 + 1.02407 w_k1_AXP_s101
                                 + 0.979013 w_k1_DIS_s101
                                 + 1.0286 w_k1_FDX_s101
                                 + 0.919213 w_k1_IBM_s101
                                 + 0.99295 w_k1_INTC_s101
                                 + 0.971925 w_k1_KO_s101
                                 + 0.750946 w_k1_MAT_s101
                                 + 0.975028 w_k1_MSFT_s101
                                 + 1.05612 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.812581 w_k1_ADM_s101
                                 + 0.913616 w_k1_AXP_s101
                                 + 1.15242 w_k1_DIS_s101
                                 + 0.997996 w_k1_FDX_s101
                                 + 1.28025 w_k1_IBM_s101
                                 + 1.0899 w_k1_INTC_s101 + 1.22779 w_k1_KO_s101
                                 + 1.00148 w_k1_MAT_s101
                                 + 1.10256 w_k1_MSFT_s101
                                 + 1.08332 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.841607 w_k1_ADM_s101
                                 + 0.897064 w_k1_AXP_s101
                                 + 0.77674 w_k1_DIS_s101
                                 + 0.889885 w_k1_FDX_s101
                                 + 0.767213 w_k1_IBM_s101
                                 + 0.709233 w_k1_INTC_s101
                                 + 0.79686 w_k1_KO_s101
                                 + 0.939608 w_k1_MAT_s101
                                 + 0.792981 w_k1_MSFT_s101
                                 + 0.912877 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.758839 w_k1_ADM_s101
                                 + 0.993903 w_k1_AXP_s101
                                 + 0.993434 w_k1_DIS_s101
                                 + 0.814201 w_k1_FDX_s101
                                 + 1.82698 w_k1_IBM_s101
                                 + 1.05477 w_k1_INTC_s101
                                 + 1.39077 w_k1_KO_s101 + 1.54292 w_k1_MAT_s101
                                 + 0.866257 w_k1_MSFT_s101
                                 + 1.06615 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.993862 w_k1_ADM_s101
                                 + 1.24849 w_k1_AXP_s101 + 1.2012 w_k1_DIS_s101
                                 + 1.20932 w_k1_FDX_s101
                                 + 0.873871 w_k1_IBM_s101
                                 + 1.13473 w_k1_INTC_s101
                                 + 1.07714 w_k1_KO_s101 + 1.0114 w_k1_MAT_s101
                                 + 1.25672 w_k1_MSFT_s101
                                 + 1.06232 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.568177 w_k1_ADM_s101
                                 + 0.638328 w_k1_AXP_s101
                                 + 0.724064 w_k1_DIS_s101
                                 + 0.752041 w_k1_FDX_s101
                                 + 0.809147 w_k1_IBM_s101
                                 + 0.71557 w_k1_INTC_s101 + 0.7888 w_k1_KO_s101
                                 + 0.726018 w_k1_MAT_s101
                                 + 0.794016 w_k1_MSFT_s101
                                 + 0.770192 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02983 w_k1_ADM_s102
                                 + 0.743816 w_k1_AXP_s102
                                 + 0.662845 w_k1_DIS_s102
                                 + 0.796006 w_k1_FDX_s102
                                 + 0.614129 w_k1_IBM_s102
                                 + 0.811737 w_k1_INTC_s102
                                 + 0.660203 w_k1_KO_s102
                                 + 0.465913 w_k1_MAT_s102
                                 + 0.807702 w_k1_MSFT_s102
                                 + 0.838598 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.34582 w_k1_ADM_s102
                                 + 1.27696 w_k1_AXP_s102
                                 + 1.11489 w_k1_DIS_s102
                                 + 1.21411 w_k1_FDX_s102
                                 + 0.922721 w_k1_IBM_s102
                                 + 1.08983 w_k1_INTC_s102
                                 + 1.00458 w_k1_KO_s102
                                 + 0.774761 w_k1_MAT_s102
                                 + 1.19206 w_k1_MSFT_s102 + 1.1323 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.724135 w_k1_ADM_s102
                                 + 0.742154 w_k1_AXP_s102
                                 + 0.54704 w_k1_DIS_s102
                                 + 0.722842 w_k1_FDX_s102
                                 + 0.790857 w_k1_IBM_s102
                                 + 0.730638 w_k1_INTC_s102
                                 + 0.691765 w_k1_KO_s102
                                 + 0.883673 w_k1_MAT_s102
                                 + 0.807289 w_k1_MSFT_s102
                                 + 0.797886 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.24972 w_k1_ADM_s102
                                 + 1.62951 w_k1_AXP_s102 + 1.5784 w_k1_DIS_s102
                                 + 1.25062 w_k1_FDX_s102
                                 + 1.55781 w_k1_IBM_s102
                                 + 1.14768 w_k1_INTC_s102
                                 + 1.27397 w_k1_KO_s102 + 1.5949 w_k1_MAT_s102
                                 + 1.24614 w_k1_MSFT_s102 + 1.1471 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.926807 w_k1_ADM_s102
                                 + 0.993584 w_k1_AXP_s102
                                 + 0.838404 w_k1_DIS_s102
                                 + 0.947661 w_k1_FDX_s102
                                 + 0.994826 w_k1_IBM_s102
                                 + 1.0734 w_k1_INTC_s102
                                 + 0.911241 w_k1_KO_s102
                                 + 1.03857 w_k1_MAT_s102
                                 + 1.04413 w_k1_MSFT_s102
                                 + 1.07189 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04585 w_k1_ADM_s102
                                 + 0.918921 w_k1_AXP_s102
                                 + 0.779453 w_k1_DIS_s102
                                 + 0.874824 w_k1_FDX_s102
                                 + 0.797393 w_k1_IBM_s102
                                 + 1.06421 w_k1_INTC_s102
                                 + 0.905427 w_k1_KO_s102
                                 + 0.561308 w_k1_MAT_s102
                                 + 0.855864 w_k1_MSFT_s102
                                 + 0.93922 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.836292 w_k1_ADM_s102
                                 + 0.936767 w_k1_AXP_s102
                                 + 1.13602 w_k1_DIS_s102
                                 + 0.997726 w_k1_FDX_s102
                                 + 1.26563 w_k1_IBM_s102
                                 + 1.20893 w_k1_INTC_s102
                                 + 1.20826 w_k1_KO_s102
                                 + 0.991296 w_k1_MAT_s102
                                 + 1.12393 w_k1_MSFT_s102
                                 + 1.07323 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.64006 w_k1_ADM_s102
                                 + 1.53494 w_k1_AXP_s102
                                 + 1.77748 w_k1_DIS_s102
                                 + 1.51701 w_k1_FDX_s102
                                 + 1.16836 w_k1_IBM_s102
                                 + 1.23893 w_k1_INTC_s102
                                 + 1.27181 w_k1_KO_s102 + 1.11627 w_k1_MAT_s102
                                 + 1.49308 w_k1_MSFT_s102 + 1.1548 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.1955 w_k1_ADM_s103
                                 + 1.25738 w_k1_AXP_s103
                                 + 1.18382 w_k1_DIS_s103
                                 + 1.14478 w_k1_FDX_s103 + 1.3147 w_k1_IBM_s103
                                 + 1.51489 w_k1_INTC_s103
                                 + 1.48792 w_k1_KO_s103 + 1.29303 w_k1_MAT_s103
                                 + 1.36857 w_k1_MSFT_s103
                                 + 1.07273 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.819408 w_k1_ADM_s103
                                 + 0.608829 w_k1_AXP_s103
                                 + 0.957134 w_k1_DIS_s103
                                 + 0.900101 w_k1_FDX_s103
                                 + 0.80102 w_k1_IBM_s103
                                 + 0.989147 w_k1_INTC_s103
                                 + 0.928736 w_k1_KO_s103
                                 + 0.637673 w_k1_MAT_s103
                                 + 1.07427 w_k1_MSFT_s103
                                 + 0.935462 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.03089 w_k1_ADM_s103
                                 + 1.21205 w_k1_AXP_s103
                                 + 1.01183 w_k1_DIS_s103
                                 + 1.29722 w_k1_FDX_s103
                                 + 0.62628 w_k1_IBM_s103
                                 + 1.28645 w_k1_INTC_s103
                                 + 0.964305 w_k1_KO_s103
                                 + 0.616341 w_k1_MAT_s103
                                 + 1.21672 w_k1_MSFT_s103
                                 + 1.02303 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.09484 w_k1_ADM_s103
                                 + 1.30921 w_k1_AXP_s103
                                 + 1.70756 w_k1_DIS_s103
                                 + 1.30745 w_k1_FDX_s103
                                 + 1.15838 w_k1_IBM_s103
                                 + 1.54482 w_k1_INTC_s103 + 1.428 w_k1_KO_s103
                                 + 1.13018 w_k1_MAT_s103
                                 + 1.42269 w_k1_MSFT_s103
                                 + 1.23535 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.17422 w_k1_ADM_s103
                                 + 1.19861 w_k1_AXP_s103
                                 + 0.881459 w_k1_DIS_s103
                                 + 0.992718 w_k1_FDX_s103
                                 + 0.907579 w_k1_IBM_s103
                                 + 0.801572 w_k1_INTC_s103
                                 + 0.976816 w_k1_KO_s103
                                 + 1.18236 w_k1_MAT_s103
                                 + 0.865033 w_k1_MSFT_s103
                                 + 0.975741 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.670943 w_k1_ADM_s103
                                 + 0.894844 w_k1_AXP_s103
                                 + 0.863462 w_k1_DIS_s103
                                 + 0.88326 w_k1_FDX_s103
                                 + 1.21861 w_k1_IBM_s103
                                 + 1.0396 w_k1_INTC_s103
                                 + 0.964682 w_k1_KO_s103
                                 + 1.10108 w_k1_MAT_s103
                                 + 0.927686 w_k1_MSFT_s103
                                 + 0.926922 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.808179 w_k1_ADM_s103
                                 + 0.690509 w_k1_AXP_s103
                                 + 0.600582 w_k1_DIS_s103
                                 + 0.755496 w_k1_FDX_s103
                                 + 0.758034 w_k1_IBM_s103
                                 + 0.807581 w_k1_INTC_s103
                                 + 0.725668 w_k1_KO_s103
                                 + 0.700566 w_k1_MAT_s103
                                 + 0.816641 w_k1_MSFT_s103
                                 + 0.921429 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.806657 w_k1_ADM_s103
                                 + 0.965397 w_k1_AXP_s103
                                 + 1.01987 w_k1_DIS_s103
                                 + 0.967926 w_k1_FDX_s103
                                 + 0.776171 w_k1_IBM_s103
                                 + 0.936321 w_k1_INTC_s103
                                 + 0.959882 w_k1_KO_s103
                                 + 0.958649 w_k1_MAT_s103
                                 + 1.01653 w_k1_MSFT_s103
                                 + 1.08063 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.02428 w_k1_ADM_s104
                                 + 1.02652 w_k1_AXP_s104
                                 + 1.08846 w_k1_DIS_s104 + 1.1931 w_k1_FDX_s104
                                 + 0.637752 w_k1_IBM_s104
                                 + 1.24523 w_k1_INTC_s104
                                 + 0.749664 w_k1_KO_s104
                                 + 0.364321 w_k1_MAT_s104
                                 + 1.21628 w_k1_MSFT_s104
                                 + 0.836479 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.0516 w_k1_ADM_s104
                                 + 0.791415 w_k1_AXP_s104
                                 + 0.841912 w_k1_DIS_s104
                                 + 0.889584 w_k1_FDX_s104
                                 + 0.852719 w_k1_IBM_s104
                                 + 0.986777 w_k1_INTC_s104
                                 + 0.833491 w_k1_KO_s104
                                 + 0.659002 w_k1_MAT_s104
                                 + 0.911503 w_k1_MSFT_s104
                                 + 0.93822 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.975022 w_k1_ADM_s104
                                 + 0.868438 w_k1_AXP_s104
                                 + 1.19494 w_k1_DIS_s104
                                 + 0.992745 w_k1_FDX_s104
                                 + 0.910881 w_k1_IBM_s104
                                 + 1.1676 w_k1_INTC_s104 + 1.14404 w_k1_KO_s104
                                 + 0.745131 w_k1_MAT_s104
                                 + 1.2103 w_k1_MSFT_s104 + 1.01056 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.29571 w_k1_ADM_s104
                                 + 1.36893 w_k1_AXP_s104
                                 + 1.54209 w_k1_DIS_s104 + 1.4569 w_k1_FDX_s104
                                 + 1.11911 w_k1_IBM_s104
                                 + 1.29205 w_k1_INTC_s104
                                 + 1.16857 w_k1_KO_s104 + 1.13616 w_k1_MAT_s104
                                 + 1.31673 w_k1_MSFT_s104
                                 + 1.17938 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.11114 w_k1_ADM_s104
                                 + 1.10153 w_k1_AXP_s104
                                 + 1.19879 w_k1_DIS_s104 + 1.0687 w_k1_FDX_s104
                                 + 1.53633 w_k1_IBM_s104
                                 + 1.08271 w_k1_INTC_s104
                                 + 1.15118 w_k1_KO_s104 + 1.44017 w_k1_MAT_s104
                                 + 1.13508 w_k1_MSFT_s104
                                 + 1.15667 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.06517 w_k1_ADM_s104
                                 + 1.13485 w_k1_AXP_s104
                                 + 1.03504 w_k1_DIS_s104
                                 + 1.11674 w_k1_FDX_s104
                                 + 1.03022 w_k1_IBM_s104
                                 + 1.20454 w_k1_INTC_s104
                                 + 0.865693 w_k1_KO_s104
                                 + 0.907541 w_k1_MAT_s104
                                 + 1.01157 w_k1_MSFT_s104
                                 + 1.04307 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.921658 w_k1_ADM_s104
                                 + 0.881329 w_k1_AXP_s104
                                 + 0.688719 w_k1_DIS_s104
                                 + 0.758875 w_k1_FDX_s104
                                 + 1.40303 w_k1_IBM_s104
                                 + 0.961187 w_k1_INTC_s104
                                 + 0.87865 w_k1_KO_s104 + 1.00795 w_k1_MAT_s104
                                 + 0.844988 w_k1_MSFT_s104
                                 + 0.889616 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.934423 w_k1_ADM_s104
                                 + 1.08146 w_k1_AXP_s104
                                 + 0.743525 w_k1_DIS_s104
                                 + 0.955782 w_k1_FDX_s104
                                 + 0.976134 w_k1_IBM_s104
                                 + 0.889264 w_k1_INTC_s104
                                 + 0.852928 w_k1_KO_s104 + 1.2649 w_k1_MAT_s104
                                 + 0.924849 w_k1_MSFT_s104
                                 + 0.94859 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.814674 w_k1_ADM_s105
                                 + 0.772708 w_k1_AXP_s105
                                 + 0.634868 w_k1_DIS_s105
                                 + 0.801436 w_k1_FDX_s105
                                 + 0.816414 w_k1_IBM_s105
                                 + 0.883653 w_k1_INTC_s105
                                 + 0.71653 w_k1_KO_s105
                                 + 0.648129 w_k1_MAT_s105
                                 + 0.814556 w_k1_MSFT_s105
                                 + 0.902827 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.14225 w_k1_ADM_s105
                                 + 0.919863 w_k1_AXP_s105
                                 + 1.35189 w_k1_DIS_s105
                                 + 1.12955 w_k1_FDX_s105
                                 + 0.975886 w_k1_IBM_s105
                                 + 1.15567 w_k1_INTC_s105
                                 + 1.18179 w_k1_KO_s105
                                 + 0.706233 w_k1_MAT_s105
                                 + 1.07049 w_k1_MSFT_s105
                                 + 1.07301 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.4686 w_k1_ADM_s105
                                 + 1.61256 w_k1_AXP_s105
                                 + 1.21469 w_k1_DIS_s105
                                 + 1.32641 w_k1_FDX_s105
                                 + 1.08582 w_k1_IBM_s105
                                 + 1.20216 w_k1_INTC_s105
                                 + 1.17075 w_k1_KO_s105 + 1.34744 w_k1_MAT_s105
                                 + 1.13939 w_k1_MSFT_s105 + 1.2724 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.45539 w_k1_ADM_s105
                                 + 1.31443 w_k1_AXP_s105
                                 + 1.63188 w_k1_DIS_s105
                                 + 1.35282 w_k1_FDX_s105
                                 + 1.23381 w_k1_IBM_s105
                                 + 1.18064 w_k1_INTC_s105 + 1.1975 w_k1_KO_s105
                                 + 0.97079 w_k1_MAT_s105
                                 + 1.23125 w_k1_MSFT_s105
                                 + 1.28444 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.08378 w_k1_ADM_s105
                                 + 1.22212 w_k1_AXP_s105
                                 + 0.813286 w_k1_DIS_s105
                                 + 0.896325 w_k1_FDX_s105
                                 + 1.04484 w_k1_IBM_s105
                                 + 1.03789 w_k1_INTC_s105
                                 + 0.929173 w_k1_KO_s105
                                 + 0.92302 w_k1_MAT_s105
                                 + 0.837367 w_k1_MSFT_s105
                                 + 0.912567 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.29898 w_k1_ADM_s105
                                 + 1.07882 w_k1_AXP_s105
                                 + 0.828401 w_k1_DIS_s105
                                 + 1.11138 w_k1_FDX_s105
                                 + 0.741341 w_k1_IBM_s105
                                 + 1.07594 w_k1_INTC_s105
                                 + 0.809372 w_k1_KO_s105
                                 + 0.720155 w_k1_MAT_s105
                                 + 0.928152 w_k1_MSFT_s105
                                 + 1.01875 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.13225 w_k1_ADM_s105
                                 + 1.05159 w_k1_AXP_s105
                                 + 1.09879 w_k1_DIS_s105
                                 + 1.03561 w_k1_FDX_s105
                                 + 1.16751 w_k1_IBM_s105
                                 + 0.866564 w_k1_INTC_s105
                                 + 1.02696 w_k1_KO_s105 + 1.29959 w_k1_MAT_s105
                                 + 0.956727 w_k1_MSFT_s105
                                 + 1.10886 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.841941 w_k1_ADM_s105
                                 + 0.712018 w_k1_AXP_s105
                                 + 0.841108 w_k1_DIS_s105
                                 + 0.891811 w_k1_FDX_s105
                                 + 0.818511 w_k1_IBM_s105
                                 + 0.875366 w_k1_INTC_s105
                                 + 0.816752 w_k1_KO_s105
                                 + 0.810579 w_k1_MAT_s105
                                 + 0.921635 w_k1_MSFT_s105
                                 + 0.898693 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.677717 w_k1_ADM_s106
                                 + 0.598911 w_k1_AXP_s106
                                 + 0.646514 w_k1_DIS_s106
                                 + 0.758655 w_k1_FDX_s106
                                 + 0.621023 w_k1_IBM_s106
                                 + 0.929514 w_k1_INTC_s106
                                 + 0.817255 w_k1_KO_s106
                                 + 0.71502 w_k1_MAT_s106
                                 + 0.867654 w_k1_MSFT_s106
                                 + 0.828277 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.955112 w_k1_ADM_s106
                                 + 0.923358 w_k1_AXP_s106
                                 + 1.02436 w_k1_DIS_s106
                                 + 1.16867 w_k1_FDX_s106
                                 + 0.778928 w_k1_IBM_s106
                                 + 1.05921 w_k1_INTC_s106 + 1.0041 w_k1_KO_s106
                                 + 1.06378 w_k1_MAT_s106
                                 + 1.03364 w_k1_MSFT_s106
                                 + 1.09604 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.749688 w_k1_ADM_s106
                                 + 0.836707 w_k1_AXP_s106
                                 + 0.798719 w_k1_DIS_s106
                                 + 0.855572 w_k1_FDX_s106
                                 + 0.8351 w_k1_IBM_s106
                                 + 1.02462 w_k1_INTC_s106
                                 + 0.819042 w_k1_KO_s106
                                 + 0.833489 w_k1_MAT_s106
                                 + 0.875009 w_k1_MSFT_s106
                                 + 0.873475 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.05395 w_k1_ADM_s106
                                 + 1.21625 w_k1_AXP_s106
                                 + 1.06453 w_k1_DIS_s106
                                 + 1.05809 w_k1_FDX_s106
                                 + 1.13422 w_k1_IBM_s106
                                 + 1.16285 w_k1_INTC_s106
                                 + 1.03874 w_k1_KO_s106
                                 + 0.991581 w_k1_MAT_s106
                                 + 1.01761 w_k1_MSFT_s106
                                 + 1.09422 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.803517 w_k1_ADM_s106
                                 + 0.872612 w_k1_AXP_s106
                                 + 0.931072 w_k1_DIS_s106
                                 + 0.824591 w_k1_FDX_s106
                                 + 1.39602 w_k1_IBM_s106
                                 + 1.09818 w_k1_INTC_s106
                                 + 1.09292 w_k1_KO_s106 + 1.53347 w_k1_MAT_s106
                                 + 0.958949 w_k1_MSFT_s106
                                 + 0.967694 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.904942 w_k1_ADM_s106
                                 + 1.0013 w_k1_AXP_s106
                                 + 0.857986 w_k1_DIS_s106
                                 + 0.976926 w_k1_FDX_s106
                                 + 1.05707 w_k1_IBM_s106
                                 + 1.19313 w_k1_INTC_s106
                                 + 1.04352 w_k1_KO_s106
                                 + 0.947478 w_k1_MAT_s106
                                 + 0.981358 w_k1_MSFT_s106
                                 + 0.895988 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.769277 w_k1_ADM_s106
                                 + 0.869362 w_k1_AXP_s106
                                 + 0.669151 w_k1_DIS_s106
                                 + 0.681719 w_k1_FDX_s106
                                 + 1.16114 w_k1_IBM_s106
                                 + 0.670732 w_k1_INTC_s106
                                 + 0.823923 w_k1_KO_s106
                                 + 1.19829 w_k1_MAT_s106
                                 + 0.654978 w_k1_MSFT_s106
                                 + 0.877195 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.03576 w_k1_ADM_s106
                                 + 1.47051 w_k1_AXP_s106
                                 + 1.45545 w_k1_DIS_s106
                                 + 1.19618 w_k1_FDX_s106
                                 + 1.34031 w_k1_IBM_s106
                                 + 1.17043 w_k1_INTC_s106
                                 + 1.31251 w_k1_KO_s106 + 1.76663 w_k1_MAT_s106
                                 + 1.17447 w_k1_MSFT_s106
                                 + 1.11154 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13259 w_k1_ADM_s107
                                 + 0.775469 w_k1_AXP_s107
                                 + 0.871343 w_k1_DIS_s107
                                 + 1.01144 w_k1_FDX_s107
                                 + 0.700337 w_k1_IBM_s107
                                 + 0.946427 w_k1_INTC_s107
                                 + 0.829293 w_k1_KO_s107
                                 + 0.61139 w_k1_MAT_s107
                                 + 0.993848 w_k1_MSFT_s107
                                 + 0.908455 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.05381 w_k1_ADM_s107
                                 + 0.846289 w_k1_AXP_s107
                                 + 0.877767 w_k1_DIS_s107
                                 + 0.933395 w_k1_FDX_s107
                                 + 1.1274 w_k1_IBM_s107
                                 + 0.88614 w_k1_INTC_s107
                                 + 0.960328 w_k1_KO_s107 + 1.1315 w_k1_MAT_s107
                                 + 0.904561 w_k1_MSFT_s107
                                 + 1.00992 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.815694 w_k1_ADM_s107
                                 + 0.834464 w_k1_AXP_s107
                                 + 0.905537 w_k1_DIS_s107
                                 + 0.887384 w_k1_FDX_s107
                                 + 0.990362 w_k1_IBM_s107
                                 + 1.01353 w_k1_INTC_s107
                                 + 0.983426 w_k1_KO_s107
                                 + 0.754979 w_k1_MAT_s107
                                 + 1.103 w_k1_MSFT_s107 + 0.972005 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.838428 w_k1_ADM_s107
                                 + 0.663355 w_k1_AXP_s107
                                 + 0.729096 w_k1_DIS_s107
                                 + 0.808918 w_k1_FDX_s107
                                 + 1.03888 w_k1_IBM_s107
                                 + 0.879537 w_k1_INTC_s107
                                 + 0.827471 w_k1_KO_s107
                                 + 0.87646 w_k1_MAT_s107
                                 + 0.874531 w_k1_MSFT_s107
                                 + 0.87938 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.58052 w_k1_ADM_s107
                                 + 1.32664 w_k1_AXP_s107
                                 + 1.48682 w_k1_DIS_s107
                                 + 1.41234 w_k1_FDX_s107
                                 + 1.02646 w_k1_IBM_s107
                                 + 1.1254 w_k1_INTC_s107 + 1.13165 w_k1_KO_s107
                                 + 1.15758 w_k1_MAT_s107
                                 + 1.41315 w_k1_MSFT_s107
                                 + 1.06156 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.11311 w_k1_ADM_s107
                                 + 0.934179 w_k1_AXP_s107
                                 + 1.63937 w_k1_DIS_s107
                                 + 1.15078 w_k1_FDX_s107
                                 + 1.28204 w_k1_IBM_s107
                                 + 1.00499 w_k1_INTC_s107 + 1.2925 w_k1_KO_s107
                                 + 0.943084 w_k1_MAT_s107
                                 + 1.31878 w_k1_MSFT_s107
                                 + 1.05817 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.10604 w_k1_ADM_s107
                                 + 0.918452 w_k1_AXP_s107
                                 + 1.35109 w_k1_DIS_s107
                                 + 1.17105 w_k1_FDX_s107
                                 + 0.792432 w_k1_IBM_s107
                                 + 1.0637 w_k1_INTC_s107
                                 + 0.980989 w_k1_KO_s107
                                 + 0.659701 w_k1_MAT_s107
                                 + 1.20956 w_k1_MSFT_s107
                                 + 1.07611 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.24019 w_k1_ADM_s107
                                 + 1.10476 w_k1_AXP_s107
                                 + 1.17969 w_k1_DIS_s107
                                 + 1.05984 w_k1_FDX_s107
                                 + 1.09819 w_k1_IBM_s107
                                 + 0.93715 w_k1_INTC_s107 + 1.1442 w_k1_KO_s107
                                 + 1.29387 w_k1_MAT_s107
                                 + 1.12086 w_k1_MSFT_s107
                                 + 1.06148 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.61688 w_k1_ADM_s110
                                 + 1.68875 w_k1_AXP_s110
                                 + 1.60792 w_k1_DIS_s110
                                 + 1.35957 w_k1_FDX_s110
                                 + 1.85399 w_k1_IBM_s110
                                 + 1.10972 w_k1_INTC_s110
                                 + 1.52646 w_k1_KO_s110 + 2.21271 w_k1_MAT_s110
                                 + 1.29634 w_k1_MSFT_s110
                                 + 1.25469 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.22746 w_k1_ADM_s110
                                 + 1.16579 w_k1_AXP_s110
                                 + 1.17228 w_k1_DIS_s110
                                 + 1.10097 w_k1_FDX_s110
                                 + 0.871857 w_k1_IBM_s110
                                 + 1.11943 w_k1_INTC_s110
                                 + 1.11951 w_k1_KO_s110 + 1.02564 w_k1_MAT_s110
                                 + 1.27797 w_k1_MSFT_s110
                                 + 1.02041 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.01548 w_k1_ADM_s110
                                 + 0.588648 w_k1_AXP_s110
                                 + 0.520819 w_k1_DIS_s110
                                 + 0.698526 w_k1_FDX_s110
                                 + 0.611457 w_k1_IBM_s110
                                 + 0.562455 w_k1_INTC_s110
                                 + 0.690344 w_k1_KO_s110
                                 + 0.631361 w_k1_MAT_s110
                                 + 0.774216 w_k1_MSFT_s110
                                 + 0.787968 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.22501 w_k1_ADM_s110
                                 + 1.20116 w_k1_AXP_s110
                                 + 1.07122 w_k1_DIS_s110
                                 + 0.999573 w_k1_FDX_s110
                                 + 1.09049 w_k1_IBM_s110
                                 + 0.941265 w_k1_INTC_s110
                                 + 1.2004 w_k1_KO_s110 + 1.35453 w_k1_MAT_s110
                                 + 1.15483 w_k1_MSFT_s110
                                 + 0.985674 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.61948 w_k1_ADM_s110
                                 + 1.30088 w_k1_AXP_s110
                                 + 1.30085 w_k1_DIS_s110
                                 + 1.08717 w_k1_FDX_s110 + 1.476 w_k1_IBM_s110
                                 + 1.06004 w_k1_INTC_s110 + 1.4855 w_k1_KO_s110
                                 + 1.6639 w_k1_MAT_s110
                                 + 1.13962 w_k1_MSFT_s110 + 1.1173 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.31185 w_k1_ADM_s110
                                 + 0.880346 w_k1_AXP_s110
                                 + 0.997947 w_k1_DIS_s110
                                 + 0.966312 w_k1_FDX_s110
                                 + 0.586556 w_k1_IBM_s110
                                 + 0.815415 w_k1_INTC_s110
                                 + 0.804894 w_k1_KO_s110
                                 + 0.68473 w_k1_MAT_s110
                                 + 0.946206 w_k1_MSFT_s110
                                 + 0.913397 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.855219 w_k1_ADM_s110
                                 + 0.891931 w_k1_AXP_s110
                                 + 0.740938 w_k1_DIS_s110
                                 + 0.739677 w_k1_FDX_s110
                                 + 1.2152 w_k1_IBM_s110
                                 + 0.822978 w_k1_INTC_s110
                                 + 0.984341 w_k1_KO_s110
                                 + 1.25193 w_k1_MAT_s110
                                 + 0.89195 w_k1_MSFT_s110
                                 + 0.902564 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.13389 w_k1_ADM_s110
                                 + 0.807286 w_k1_AXP_s110
                                 + 0.76584 w_k1_DIS_s110
                                 + 0.864096 w_k1_FDX_s110
                                 + 0.737488 w_k1_IBM_s110
                                 + 0.737199 w_k1_INTC_s110
                                 + 0.725185 w_k1_KO_s110
                                 + 0.748213 w_k1_MAT_s110
                                 + 0.831041 w_k1_MSFT_s110
                                 + 0.824625 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.31353 w_k1_ADM_s111
                                 + 1.24012 w_k1_AXP_s111
                                 + 1.72239 w_k1_DIS_s111
                                 + 1.40047 w_k1_FDX_s111
                                 + 1.45937 w_k1_IBM_s111 + 1.513 w_k1_INTC_s111
                                 + 1.61709 w_k1_KO_s111 + 1.30143 w_k1_MAT_s111
                                 + 1.43573 w_k1_MSFT_s111
                                 + 1.32485 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.959276 w_k1_ADM_s111
                                 + 1.21624 w_k1_AXP_s111
                                 + 0.988905 w_k1_DIS_s111
                                 + 0.965126 w_k1_FDX_s111
                                 + 1.30054 w_k1_IBM_s111
                                 + 1.12124 w_k1_INTC_s111
                                 + 1.16071 w_k1_KO_s111 + 1.32406 w_k1_MAT_s111
                                 + 1.08519 w_k1_MSFT_s111
                                 + 0.997201 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.706475 w_k1_ADM_s111
                                 + 0.622652 w_k1_AXP_s111
                                 + 0.588852 w_k1_DIS_s111
                                 + 0.711301 w_k1_FDX_s111
                                 + 0.731021 w_k1_IBM_s111
                                 + 0.77171 w_k1_INTC_s111
                                 + 0.695575 w_k1_KO_s111
                                 + 0.594437 w_k1_MAT_s111
                                 + 0.776582 w_k1_MSFT_s111
                                 + 0.731884 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.903522 w_k1_ADM_s111
                                 + 1.00004 w_k1_AXP_s111
                                 + 1.21717 w_k1_DIS_s111
                                 + 0.971304 w_k1_FDX_s111
                                 + 1.84324 w_k1_IBM_s111
                                 + 1.1055 w_k1_INTC_s111 + 1.4373 w_k1_KO_s111
                                 + 1.47522 w_k1_MAT_s111
                                 + 1.17503 w_k1_MSFT_s111 + 1.1343 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.827457 w_k1_ADM_s111
                                 + 0.829034 w_k1_AXP_s111
                                 + 0.805719 w_k1_DIS_s111
                                 + 0.813213 w_k1_FDX_s111
                                 + 0.974311 w_k1_IBM_s111
                                 + 0.906957 w_k1_INTC_s111
                                 + 1.05176 w_k1_KO_s111 + 1.09287 w_k1_MAT_s111
                                 + 0.97697 w_k1_MSFT_s111
                                 + 0.953535 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.852687 w_k1_ADM_s111
                                 + 0.895935 w_k1_AXP_s111
                                 + 0.904992 w_k1_DIS_s111
                                 + 1.05625 w_k1_FDX_s111
                                 + 0.840377 w_k1_IBM_s111
                                 + 1.11416 w_k1_INTC_s111
                                 + 0.853679 w_k1_KO_s111
                                 + 0.651905 w_k1_MAT_s111
                                 + 0.989977 w_k1_MSFT_s111
                                 + 0.980066 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0409 w_k1_ADM_s111
                                 + 1.15207 w_k1_AXP_s111
                                 + 1.24389 w_k1_DIS_s111
                                 + 1.08218 w_k1_FDX_s111
                                 + 1.26015 w_k1_IBM_s111
                                 + 1.10044 w_k1_INTC_s111
                                 + 1.24247 w_k1_KO_s111 + 1.35794 w_k1_MAT_s111
                                 + 1.11171 w_k1_MSFT_s111
                                 + 1.14606 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.23144 w_k1_ADM_s111
                                 + 1.337 w_k1_AXP_s111 + 1.3878 w_k1_DIS_s111
                                 + 1.25788 w_k1_FDX_s111
                                 + 0.890637 w_k1_IBM_s111
                                 + 1.16332 w_k1_INTC_s111
                                 + 1.16859 w_k1_KO_s111
                                 + 0.751044 w_k1_MAT_s111
                                 + 1.27418 w_k1_MSFT_s111
                                 + 1.11241 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.02011 w_k1_ADM_s112
                                 + 0.795167 w_k1_AXP_s112
                                 + 0.800209 w_k1_DIS_s112
                                 + 0.835672 w_k1_FDX_s112
                                 + 1.01663 w_k1_IBM_s112
                                 + 0.818071 w_k1_INTC_s112
                                 + 0.961522 w_k1_KO_s112
                                 + 0.767756 w_k1_MAT_s112
                                 + 1.01856 w_k1_MSFT_s112
                                 + 1.06564 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.962892 w_k1_ADM_s112
                                 + 1.21566 w_k1_AXP_s112
                                 + 1.53854 w_k1_DIS_s112
                                 + 1.18586 w_k1_FDX_s112
                                 + 1.61828 w_k1_IBM_s112
                                 + 1.16358 w_k1_INTC_s112
                                 + 1.51094 w_k1_KO_s112 + 1.33729 w_k1_MAT_s112
                                 + 1.29935 w_k1_MSFT_s112
                                 + 1.16365 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.892255 w_k1_ADM_s112
                                 + 0.987816 w_k1_AXP_s112
                                 + 1.03361 w_k1_DIS_s112
                                 + 1.05169 w_k1_FDX_s112
                                 + 0.883263 w_k1_IBM_s112
                                 + 0.920452 w_k1_INTC_s112
                                 + 0.938046 w_k1_KO_s112 + 1.0509 w_k1_MAT_s112
                                 + 1.12857 w_k1_MSFT_s112
                                 + 0.969354 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.693447 w_k1_ADM_s112
                                 + 0.546332 w_k1_AXP_s112
                                 + 0.593038 w_k1_DIS_s112
                                 + 0.614123 w_k1_FDX_s112
                                 + 0.904059 w_k1_IBM_s112
                                 + 0.68559 w_k1_INTC_s112
                                 + 0.885085 w_k1_KO_s112
                                 + 0.694576 w_k1_MAT_s112
                                 + 0.789641 w_k1_MSFT_s112
                                 + 0.799459 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.691572 w_k1_ADM_s112
                                 + 0.548348 w_k1_AXP_s112
                                 + 0.472749 w_k1_DIS_s112
                                 + 0.606021 w_k1_FDX_s112
                                 + 0.681806 w_k1_IBM_s112
                                 + 0.639082 w_k1_INTC_s112
                                 + 0.606687 w_k1_KO_s112
                                 + 0.455685 w_k1_MAT_s112
                                 + 0.672358 w_k1_MSFT_s112
                                 + 0.793733 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.36023 w_k1_ADM_s112
                                 + 1.1838 w_k1_AXP_s112 + 1.54676 w_k1_DIS_s112
                                 + 1.34379 w_k1_FDX_s112
                                 + 0.871248 w_k1_IBM_s112
                                 + 1.2376 w_k1_INTC_s112 + 1.06743 w_k1_KO_s112
                                 + 0.90977 w_k1_MAT_s112
                                 + 1.50602 w_k1_MSFT_s112
                                 + 1.12505 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.28134 w_k1_ADM_s112
                                 + 1.17562 w_k1_AXP_s112
                                 + 1.55516 w_k1_DIS_s112
                                 + 1.21001 w_k1_FDX_s112
                                 + 1.15172 w_k1_IBM_s112
                                 + 1.01961 w_k1_INTC_s112
                                 + 1.18657 w_k1_KO_s112 + 1.0522 w_k1_MAT_s112
                                 + 1.31817 w_k1_MSFT_s112
                                 + 1.17411 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.8523 w_k1_ADM_s112
                                 + 0.693362 w_k1_AXP_s112
                                 + 0.845614 w_k1_DIS_s112
                                 + 0.921596 w_k1_FDX_s112
                                 + 0.673603 w_k1_IBM_s112
                                 + 0.882131 w_k1_INTC_s112
                                 + 0.807125 w_k1_KO_s112
                                 + 0.717524 w_k1_MAT_s112
                                 + 1.06231 w_k1_MSFT_s112
                                 + 0.88786 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.08839 w_k1_ADM_s113
                                 + 0.945495 w_k1_AXP_s113
                                 + 1.0658 w_k1_DIS_s113 + 1.13587 w_k1_FDX_s113
                                 + 0.894194 w_k1_IBM_s113
                                 + 1.0538 w_k1_INTC_s113
                                 + 0.983727 w_k1_KO_s113
                                 + 0.703303 w_k1_MAT_s113
                                 + 1.06736 w_k1_MSFT_s113
                                 + 1.00223 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.34631 w_k1_ADM_s113
                                 + 1.66015 w_k1_AXP_s113 + 1.4228 w_k1_DIS_s113
                                 + 1.46099 w_k1_FDX_s113 + 1.365 w_k1_IBM_s113
                                 + 1.28754 w_k1_INTC_s113
                                 + 1.27218 w_k1_KO_s113 + 1.28308 w_k1_MAT_s113
                                 + 1.26777 w_k1_MSFT_s113
                                 + 1.22695 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.685933 w_k1_ADM_s113
                                 + 0.589743 w_k1_AXP_s113
                                 + 0.623213 w_k1_DIS_s113
                                 + 0.737973 w_k1_FDX_s113
                                 + 0.852654 w_k1_IBM_s113
                                 + 0.797725 w_k1_INTC_s113
                                 + 0.695634 w_k1_KO_s113
                                 + 0.711576 w_k1_MAT_s113
                                 + 0.752434 w_k1_MSFT_s113
                                 + 0.781985 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.919187 w_k1_ADM_s113
                                 + 1.06733 w_k1_AXP_s113
                                 + 1.04786 w_k1_DIS_s113
                                 + 0.959321 w_k1_FDX_s113
                                 + 1.50186 w_k1_IBM_s113
                                 + 0.901549 w_k1_INTC_s113
                                 + 1.03587 w_k1_KO_s113 + 1.32215 w_k1_MAT_s113
                                 + 0.968514 w_k1_MSFT_s113
                                 + 0.971409 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.862368 w_k1_ADM_s113
                                 + 0.930502 w_k1_AXP_s113
                                 + 0.762167 w_k1_DIS_s113
                                 + 0.774763 w_k1_FDX_s113
                                 + 1.19034 w_k1_IBM_s113
                                 + 0.743288 w_k1_INTC_s113
                                 + 0.961958 w_k1_KO_s113 + 1.0147 w_k1_MAT_s113
                                 + 0.800515 w_k1_MSFT_s113
                                 + 0.91298 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.3432 w_k1_ADM_s113
                                 + 1.20626 w_k1_AXP_s113
                                 + 1.50127 w_k1_DIS_s113
                                 + 1.27284 w_k1_FDX_s113
                                 + 1.05427 w_k1_IBM_s113
                                 + 1.28335 w_k1_INTC_s113
                                 + 1.30247 w_k1_KO_s113 + 1.04033 w_k1_MAT_s113
                                 + 1.31973 w_k1_MSFT_s113 + 1.2038 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.991277 w_k1_ADM_s113
                                 + 0.835673 w_k1_AXP_s113
                                 + 0.947334 w_k1_DIS_s113
                                 + 0.916026 w_k1_FDX_s113
                                 + 0.927782 w_k1_IBM_s113
                                 + 1.04653 w_k1_INTC_s113
                                 + 0.986133 w_k1_KO_s113
                                 + 0.824116 w_k1_MAT_s113
                                 + 0.869628 w_k1_MSFT_s113
                                 + 0.929019 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.985135 w_k1_ADM_s113
                                 + 0.901572 w_k1_AXP_s113
                                 + 0.767815 w_k1_DIS_s113
                                 + 0.993325 w_k1_FDX_s113
                                 + 0.691457 w_k1_IBM_s113
                                 + 0.957308 w_k1_INTC_s113
                                 + 0.771403 w_k1_KO_s113
                                 + 0.630717 w_k1_MAT_s113
                                 + 0.779022 w_k1_MSFT_s113
                                 + 0.782178 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.988905 w_k1_ADM_s114
                                 + 1.0133 w_k1_AXP_s114 + 1.1424 w_k1_DIS_s114
                                 + 1.07605 w_k1_FDX_s114
                                 + 1.07986 w_k1_IBM_s114
                                 + 1.10533 w_k1_INTC_s114
                                 + 1.07458 w_k1_KO_s114 + 0.91478 w_k1_MAT_s114
                                 + 1.02514 w_k1_MSFT_s114
                                 + 1.00099 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.642579 w_k1_ADM_s114
                                 + 0.690046 w_k1_AXP_s114
                                 + 0.729368 w_k1_DIS_s114
                                 + 0.801934 w_k1_FDX_s114
                                 + 0.921386 w_k1_IBM_s114
                                 + 0.870163 w_k1_INTC_s114
                                 + 0.912914 w_k1_KO_s114
                                 + 0.647132 w_k1_MAT_s114
                                 + 0.828774 w_k1_MSFT_s114
                                 + 0.821402 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.772509 w_k1_ADM_s114
                                 + 0.743734 w_k1_AXP_s114
                                 + 0.901163 w_k1_DIS_s114
                                 + 0.995557 w_k1_FDX_s114
                                 + 0.680102 w_k1_IBM_s114
                                 + 0.954119 w_k1_INTC_s114
                                 + 0.845041 w_k1_KO_s114
                                 + 0.521115 w_k1_MAT_s114
                                 + 0.960734 w_k1_MSFT_s114
                                 + 0.792197 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.734298 w_k1_ADM_s114
                                 + 0.709193 w_k1_AXP_s114
                                 + 0.773134 w_k1_DIS_s114
                                 + 0.829199 w_k1_FDX_s114
                                 + 1.10818 w_k1_IBM_s114
                                 + 1.11043 w_k1_INTC_s114
                                 + 0.977443 w_k1_KO_s114
                                 + 0.813936 w_k1_MAT_s114
                                 + 0.917224 w_k1_MSFT_s114
                                 + 0.823551 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0312 w_k1_ADM_s114
                                 + 1.07605 w_k1_AXP_s114
                                 + 0.81064 w_k1_DIS_s114
                                 + 1.00487 w_k1_FDX_s114
                                 + 0.772542 w_k1_IBM_s114
                                 + 0.893295 w_k1_INTC_s114
                                 + 0.832533 w_k1_KO_s114
                                 + 0.723688 w_k1_MAT_s114
                                 + 0.923982 w_k1_MSFT_s114
                                 + 0.963653 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.976087 w_k1_ADM_s114
                                 + 1.03676 w_k1_AXP_s114
                                 + 0.862172 w_k1_DIS_s114
                                 + 0.985429 w_k1_FDX_s114
                                 + 1.13367 w_k1_IBM_s114
                                 + 0.826694 w_k1_INTC_s114
                                 + 0.906745 w_k1_KO_s114
                                 + 1.24807 w_k1_MAT_s114
                                 + 0.864901 w_k1_MSFT_s114
                                 + 0.937207 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.879808 w_k1_ADM_s114
                                 + 0.86308 w_k1_AXP_s114
                                 + 1.15152 w_k1_DIS_s114
                                 + 1.00373 w_k1_FDX_s114
                                 + 1.54581 w_k1_IBM_s114
                                 + 0.981486 w_k1_INTC_s114
                                 + 1.30551 w_k1_KO_s114 + 1.46868 w_k1_MAT_s114
                                 + 0.969364 w_k1_MSFT_s114 + 1.016 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.22468 w_k1_ADM_s114
                                 + 1.49946 w_k1_AXP_s114
                                 + 1.42544 w_k1_DIS_s114 + 1.3602 w_k1_FDX_s114
                                 + 1.31805 w_k1_IBM_s114
                                 + 1.17641 w_k1_INTC_s114
                                 + 1.24853 w_k1_KO_s114 + 1.68767 w_k1_MAT_s114
                                 + 1.17962 w_k1_MSFT_s114
                                 + 1.06555 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.17388 w_k1_ADM_s115
                                 + 1.17188 w_k1_AXP_s115
                                 + 1.54733 w_k1_DIS_s115 + 1.4686 w_k1_FDX_s115
                                 + 1.05099 w_k1_IBM_s115
                                 + 1.20525 w_k1_INTC_s115
                                 + 1.27362 w_k1_KO_s115 + 0.97412 w_k1_MAT_s115
                                 + 1.4153 w_k1_MSFT_s115 + 1.12754 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.799254 w_k1_ADM_s115
                                 + 0.803552 w_k1_AXP_s115
                                 + 0.723737 w_k1_DIS_s115
                                 + 0.880643 w_k1_FDX_s115
                                 + 0.708439 w_k1_IBM_s115
                                 + 0.909166 w_k1_INTC_s115
                                 + 0.735723 w_k1_KO_s115
                                 + 0.590381 w_k1_MAT_s115
                                 + 0.802141 w_k1_MSFT_s115
                                 + 0.832459 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.00048 w_k1_ADM_s115
                                 + 1.07959 w_k1_AXP_s115
                                 + 1.16508 w_k1_DIS_s115
                                 + 1.24867 w_k1_FDX_s115
                                 + 0.788154 w_k1_IBM_s115
                                 + 1.33666 w_k1_INTC_s115
                                 + 1.02104 w_k1_KO_s115
                                 + 0.699562 w_k1_MAT_s115
                                 + 1.25422 w_k1_MSFT_s115
                                 + 0.946278 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.904477 w_k1_ADM_s115
                                 + 1.01796 w_k1_AXP_s115
                                 + 0.984328 w_k1_DIS_s115
                                 + 1.05202 w_k1_FDX_s115
                                 + 0.966156 w_k1_IBM_s115
                                 + 0.956726 w_k1_INTC_s115
                                 + 0.991908 w_k1_KO_s115
                                 + 1.00053 w_k1_MAT_s115
                                 + 0.986256 w_k1_MSFT_s115
                                 + 0.993551 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.784802 w_k1_ADM_s115
                                 + 1.02195 w_k1_AXP_s115
                                 + 0.610692 w_k1_DIS_s115
                                 + 0.81789 w_k1_FDX_s115
                                 + 1.08033 w_k1_IBM_s115
                                 + 0.86277 w_k1_INTC_s115
                                 + 0.888989 w_k1_KO_s115 + 1.1024 w_k1_MAT_s115
                                 + 0.785563 w_k1_MSFT_s115
                                 + 0.874598 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.16187 w_k1_ADM_s115
                                 + 1.31201 w_k1_AXP_s115
                                 + 1.19418 w_k1_DIS_s115
                                 + 1.16095 w_k1_FDX_s115
                                 + 1.47502 w_k1_IBM_s115
                                 + 1.10995 w_k1_INTC_s115 + 1.2942 w_k1_KO_s115
                                 + 1.28823 w_k1_MAT_s115
                                 + 1.21249 w_k1_MSFT_s115
                                 + 1.06077 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.08938 w_k1_ADM_s115
                                 + 1.00391 w_k1_AXP_s115
                                 + 0.652118 w_k1_DIS_s115
                                 + 0.976931 w_k1_FDX_s115
                                 + 1.17654 w_k1_IBM_s115
                                 + 0.980406 w_k1_INTC_s115
                                 + 0.969094 w_k1_KO_s115
                                 + 0.977148 w_k1_MAT_s115
                                 + 0.854844 w_k1_MSFT_s115
                                 + 0.958687 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.790573 w_k1_ADM_s115
                                 + 0.803128 w_k1_AXP_s115
                                 + 0.962914 w_k1_DIS_s115
                                 + 0.936926 w_k1_FDX_s115
                                 + 1.25884 w_k1_IBM_s115
                                 + 1.06411 w_k1_INTC_s115
                                 + 1.05914 w_k1_KO_s115
                                 + 0.993174 w_k1_MAT_s115
                                 + 1.05411 w_k1_MSFT_s115
                                 + 1.03495 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.12664 w_k1_ADM_s116
                                 + 1.01649 w_k1_AXP_s116
                                 + 0.764752 w_k1_DIS_s116
                                 + 0.950895 w_k1_FDX_s116
                                 + 1.01703 w_k1_IBM_s116
                                 + 0.732499 w_k1_INTC_s116
                                 + 0.781106 w_k1_KO_s116
                                 + 1.09101 w_k1_MAT_s116
                                 + 0.845533 w_k1_MSFT_s116
                                 + 0.919948 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.851152 w_k1_ADM_s116
                                 + 0.637905 w_k1_AXP_s116
                                 + 0.7266 w_k1_DIS_s116
                                 + 0.857781 w_k1_FDX_s116
                                 + 0.719748 w_k1_IBM_s116
                                 + 0.75063 w_k1_INTC_s116
                                 + 0.733415 w_k1_KO_s116
                                 + 0.606797 w_k1_MAT_s116
                                 + 0.837834 w_k1_MSFT_s116
                                 + 0.932356 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.989207 w_k1_ADM_s116
                                 + 1.00522 w_k1_AXP_s116
                                 + 0.757667 w_k1_DIS_s116
                                 + 0.89562 w_k1_FDX_s116
                                 + 1.16849 w_k1_IBM_s116
                                 + 0.759796 w_k1_INTC_s116
                                 + 1.03576 w_k1_KO_s116 + 1.38874 w_k1_MAT_s116
                                 + 0.820538 w_k1_MSFT_s116
                                 + 0.959838 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.994751 w_k1_ADM_s116
                                 + 0.915579 w_k1_AXP_s116
                                 + 1.12011 w_k1_DIS_s116
                                 + 1.13753 w_k1_FDX_s116
                                 + 0.747333 w_k1_IBM_s116
                                 + 0.960282 w_k1_INTC_s116
                                 + 1.04312 w_k1_KO_s116
                                 + 0.766694 w_k1_MAT_s116
                                 + 1.10617 w_k1_MSFT_s116
                                 + 0.930608 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.19026 w_k1_ADM_s116
                                 + 1.15855 w_k1_AXP_s116
                                 + 1.57173 w_k1_DIS_s116
                                 + 1.39831 w_k1_FDX_s116
                                 + 1.23032 w_k1_IBM_s116
                                 + 1.10726 w_k1_INTC_s116
                                 + 1.37661 w_k1_KO_s116 + 1.18403 w_k1_MAT_s116
                                 + 1.24383 w_k1_MSFT_s116
                                 + 1.24991 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.11433 w_k1_ADM_s116
                                 + 0.864567 w_k1_AXP_s116
                                 + 1.25585 w_k1_DIS_s116
                                 + 0.990801 w_k1_FDX_s116
                                 + 1.6619 w_k1_IBM_s116
                                 + 1.05044 w_k1_INTC_s116
                                 + 1.34576 w_k1_KO_s116 + 1.18368 w_k1_MAT_s116
                                 + 1.08588 w_k1_MSFT_s116
                                 + 1.18922 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.24385 w_k1_ADM_s116
                                 + 1.41905 w_k1_AXP_s116
                                 + 1.22836 w_k1_DIS_s116
                                 + 1.22179 w_k1_FDX_s116
                                 + 1.25175 w_k1_IBM_s116
                                 + 1.02956 w_k1_INTC_s116
                                 + 1.41788 w_k1_KO_s116 + 1.81392 w_k1_MAT_s116
                                 + 1.31677 w_k1_MSFT_s116
                                 + 1.15119 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.852522 w_k1_ADM_s116
                                 + 0.87465 w_k1_AXP_s116
                                 + 1.09257 w_k1_DIS_s116
                                 + 1.07152 w_k1_FDX_s116
                                 + 0.984071 w_k1_IBM_s116
                                 + 1.01495 w_k1_INTC_s116
                                 + 1.11482 w_k1_KO_s116 + 1.01269 w_k1_MAT_s116
                                 + 1.15906 w_k1_MSFT_s116
                                 + 1.03329 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.839134 w_k1_ADM_s117
                                 + 0.949258 w_k1_AXP_s117
                                 + 0.862263 w_k1_DIS_s117
                                 + 0.894932 w_k1_FDX_s117
                                 + 0.943203 w_k1_IBM_s117
                                 + 0.921857 w_k1_INTC_s117
                                 + 1.08044 w_k1_KO_s117 + 1.15738 w_k1_MAT_s117
                                 + 0.965044 w_k1_MSFT_s117
                                 + 1.02185 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.847931 w_k1_ADM_s117
                                 + 1.19422 w_k1_AXP_s117
                                 + 1.17864 w_k1_DIS_s117
                                 + 1.02075 w_k1_FDX_s117
                                 + 1.57239 w_k1_IBM_s117
                                 + 1.07173 w_k1_INTC_s117
                                 + 1.35935 w_k1_KO_s117 + 1.80235 w_k1_MAT_s117
                                 + 1.02369 w_k1_MSFT_s117
                                 + 1.17819 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.837254 w_k1_ADM_s117
                                 + 0.69204 w_k1_AXP_s117
                                 + 0.68194 w_k1_DIS_s117
                                 + 0.832825 w_k1_FDX_s117
                                 + 0.627784 w_k1_IBM_s117
                                 + 0.990426 w_k1_INTC_s117
                                 + 0.731298 w_k1_KO_s117
                                 + 0.511105 w_k1_MAT_s117
                                 + 0.875824 w_k1_MSFT_s117
                                 + 0.890614 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.00695 w_k1_ADM_s117
                                 + 1.04316 w_k1_AXP_s117
                                 + 0.92295 w_k1_DIS_s117
                                 + 1.08004 w_k1_FDX_s117
                                 + 0.686471 w_k1_IBM_s117
                                 + 1.1133 w_k1_INTC_s117
                                 + 0.995864 w_k1_KO_s117
                                 + 1.09161 w_k1_MAT_s117
                                 + 1.13441 w_k1_MSFT_s117
                                 + 1.08277 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.21575 w_k1_ADM_s117
                                 + 1.27579 w_k1_AXP_s117
                                 + 1.34656 w_k1_DIS_s117 + 1.324 w_k1_FDX_s117
                                 + 1.09676 w_k1_IBM_s117
                                 + 1.26936 w_k1_INTC_s117
                                 + 1.17215 w_k1_KO_s117 + 1.25529 w_k1_MAT_s117
                                 + 1.4161 w_k1_MSFT_s117 + 1.21795 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.997153 w_k1_ADM_s117
                                 + 1.19852 w_k1_AXP_s117
                                 + 0.984518 w_k1_DIS_s117
                                 + 1.07813 w_k1_FDX_s117
                                 + 1.07871 w_k1_IBM_s117
                                 + 1.11635 w_k1_INTC_s117
                                 + 1.08674 w_k1_KO_s117 + 1.33908 w_k1_MAT_s117
                                 + 1.02158 w_k1_MSFT_s117
                                 + 1.17639 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.590942 w_k1_ADM_s117
                                 + 0.719606 w_k1_AXP_s117
                                 + 0.604301 w_k1_DIS_s117
                                 + 0.689927 w_k1_FDX_s117
                                 + 0.840539 w_k1_IBM_s117
                                 + 0.952551 w_k1_INTC_s117
                                 + 0.714881 w_k1_KO_s117
                                 + 0.427625 w_k1_MAT_s117
                                 + 0.737134 w_k1_MSFT_s117
                                 + 0.804237 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.13324 w_k1_ADM_s117
                                 + 1.28451 w_k1_AXP_s117
                                 + 1.72086 w_k1_DIS_s117
                                 + 1.19299 w_k1_FDX_s117
                                 + 1.48648 w_k1_IBM_s117
                                 + 1.13863 w_k1_INTC_s117 + 1.4459 w_k1_KO_s117
                                 + 1.52267 w_k1_MAT_s117
                                 + 1.2471 w_k1_MSFT_s117 + 1.37315 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.00949524031416525 w_k1_ADM_s000
                                 + 0.0136387565015734 w_k1_ADM_s001
                                 + 0.0133215126144629 w_k1_ADM_s002
                                 + 0.0155356609194531 w_k1_ADM_s003
                                 + 0.0228813628276607 w_k1_ADM_s004
                                 + 0.00653645141696546 w_k1_ADM_s005
                                 + 0.0273484889297657 w_k1_ADM_s006
                                 + 0.0118752911364127 w_k1_ADM_s007
                                 + 0.00970004511399455 w_k1_ADM_s010
                                 + 0.00801103654130278 w_k1_ADM_s011
                                 + 0.01478412132257 w_k1_ADM_s012
                                 + 0.0139325038398512 w_k1_ADM_s013
                                 + 0.00594373108081517 w_k1_ADM_s014
                                 + 0.010852332752859 w_k1_ADM_s015
                                 + 0.00448649895186437 w_k1_ADM_s016
                                 + 0.00927656374463308 w_k1_ADM_s017
                                 + 0.0134121710566952 w_k1_ADM_s100
                                 + 0.0266375112407018 w_k1_ADM_s101
                                 + 0.0332950831863758 w_k1_ADM_s102
                                 + 0.0224583146206385 w_k1_ADM_s103
                                 + 0.032591119631111 w_k1_ADM_s104
                                 + 0.0412770569362365 w_k1_ADM_s105
                                 + 0.0125951249148918 w_k1_ADM_s106
                                 + 0.0280608968549542 w_k1_ADM_s107
                                 + 0.0924753092583089 w_k1_ADM_s110
                                 + 0.0611313659237908 w_k1_ADM_s111
                                 + 0.0645707827818882 w_k1_ADM_s112
                                 + 0.152170314237552 w_k1_ADM_s113
                                 + 0.0839462993219347 w_k1_ADM_s114
                                 + 0.0706056681547415 w_k1_ADM_s115
                                 + 0.0543253138691742 w_k1_ADM_s116
                                 + 0.0300642935324057 w_k1_ADM_s117
                                 + 0.00944940518975579 w_k1_AXP_s000
                                 + 0.0142486659808017 w_k1_AXP_s001
                                 + 0.014471730350309 w_k1_AXP_s002
                                 + 0.0157671003666136 w_k1_AXP_s003
                                 + 0.0193155770205782 w_k1_AXP_s004
                                 + 0.00621463781197986 w_k1_AXP_s005
                                 + 0.0294869142968652 w_k1_AXP_s006
                                 + 0.0125400149695625 w_k1_AXP_s007
                                 + 0.00930953634320769 w_k1_AXP_s010
                                 + 0.00632137476254132 w_k1_AXP_s011
                                 + 0.0168063471153918 w_k1_AXP_s012
                                 + 0.0130516385410458 w_k1_AXP_s013
                                 + 0.00633669755943692 w_k1_AXP_s014
                                 + 0.0123633830593684 w_k1_AXP_s015
                                 + 0.00444198614417251 w_k1_AXP_s016
                                 + 0.00806446172357575 w_k1_AXP_s017
                                 + 0.0133402034910513 w_k1_AXP_s100
                                 + 0.0308811939929026 w_k1_AXP_s101
                                 + 0.0330389704679833 w_k1_AXP_s102
                                 + 0.0244600552866377 w_k1_AXP_s103
                                 + 0.0312200240019272 w_k1_AXP_s104
                                 + 0.0387497087621152 w_k1_AXP_s105
                                 + 0.0142721550712591 w_k1_AXP_s106
                                 + 0.0231264081244392 w_k1_AXP_s107
                                 + 0.0805290274392274 w_k1_AXP_s110
                                 + 0.0640286806266473 w_k1_AXP_s111
                                 + 0.0610043416866315 w_k1_AXP_s112
                                 + 0.148216333931512 w_k1_AXP_s113
                                 + 0.0896316412715008 w_k1_AXP_s114
                                 + 0.0758055288717015 w_k1_AXP_s115
                                 + 0.0516297869238585 w_k1_AXP_s116
                                 + 0.0333032245962958 w_k1_AXP_s117
                                 + 0.00904163126647597 w_k1_DIS_s000
                                 + 0.0138305983459096 w_k1_DIS_s001
                                 + 0.0131307006532937 w_k1_DIS_s002
                                 + 0.0157584542481001 w_k1_DIS_s003
                                 + 0.0221408313895742 w_k1_DIS_s004
                                 + 0.00605429960506401 w_k1_DIS_s005
                                 + 0.0296253046774566 w_k1_DIS_s006
                                 + 0.0115478263520211 w_k1_DIS_s007
                                 + 0.00926791939077017 w_k1_DIS_s010
                                 + 0.00713817720470108 w_k1_DIS_s011
                                 + 0.0154319094424025 w_k1_DIS_s012
                                 + 0.0121640280969021 w_k1_DIS_s013
                                 + 0.00602142301389042 w_k1_DIS_s014
                                 + 0.0125505959006385 w_k1_DIS_s015
                                 + 0.00474472887209767 w_k1_DIS_s016
                                 + 0.00865732347756244 w_k1_DIS_s017
                                 + 0.0126429975416987 w_k1_DIS_s100
                                 + 0.0302388876784063 w_k1_DIS_s101
                                 + 0.0314414897417904 w_k1_DIS_s102
                                 + 0.0258844025714871 w_k1_DIS_s103
                                 + 0.0317988852828615 w_k1_DIS_s104
                                 + 0.038299668218225 w_k1_DIS_s105
                                 + 0.0138652121693903 w_k1_DIS_s106
                                 + 0.026730616980247 w_k1_DIS_s107
                                 + 0.0773720189368336 w_k1_DIS_s110
                                 + 0.0661941507804037 w_k1_DIS_s111
                                 + 0.0730703077540265 w_k1_DIS_s112
                                 + 0.151431567874991 w_k1_DIS_s113
                                 + 0.0898971670481055 w_k1_DIS_s114
                                 + 0.0736128087776578 w_k1_DIS_s115
                                 + 0.057338030829733 w_k1_DIS_s116
                                 + 0.0324082688943131 w_k1_DIS_s117
                                 + 0.00904013769618164 w_k1_FDX_s000
                                 + 0.0138642543176119 w_k1_FDX_s001
                                 + 0.0131723138836245 w_k1_FDX_s002
                                 + 0.0148951357719188 w_k1_FDX_s003
                                 + 0.0221720008068417 w_k1_FDX_s004
                                 + 0.00609763008617619 w_k1_FDX_s005
                                 + 0.0287473592585094 w_k1_FDX_s006
                                 + 0.011324285045496 w_k1_FDX_s007
                                 + 0.00940451254763432 w_k1_FDX_s010
                                 + 0.00662668243117327 w_k1_FDX_s011
                                 + 0.0159662211395735 w_k1_FDX_s012
                                 + 0.0122795933311721 w_k1_FDX_s013
                                 + 0.00610567410590307 w_k1_FDX_s014
                                 + 0.0124428537791119 w_k1_FDX_s015
                                 + 0.00463110698385877 w_k1_FDX_s016
                                 + 0.00824601740636166 w_k1_FDX_s017
                                 + 0.0126128357192158 w_k1_FDX_s100
                                 + 0.0294028799623791 w_k1_FDX_s101
                                 + 0.031639645322396 w_k1_FDX_s102
                                 + 0.0250031176523954 w_k1_FDX_s103
                                 + 0.0317772940643033 w_k1_FDX_s104
                                 + 0.0384182879810263 w_k1_FDX_s105
                                 + 0.0140160805546921 w_k1_FDX_s106
                                 + 0.0257253939813232 w_k1_FDX_s107
                                 + 0.0734880750724727 w_k1_FDX_s110
                                 + 0.0646299081070585 w_k1_FDX_s111
                                 + 0.0654304513550561 w_k1_FDX_s112
                                 + 0.152289549638766 w_k1_FDX_s113
                                 + 0.0925904265851868 w_k1_FDX_s114
                                 + 0.0790705401946756 w_k1_FDX_s115
                                 + 0.0580356915215176 w_k1_FDX_s116
                                 + 0.0324468423184761 w_k1_FDX_s117
                                 + 0.0100981665295835 w_k1_IBM_s000
                                 + 0.0137178210177303 w_k1_IBM_s001
                                 + 0.0131226117128822 w_k1_IBM_s002
                                 + 0.0153519175505149 w_k1_IBM_s003
                                 + 0.0219651087417767 w_k1_IBM_s004
                                 + 0.00576072341648513 w_k1_IBM_s005
                                 + 0.0279936102020158 w_k1_IBM_s006
                                 + 0.0109877020822676 w_k1_IBM_s007
                                 + 0.0093243668012783 w_k1_IBM_s010
                                 + 0.00670229537257137 w_k1_IBM_s011
                                 + 0.0163676650662587 w_k1_IBM_s012
                                 + 0.0124428154474103 w_k1_IBM_s013
                                 + 0.00645479748305972 w_k1_IBM_s014
                                 + 0.0127536828335972 w_k1_IBM_s015
                                 + 0.00477014579972677 w_k1_IBM_s016
                                 + 0.00822189192224918 w_k1_IBM_s017
                                 + 0.0130405881395665 w_k1_IBM_s100
                                 + 0.032258498528256 w_k1_IBM_s101
                                 + 0.0304070435315147 w_k1_IBM_s102
                                 + 0.0237840396182255 w_k1_IBM_s103
                                 + 0.0342946683787562 w_k1_IBM_s104
                                 + 0.0370893904449751 w_k1_IBM_s105
                                 + 0.0144802288093431 w_k1_IBM_s106
                                 + 0.0240877564416515 w_k1_IBM_s107
                                 + 0.0750936244773891 w_k1_IBM_s110
                                 + 0.0692441117476363 w_k1_IBM_s111
                                 + 0.0668543517051931 w_k1_IBM_s112
                                 + 0.162178484096058 w_k1_IBM_s113
                                 + 0.0946577018282965 w_k1_IBM_s114
                                 + 0.0769819368986411 w_k1_IBM_s115
                                 + 0.0550326166379749 w_k1_IBM_s116
                                 + 0.0328062829182289 w_k1_IBM_s117
                                 + 0.0101660022565805 w_k1_INTC_s000
                                 + 0.0153223306655223 w_k1_INTC_s001
                                 + 0.0122734561308696 w_k1_INTC_s002
                                 + 0.0134693802572995 w_k1_INTC_s003
                                 + 0.0202494663025713 w_k1_INTC_s004
                                 + 0.00621189099935531 w_k1_INTC_s005
                                 + 0.0265651376934507 w_k1_INTC_s006
                                 + 0.00936527876659908 w_k1_INTC_s007
                                 + 0.0090798898929468 w_k1_INTC_s010
                                 + 0.00679477871384338 w_k1_INTC_s011
                                 + 0.0147282821043081 w_k1_INTC_s012
                                 + 0.0112699555346561 w_k1_INTC_s013
                                 + 0.00625387705426188 w_k1_INTC_s014
                                 + 0.0104532363672469 w_k1_INTC_s015
                                 + 0.00403643869107632 w_k1_INTC_s016
                                 + 0.00927772687221845 w_k1_INTC_s017
                                 + 0.0134433927385074 w_k1_INTC_s100
                                 + 0.0281923524357986 w_k1_INTC_s101
                                 + 0.032349217684414 w_k1_INTC_s102
                                 + 0.0281605208954034 w_k1_INTC_s103
                                 + 0.0337133663105456 w_k1_INTC_s104
                                 + 0.0362919460826217 w_k1_INTC_s105
                                 + 0.0152428106232997 w_k1_INTC_s106
                                 + 0.0236357843729494 w_k1_INTC_s107
                                 + 0.0686731917776172 w_k1_INTC_s110
                                 + 0.0689466174927538 w_k1_INTC_s111
                                 + 0.0612861590636285 w_k1_INTC_s112
                                 + 0.151158368569077 w_k1_INTC_s113
                                 + 0.0900526741927251 w_k1_INTC_s114
                                 + 0.0772924621161236 w_k1_INTC_s115
                                 + 0.050431498432684 w_k1_INTC_s116
                                 + 0.034311075470777 w_k1_INTC_s117
                                 + 0.0103941354264569 w_k1_KO_s000
                                 + 0.0135475947324071 w_k1_KO_s001
                                 + 0.0138207418541029 w_k1_KO_s002
                                 + 0.0142163036614411 w_k1_KO_s003
                                 + 0.0221873084715687 w_k1_KO_s004
                                 + 0.00560390653101467 w_k1_KO_s005
                                 + 0.0278564267969346 w_k1_KO_s006
                                 + 0.0108052068505113 w_k1_KO_s007
                                 + 0.00892306436183591 w_k1_KO_s010
                                 + 0.00684164842334158 w_k1_KO_s011
                                 + 0.0141036813664378 w_k1_KO_s012
                                 + 0.0116852915778284 w_k1_KO_s013
                                 + 0.00599140068435655 w_k1_KO_s014
                                 + 0.0101255426403953 w_k1_KO_s015
                                 + 0.00436114447878982 w_k1_KO_s016
                                 + 0.00815224894919265 w_k1_KO_s017
                                 + 0.0132870738598131 w_k1_KO_s100
                                 + 0.031176808484722 w_k1_KO_s101
                                 + 0.0302054237799633 w_k1_KO_s102
                                 + 0.0263221078815692 w_k1_KO_s103
                                 + 0.0299232743229555 w_k1_KO_s104
                                 + 0.0356141812357159 w_k1_KO_s105
                                 + 0.0143726393478906 w_k1_KO_s106
                                 + 0.0246555461235921 w_k1_KO_s107
                                 + 0.0804069723249777 w_k1_KO_s110
                                 + 0.0705297249214207 w_k1_KO_s111
                                 + 0.067820419169236 w_k1_KO_s112
                                 + 0.150372245673261 w_k1_KO_s113
                                 + 0.0912257861758667 w_k1_KO_s114
                                 + 0.0756292704392704 w_k1_KO_s115
                                 + 0.0588627645131765 w_k1_KO_s116
                                 + 0.0343169680935161 w_k1_KO_s117
                                 + 0.00881978925067673 w_k1_MAT_s000
                                 + 0.0129558261991434 w_k1_MAT_s001
                                 + 0.0167219165912263 w_k1_MAT_s002
                                 + 0.0143122408069836 w_k1_MAT_s003
                                 + 0.0214653345449856 w_k1_MAT_s004
                                 + 0.00586546715748094 w_k1_MAT_s005
                                 + 0.0283132989402686 w_k1_MAT_s006
                                 + 0.0160106483894102 w_k1_MAT_s007
                                 + 0.0107969008661047 w_k1_MAT_s010
                                 + 0.0058388463271182 w_k1_MAT_s011
                                 + 0.017085586443742 w_k1_MAT_s012
                                 + 0.0127378336787465 w_k1_MAT_s013
                                 + 0.00486592654796332 w_k1_MAT_s014
                                 + 0.0113964908119828 w_k1_MAT_s015
                                 + 0.00452006281252245 w_k1_MAT_s016
                                 + 0.00778094458750745 w_k1_MAT_s017
                                 + 0.0119694949420193 w_k1_MAT_s100
                                 + 0.0342779171117146 w_k1_MAT_s101
                                 + 0.0269446687465311 w_k1_MAT_s102
                                 + 0.0232624760567244 w_k1_MAT_s103
                                 + 0.030266794508792 w_k1_MAT_s104
                                 + 0.0360081946416565 w_k1_MAT_s105
                                 + 0.0160568028919336 w_k1_MAT_s106
                                 + 0.0236175295128622 w_k1_MAT_s107
                                 + 0.0867967140570803 w_k1_MAT_s110
                                 + 0.0652662757295695 w_k1_MAT_s111
                                 + 0.0598624361314275 w_k1_MAT_s112
                                 + 0.142904685601633 w_k1_MAT_s113
                                 + 0.0902308809747404 w_k1_MAT_s114
                                 + 0.069842175829783 w_k1_MAT_s115
                                 + 0.0574451280600087 w_k1_MAT_s116
                                 + 0.0365516533734111 w_k1_MAT_s117
                                 + 0.00912909769783155 w_k1_MSFT_s000
                                 + 0.0149975783672508 w_k1_MSFT_s001
                                 + 0.0131166089129949 w_k1_MSFT_s002
                                 + 0.0138184964745556 w_k1_MSFT_s003
                                 + 0.0212225937217459 w_k1_MSFT_s004
                                 + 0.00669955144133157 w_k1_MSFT_s005
                                 + 0.0296815264417636 w_k1_MSFT_s006
                                 + 0.0106637483522067 w_k1_MSFT_s007
                                 + 0.00994678110762762 w_k1_MSFT_s010
                                 + 0.00705722995770736 w_k1_MSFT_s011
                                 + 0.0153111998007692 w_k1_MSFT_s012
                                 + 0.0125624773449749 w_k1_MSFT_s013
                                 + 0.00563452991074721 w_k1_MSFT_s014
                                 + 0.0116110159412499 w_k1_MSFT_s015
                                 + 0.00414736691153264 w_k1_MSFT_s016
                                 + 0.00845775190582293 w_k1_MSFT_s017
                                 + 0.013456744931582 w_k1_MSFT_s100
                                 + 0.0286652942255102 w_k1_MSFT_s101
                                 + 0.0326179282937697 w_k1_MSFT_s102
                                 + 0.0270315225074852 w_k1_MSFT_s103
                                 + 0.0326163952087515 w_k1_MSFT_s104
                                 + 0.0358220874836374 w_k1_MSFT_s105
                                 + 0.0138673553551372 w_k1_MSFT_s106
                                 + 0.0268720924848745 w_k1_MSFT_s107
                                 + 0.0800410767036137 w_k1_MSFT_s110
                                 + 0.0691635664304713 w_k1_MSFT_s111
                                 + 0.0736098848609816 w_k1_MSFT_s112
                                 + 0.14627241625285 w_k1_MSFT_s113
                                 + 0.0874161961698237 w_k1_MSFT_s114
                                 + 0.0771913526927204 w_k1_MSFT_s115
                                 + 0.0574413013218588 w_k1_MSFT_s116
                                 + 0.0335909489082509 w_k1_MSFT_s117
                                 + 0.00965602782868333 w_k1_PG_s000
                                 + 0.014342937513605 w_k1_PG_s001
                                 + 0.0141626671591534 w_k1_PG_s002
                                 + 0.0140644208228736 w_k1_PG_s003
                                 + 0.0203551311643137 w_k1_PG_s004
                                 + 0.00603059231168785 w_k1_PG_s005
                                 + 0.0289799947441056 w_k1_PG_s006
                                 + 0.0114262817318969 w_k1_PG_s007
                                 + 0.00884321070777066 w_k1_PG_s010
                                 + 0.00661790369683619 w_k1_PG_s011
                                 + 0.0164134252712292 w_k1_PG_s012
                                 + 0.0131833507517087 w_k1_PG_s013
                                 + 0.00559745202534406 w_k1_PG_s014
                                 + 0.0115253324819724 w_k1_PG_s015
                                 + 0.00431656184899507 w_k1_PG_s016
                                 + 0.00850248348933216 w_k1_PG_s017
                                 + 0.0120135901998752 w_k1_PG_s100
                                 + 0.0306506763358613 w_k1_PG_s101
                                 + 0.0314899221306467 w_k1_PG_s102
                                 + 0.0248031848150126 w_k1_PG_s103
                                 + 0.0314832544203293 w_k1_PG_s104
                                 + 0.0388729726236422 w_k1_PG_s105
                                 + 0.013923189257703 w_k1_PG_s106
                                 + 0.0242710400241152 w_k1_PG_s107
                                 + 0.0722282950813617 w_k1_PG_s110
                                 + 0.0660945842290818 w_k1_PG_s111
                                 + 0.0663960980821072 w_k1_PG_s112
                                 + 0.14677485620122 w_k1_PG_s113
                                 + 0.0843895265484241 w_k1_PG_s114
                                 + 0.0727679450879439 w_k1_PG_s115
                                 + 0.055907853363389 w_k1_PG_s116
                                 + 0.0350981805450252 w_k1_PG_s117 >= 1.2
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
