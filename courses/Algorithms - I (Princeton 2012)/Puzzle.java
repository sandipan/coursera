/**
 * File: Puzzle.java
 * Author: Brian Borowski
 * Date created: March 2000
 * Date last modified: May 3, 2011
 */
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;

import javax.swing.JPanel;

public class Puzzle extends JPanel {
    private static final long serialVersionUID = 1L;
    private Image[] imageArray;
    private Image movingImage;
    private int numOfTiles, dimension, imageWidth, imageHeight, panelWidth, panelHeight,
                r0, r1, c0, c1, x, y, movingCoord, sleepMs;
    private boolean animationDone = true;
    private byte[] state;

    public Puzzle(final int numOfTiles) {
        super();
        setNumOfTiles(numOfTiles);
        setSize(panelWidth, panelHeight);
        setDoubleBuffered(true);
    }

    public void setNumOfTiles(final int numOfTiles) {
        this.numOfTiles = numOfTiles;
        dimension = (int)Math.sqrt(numOfTiles);
        imageArray = new Image[numOfTiles];
        final String file = "images/" + (numOfTiles - 1) + "-puzzle/shrksign_";
        imageArray[0] = null;
        for (int i = 1; i < numOfTiles; ++i) {
            final StringBuilder builder = new StringBuilder(file);
            if (i <= 9) {
                builder.append("0");
            }
            builder.append(Integer.toString(i));
            builder.append(".gif");
            imageArray[i] = ImagePanel.getImage(builder.toString());
        }
        imageWidth  = imageArray[1].getWidth(null);
        imageHeight = imageArray[1].getHeight(null);
        panelWidth  = imageWidth * dimension;
        panelHeight = imageHeight * dimension;
        state = new byte[numOfTiles];
        sleepMs = 500 / imageWidth;
        animationDone = true;
        resetOrder();
    }

    public void resetOrder() {
        for (int i = 0; i < numOfTiles - 1; ++i) {
            state[i] = (byte)(i + 1);
        }
        state[numOfTiles - 1] = 0;
        repaint();
    }

    public void setOrder(final byte[] state) {
        System.arraycopy(state, 0, this.state, 0, state.length);
        repaint();
    }

    public void stopAnimation() {
        animationDone = true;
    }

    public void animatePuzzleTo(final byte[] newState) {
        int newPosOfTile = 0, currentPosOfTile = 0;
        for (int i = 0; i < numOfTiles; ++i) {
            if (state[i] == 0) {
                newPosOfTile = i;
            }
            if (newState[i] == 0) {
                currentPosOfTile = i;
            }
        }

        r0 = newPosOfTile / dimension;
        c0 = newPosOfTile % dimension;
        r1 = currentPosOfTile / dimension;
        c1 = currentPosOfTile % dimension;
        movingImage = imageArray[state[currentPosOfTile]];
        System.arraycopy(newState, 0, state, 0, newState.length);

        if (r0 == r1) {
            movingCoord = c1 * imageWidth;
        } else {
            movingCoord = r1 * imageHeight;
        }

        final Thread t = new Thread(new Runnable() {
            public void run() {
                x = c0 * imageWidth;
                y = r0 * imageHeight;
                animationDone = false;
                while (!animationDone) {
                    if (r0 == r1) {
                        if (c1 < c0) {
                            if (movingCoord <= x) {
                                ++movingCoord;
                            } else {
                                animationDone = true;
                            }
                        } else {
                            if (movingCoord >= x) {
                                --movingCoord;
                            } else {
                                animationDone = true;
                            }
                        }
                    } else {
                        if (r1 < r0) {
                            if (movingCoord < y) {
                                ++movingCoord;
                            } else {
                                animationDone = true;
                            }
                        } else {
                            if (movingCoord >= y) {
                                --movingCoord;
                            } else {
                                animationDone = true;
                            }
                        }
                    }
                    repaint();
                    try {
                        Thread.sleep(sleepMs);
                    } catch (final InterruptedException ie) { }
                }
            }
        });

        t.start();
        try {
            t.join();
        } catch (final InterruptedException ie) { }
    }

    public Dimension getPreferredSize() {
        return new Dimension(panelWidth, panelHeight);
    }

    protected void paintComponent(final Graphics g) {
        int pos = 0;
        for (int row = 0; row < dimension; ++row) {
            for (int col = 0; col < dimension; ++col) {
                if (state[pos] != 0) {
                    g.setColor(Color.blue);
                    g.fill3DRect(
                        col * imageWidth, row * imageHeight, imageWidth, imageHeight, true);
                    final Image image = imageArray[state[pos]];
                    g.drawImage(image, col * imageWidth, row * imageHeight, imageWidth,
                        imageHeight, null);
                } else {
                    g.setColor(Color.black);
                    g.fillRect(col * imageWidth, row * imageHeight, imageWidth, imageHeight);
                }
                ++pos;
            }
        }
        if (!animationDone) {
            paintMovingTileRegion(g);
        }
    }

    private void paintMovingTileRegion(final Graphics g) {
        g.setColor(Color.black);
        g.fillRect(x, y, imageWidth, imageHeight);
        g.fillRect(c1 * imageHeight, r1 * imageWidth, imageWidth, imageHeight);
        g.setColor(Color.blue);

        if (r0 == r1) {
            g.fill3DRect(movingCoord, y, imageWidth, imageHeight, true);
            g.drawImage(movingImage, movingCoord, y, imageWidth, imageHeight, null);
        } else {
            g.fill3DRect(x, movingCoord, imageWidth, imageHeight, true);
            g.drawImage(movingImage, x, movingCoord, imageWidth, imageHeight, null);
        }
    }
}
